"""
fragment_matching.py - Finding Matches Between Fragment Break Surfaces
======================================================================
Given multiple fragments with extracted break-surface features, finds
candidate matching pairs and their point-level correspondences.

Pipeline:
  1. Pairwise compatibility screening using summary statistics (fast)
  2. FPFH descriptor matching with mutual nearest-neighbour + ratio test
  3. RANSAC geometric validation → initial rigid transformation
  4. Geometric match scoring (inlier ratio, coverage, normal compatibility, RMS)
  5. Optional ML re-ranking via a trained logistic regression scorer

Usage:
    python fragment_matching.py                    # all *_features.npz in results/
    python fragment_matching.py results/frag1_features.npz results/frag2_features.npz
    python fragment_matching.py --ml_model results/ml_scorer.joblib
"""
import os
import sys
import glob
import argparse
import itertools

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

import config


# ── tunable parameters ────────────────────────────────────────────────────────
MNN_RATIO          = 0.9     # Lowe ratio-test threshold (mutual NN in feature space)
RANSAC_DISTANCE    = 2.0     # Max correspondence distance for RANSAC inlier check
RANSAC_N           = 3       # Minimum sample size for RANSAC
RANSAC_ITER        = 50_000  # Max RANSAC iterations
MIN_INLIERS        = 10      # Minimum inliers to declare a valid match


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1  Fast compatibility screening
# ──────────────────────────────────────────────────────────────────────────────

def fragment_compatibility_score(summary_a, summary_b):
    """
    Compute a quick compatibility score [0, 1] from fragment summary statistics.
    Higher = more likely to be a true matching pair.

    Criteria:
    - Normal anti-parallelism: break-surface normals should be roughly opposing
    - Size compatibility: break regions should have similar area/extent
    - Roughness similarity: similar erosion levels suggest same stone
    - Curvature similarity: same carving / texture style

    Returns:
        score:   float in [0, 1]
        details: dict of individual component scores
    """
    components = []  # (name, score, weight)

    na = np.array(summary_a.get('mean_normal', [0.0, 0.0, 1.0]))
    nb = np.array(summary_b.get('mean_normal', [0.0, 0.0, 1.0]))
    # Anti-parallel → dot ≈ -1 → map to [0, 1] where 1 = perfectly anti-parallel
    normal_score = (-float(na @ nb) + 1.0) / 2.0
    components.append(('normal_antiparallel', normal_score, 0.40))

    area_a = summary_a.get('surface_area_proxy', 0.0)
    area_b = summary_b.get('surface_area_proxy', 0.0)
    if area_a > 0 and area_b > 0:
        size_score = min(area_a, area_b) / max(area_a, area_b)
        components.append(('size_ratio', size_score, 0.25))

    ra, rb = summary_a.get('roughness', 0.0), summary_b.get('roughness', 0.0)
    if ra + rb > 1e-6:
        rough_score = 1.0 - abs(ra - rb) / (ra + rb + 1e-10)
        components.append(('roughness_sim', rough_score, 0.15))

    ca = summary_a.get('mean_curvature', 0.0)
    cb = summary_b.get('mean_curvature', 0.0)
    curv_score = 1.0 - min(abs(ca - cb) / (ca + cb + 1e-10), 1.0)
    components.append(('curvature_sim', curv_score, 0.20))

    total_w = sum(w for _, _, w in components)
    score   = sum(s * w for _, s, w in components) / total_w

    details = {n: s for n, s, _ in components}
    details['overall'] = score
    return score, details


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2  FPFH descriptor matching
# ──────────────────────────────────────────────────────────────────────────────

def match_fpfh_descriptors(fpfh_a, fpfh_b, ratio=MNN_RATIO):
    """
    Mutual nearest-neighbour matching in FPFH feature space with ratio test.

    Args:
        fpfh_a: (M, 33) FPFH descriptors for fragment A break surface
        fpfh_b: (N, 33) FPFH descriptors for fragment B break surface
        ratio:  Lowe ratio-test threshold

    Returns:
        corr_a: (K,) indices into fpfh_a of accepted correspondences
        corr_b: (K,) indices into fpfh_b of accepted correspondences
    """
    if len(fpfh_a) == 0 or len(fpfh_b) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    eps = 1e-10
    na  = fpfh_a / (np.linalg.norm(fpfh_a, axis=1, keepdims=True) + eps)
    nb  = fpfh_b / (np.linalg.norm(fpfh_b, axis=1, keepdims=True) + eps)

    tree_b = cKDTree(nb)
    tree_a = cKDTree(na)

    k_query = min(2, len(nb))
    dist_a2b, idx_a2b = tree_b.query(na, k=k_query)
    dist_b2a, idx_b2a = tree_a.query(nb, k=1)

    corr_a, corr_b = [], []
    for i in range(len(na)):
        # ratio test (only meaningful when we have at least 2 neighbours)
        if k_query >= 2:
            if dist_a2b[i, 0] >= ratio * dist_a2b[i, 1]:
                continue
        j = int(idx_a2b[i, 0])
        # mutual nearest-neighbour check
        if int(idx_b2a[j]) == i:
            corr_a.append(i)
            corr_b.append(j)

    return np.array(corr_a, np.int64), np.array(corr_b, np.int64)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3  RANSAC geometric validation
# ──────────────────────────────────────────────────────────────────────────────

def ransac_match(pts_a, pts_b, corr_a, corr_b,
                 dist_threshold=RANSAC_DISTANCE,
                 n_iterations=RANSAC_ITER):
    """
    RANSAC geometric consistency check via Open3D.

    Args:
        pts_a, pts_b: (N, 3) break-surface point arrays
        corr_a, corr_b: putative correspondence index arrays
        dist_threshold: inlier distance threshold
        n_iterations:   max RANSAC iterations

    Returns:
        T:         (4, 4) best rigid transformation  (A → B)
        inlier_a:  inlier indices into pts_a
        inlier_b:  inlier indices into pts_b
        n_inliers: number of inliers
    """
    if len(corr_a) < RANSAC_N:
        return np.eye(4), np.empty(0, np.int64), np.empty(0, np.int64), 0

    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(pts_a)
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(pts_b)

    corr_o3d = o3d.utility.Vector2iVector(
        np.stack([corr_a, corr_b], axis=1))

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd_a, pcd_b, corr_o3d,
        max_correspondence_distance=dist_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=RANSAC_N,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(n_iterations, 0.999))

    T = np.array(result.transformation)

    # Degenerate result – RANSAC failed
    if np.allclose(T, np.eye(4)):
        return T, np.empty(0, np.int64), np.empty(0, np.int64), 0

    # Recompute inlier correspondences under T
    pts_a_tf  = (T[:3, :3] @ pts_a[corr_a].T).T + T[:3, 3]
    dists     = np.linalg.norm(pts_a_tf - pts_b[corr_b], axis=1)
    inlier_m  = dists < dist_threshold

    return T, corr_a[inlier_m], corr_b[inlier_m], int(inlier_m.sum())


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4  Match scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_match(pts_a, pts_b, normals_a, normals_b,
                corr_a, corr_b, inlier_a, inlier_b, T):
    """
    Comprehensive geometric match quality score ∈ [0, 1].

    Components:
    - Inlier ratio        (inliers / putative correspondences)
    - Coverage            (inliers / break-surface size)
    - Normal compatibility (matched normals should be anti-parallel after T)
    - RMS alignment residual

    Returns:
        score:   float in [0, 1]
        details: dict of individual components
    """
    n_putative = len(corr_a)
    n_inliers  = len(inlier_a)
    details    = {'n_putative': n_putative, 'n_inliers': n_inliers}

    if n_putative == 0 or n_inliers < MIN_INLIERS:
        details['overall'] = 0.0
        return 0.0, details

    components = []

    # 1. Inlier ratio
    inlier_ratio = n_inliers / n_putative
    details['inlier_ratio'] = inlier_ratio
    components.append(('inlier_ratio', min(inlier_ratio * 2.0, 1.0), 0.25))

    # 2. Coverage: fraction of each break surface represented by inliers
    cov_a = n_inliers / max(len(pts_a), 1)
    cov_b = n_inliers / max(len(pts_b), 1)
    coverage = (cov_a + cov_b) / 2.0
    details['coverage'] = coverage
    components.append(('coverage', min(coverage * 5.0, 1.0), 0.20))

    # 3. Normal anti-parallelism at inlier correspondences
    R = T[:3, :3]
    if (normals_a is not None and normals_b is not None
            and len(normals_a) > 0 and len(inlier_a) > 0):
        nrm_a_rot = (R @ normals_a[inlier_a].T).T
        dots      = (nrm_a_rot * normals_b[inlier_b]).sum(axis=1)
        # anti-parallel → dot ~ -1 → score → [0,1] where 1 = perfect anti-parallel
        normal_compat = (-float(dots.mean()) + 1.0) / 2.0
        details['normal_compatibility'] = normal_compat
        components.append(('normal_compat', normal_compat, 0.25))

    # 4. RMS residual at inlier correspondences
    pts_a_tf  = (R @ pts_a[inlier_a].T).T + T[:3, 3]
    residuals = np.linalg.norm(pts_a_tf - pts_b[inlier_b], axis=1)
    rms       = float(np.sqrt((residuals ** 2).mean()))
    details['rms_residual'] = rms
    rms_score = max(0.0, 1.0 - rms / RANSAC_DISTANCE)
    components.append(('rms_score', rms_score, 0.30))

    total_w = sum(w for _, _, w in components)
    score   = sum(s * w for _, s, w in components) / total_w
    details['overall'] = score

    return score, details


# ──────────────────────────────────────────────────────────────────────────────
# STEP 5  ML re-ranking
# ──────────────────────────────────────────────────────────────────────────────

def build_match_feature_vector(compat_details, match_details):
    """Build a fixed-length (11-d) feature vector for ML scoring."""
    return np.array([
        compat_details.get('normal_antiparallel', 0.0),
        compat_details.get('size_ratio',          0.0),
        compat_details.get('roughness_sim',        0.0),
        compat_details.get('curvature_sim',        0.0),
        compat_details.get('overall',              0.0),
        match_details.get('inlier_ratio',          0.0),
        float(match_details.get('n_inliers',       0)),
        match_details.get('coverage',              0.0),
        match_details.get('normal_compatibility',  0.0),
        match_details.get('rms_residual',  RANSAC_DISTANCE),
        match_details.get('overall',               0.0),
    ], dtype=np.float32)


class MLMatchScorer:
    """
    Logistic regression re-ranker for match pairs.

    Train with labeled examples  (y=1 = true match, y=0 = false match).
    Falls back gracefully to the geometric score when not trained.

    Usage:
        scorer = MLMatchScorer()
        scorer.fit(X_train, y_train)
        scorer.save('results/ml_scorer.joblib')

        scorer = MLMatchScorer('results/ml_scorer.joblib')
        probs  = scorer.predict_proba(X)
    """

    def __init__(self, model_path=None):
        self.scaler  = StandardScaler()
        self.clf     = LogisticRegression(C=1.0, max_iter=1000,
                                          class_weight='balanced')
        self.trained = False
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def fit(self, X, y):
        X_s = self.scaler.fit_transform(X)
        self.clf.fit(X_s, y)
        self.trained = True

    def predict_proba(self, X):
        """Returns predicted match probability ∈ [0, 1] for each row of X."""
        if not self.trained:
            return None
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]

    def save(self, path):
        joblib.dump({'scaler': self.scaler, 'clf': self.clf,
                     'trained': self.trained}, path)

    def load(self, path):
        obj          = joblib.load(path)
        self.scaler  = obj['scaler']
        self.clf     = obj['clf']
        self.trained = obj['trained']


# ──────────────────────────────────────────────────────────────────────────────
# Full pair matching
# ──────────────────────────────────────────────────────────────────────────────

def match_fragment_pair(feat_a, feat_b, verbose=True):
    """
    Run the complete matching pipeline between two fragments.

    Args:
        feat_a, feat_b: dicts with keys
            name, break_pts, break_normals, fpfh, pca_features, summary

    Returns:
        result dict with all match information
    """
    name_a, name_b = feat_a['name'], feat_b['name']
    if verbose:
        print(f"\n  Matching: {name_a}  <->  {name_b}")

    # Step 1 – compatibility
    compat_score, compat_details = fragment_compatibility_score(
        feat_a['summary'], feat_b['summary'])
    if verbose:
        print(f"    Compatibility: {compat_score:.3f}  "
              f"(normal={compat_details.get('normal_antiparallel', 0):.3f}  "
              f"size={compat_details.get('size_ratio', 0):.3f})")

    # Step 2 – FPFH matching
    corr_a, corr_b = match_fpfh_descriptors(feat_a['fpfh'], feat_b['fpfh'])
    if verbose:
        print(f"    Putative correspondences: {len(corr_a)}")

    if len(corr_a) < RANSAC_N:
        return {
            'name_a': name_a, 'name_b': name_b,
            'score': 0.0, 'final_score': 0.0,
            'n_inliers': 0,
            'transformation': np.eye(4).tolist(),
            'compatibility': compat_score,
            'match_details': {'overall': 0.0, 'n_inliers': 0},
            'compat_details': compat_details,
            'correspondence_a': [], 'correspondence_b': [],
            'inlier_a': [], 'inlier_b': [],
            'feature_vector': build_match_feature_vector(
                compat_details, {'overall': 0.0}).tolist(),
        }

    # Step 3 – RANSAC
    T, inlier_a, inlier_b, n_inliers = ransac_match(
        feat_a['break_pts'], feat_b['break_pts'], corr_a, corr_b)
    if verbose:
        print(f"    RANSAC inliers: {n_inliers}")

    # Step 4 – scoring
    score, match_details = score_match(
        feat_a['break_pts'],     feat_b['break_pts'],
        feat_a['break_normals'], feat_b['break_normals'],
        corr_a, corr_b, inlier_a, inlier_b, T)

    if verbose:
        print(f"    Match score: {score:.3f}  "
              f"(inlier_ratio={match_details.get('inlier_ratio', 0):.3f}  "
              f"rms={match_details.get('rms_residual', 0):.3f})")

    fv = build_match_feature_vector(compat_details, match_details)

    return {
        'name_a': name_a, 'name_b': name_b,
        'score': score, 'final_score': score,
        'n_inliers': n_inliers,
        'transformation': T.tolist(),
        'compatibility': compat_score,
        'correspondence_a': corr_a.tolist(),
        'correspondence_b': corr_b.tolist(),
        'inlier_a': inlier_a.tolist(),
        'inlier_b': inlier_b.tolist(),
        'match_details': match_details,
        'compat_details': compat_details,
        'feature_vector': fv.tolist(),
    }


def match_all_fragments(feature_files, ml_scorer=None, verbose=True):
    """
    Match every pair of fragments and return a ranked list of candidates.

    Args:
        feature_files: list of *_features.npz paths
        ml_scorer:     optional MLMatchScorer for re-ranking
        verbose:       print progress

    Returns:
        list of match result dicts, sorted by final_score descending
    """
    # ── load fragments ────────────────────────────────────────────────────
    fragments = []
    for fp in feature_files:
        data = np.load(fp, allow_pickle=True)
        name = os.path.basename(fp).replace('_features.npz', '')

        summary_path = fp.replace('_features.npz', '_summary.npz')
        summary = {}
        if os.path.exists(summary_path):
            sd      = np.load(summary_path, allow_pickle=True)
            summary = {k: sd[k].tolist() for k in sd.files}

        frag = {
            'name':          name,
            'break_pts':     data['break_pts'],
            'break_normals': data['break_normals'],
            'fpfh':          data['fpfh'],
            'pca_features':  data['pca_features'],
            'summary':       summary,
        }
        fragments.append(frag)
        if verbose:
            print(f"  Loaded {name}: {len(frag['break_pts']):,} break pts, "
                  f"{len(frag['fpfh'])} descriptors")

    n      = len(fragments)
    n_pairs = n * (n - 1) // 2
    print(f"\n  Matching {n_pairs} pair(s)…")

    results = []
    for i, j in itertools.combinations(range(n), 2):
        res = match_fragment_pair(fragments[i], fragments[j], verbose=verbose)

        # optional ML re-ranking
        if ml_scorer and ml_scorer.trained:
            fv          = np.array(res['feature_vector'], dtype=np.float32).reshape(1, -1)
            ml_prob     = float(ml_scorer.predict_proba(fv)[0])
            res['ml_score']    = ml_prob
            res['final_score'] = 0.5 * res['score'] + 0.5 * ml_prob
        else:
            res['final_score'] = res['score']

        results.append(res)

    results.sort(key=lambda r: r['final_score'], reverse=True)
    return results


def print_match_report(results):
    """Pretty-print a ranked match report."""
    print(f"\n{'#'*70}")
    print(f"  FRAGMENT MATCH REPORT   ({len(results)} pair(s))")
    print(f"{'#'*70}")
    print(f"\n  {'Pair':<38} {'Score':>7} {'Inliers':>8} {'Compat':>8}")
    print(f"  {'-'*65}")
    for r in results:
        pair = f"{r['name_a']}  <->  {r['name_b']}"
        flag = " ***" if r['final_score'] > 0.5 else ""
        print(f"  {pair:<38} {r['final_score']:>7.3f} "
              f"{r['n_inliers']:>8} {r['compatibility']:>8.3f}{flag}")

    good = [r for r in results if r['final_score'] > 0.5]
    print(f"\n  High-confidence matches (score > 0.50): {len(good)}")
    for r in good:
        print(f"    {r['name_a']}  <->  {r['name_b']}  "
              f"score={r['final_score']:.3f}  inliers={r['n_inliers']}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Find matches between fragment break surfaces")
    parser.add_argument("files", nargs="*",
                        help="*_features.npz files  (default: all in results/)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output NPZ path for match results")
    parser.add_argument("--ml_model", type=str, default=None,
                        help="Trained MLMatchScorer joblib file")
    args = parser.parse_args()

    results_dir = os.path.join(config.BASE_DIR, "results")

    if args.files:
        feature_files = args.files
    else:
        feature_files = sorted(
            glob.glob(os.path.join(results_dir, "*_features.npz")))

    if not feature_files:
        print("No *_features.npz files found. Run feature_extraction.py first.")
        sys.exit(1)

    if len(feature_files) < 2:
        print("Need at least 2 fragment feature files to match.")
        sys.exit(1)

    print(f"\nFound {len(feature_files)} fragment(s):")
    for f in feature_files:
        print(f"  {os.path.basename(f)}")

    ml_scorer = None
    if args.ml_model:
        ml_scorer = MLMatchScorer(model_path=args.ml_model)
        if ml_scorer.trained:
            print(f"  Loaded ML scorer from {args.ml_model}")

    results = match_all_fragments(feature_files, ml_scorer=ml_scorer)
    print_match_report(results)

    # ── save results ──────────────────────────────────────────────────────
    output = args.output or os.path.join(results_dir, "match_results.npz")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez(output,
             names_a        = np.array([r['name_a']        for r in results]),
             names_b        = np.array([r['name_b']        for r in results]),
             scores         = np.array([r['final_score']   for r in results]),
             n_inliers      = np.array([r['n_inliers']     for r in results]),
             transformations= np.array([r['transformation']for r in results]),
             feature_vectors= np.array([r['feature_vector']for r in results]))
    print(f"\n  Match results saved to {output}")


if __name__ == "__main__":
    main()
