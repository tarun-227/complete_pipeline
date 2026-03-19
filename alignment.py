"""
alignment.py - Robust Gap-Aware Alignment of Matched Fragment Break Surfaces
=============================================================================
Aligns matched fragment pairs using a two-stage strategy:

  Stage 1 – Standard point-to-plane ICP
    Fast alignment initialised from the RANSAC transformation produced by
    fragment_matching.py.

  Stage 2 – Gap-aware iterative refinement
    Break surfaces are often separated by gaps due to material erosion, surface
    wear, or looter damage.  After each ICP round we compute the residual
    distribution, estimate a dynamic "gap threshold" as a percentile of those
    residuals, then re-run ICP excluding correspondences beyond the threshold.
    This avoids over-fitting to bridging noise and produces a physically
    meaningful alignment.

Outputs:
  <nameA>_to_<nameB>_aligned.ply   – coloured combined point cloud (Red=A, Blue=B)
  <nameA>_to_<nameB>_alignment.npz – transformation matrix + quality metrics

Usage:
    python alignment.py                              # top-5 matches from results/
    python alignment.py results/frag1_features.npz results/frag2_features.npz
    python alignment.py --top_k 10
"""
import os
import sys
import glob
import argparse

import numpy as np
import open3d as o3d

import config


# ── tunable parameters ────────────────────────────────────────────────────────
ICP_DISTANCE     = 3.0   # Max correspondence distance for ICP
ICP_MAX_ITER     = 200   # ICP max iterations per round
ICP_TOLERANCE    = 1e-5  # ICP convergence tolerance
GAP_PERCENTILE   = 85    # Residual percentile used to estimate the gap threshold
GAP_SCALE        = 1.5   # gap_threshold = GAP_SCALE × percentile(residuals)
GAP_ROUNDS       = 3     # Number of gap-aware refinement rounds
MIN_OVERLAP      = 0.10  # Min overlap fraction for a valid alignment


# ──────────────────────────────────────────────────────────────────────────────
# Point-to-plane ICP (Open3D wrapper)
# ──────────────────────────────────────────────────────────────────────────────

def point_to_plane_icp(source_pts, source_normals,
                        target_pts, target_normals,
                        init_T=None, max_dist=ICP_DISTANCE,
                        max_iter=ICP_MAX_ITER, tolerance=ICP_TOLERANCE):
    """
    Point-to-plane ICP via Open3D.

    Args:
        source_pts, source_normals: (N, 3) source fragment
        target_pts, target_normals: (M, 3) target fragment
        init_T: (4, 4) initial transformation or None (→ identity)
        max_dist:  max correspondence distance
        max_iter:  ICP max iterations
        tolerance: convergence tolerance

    Returns:
        T:      (4, 4) transformation  (source → target)
        result: Open3D ICPConvergenceCriteria object
    """
    src = o3d.geometry.PointCloud()
    src.points  = o3d.utility.Vector3dVector(source_pts)
    src.normals = o3d.utility.Vector3dVector(source_normals)

    tgt = o3d.geometry.PointCloud()
    tgt.points  = o3d.utility.Vector3dVector(target_pts)
    tgt.normals = o3d.utility.Vector3dVector(target_normals)

    if init_T is None:
        init_T = np.eye(4)

    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=max_dist,
        init=init_T,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter,
            relative_rmse=tolerance,
            relative_fitness=tolerance))

    return np.array(result.transformation), result


# ──────────────────────────────────────────────────────────────────────────────
# Residual computation helper
# ──────────────────────────────────────────────────────────────────────────────

def _compute_residuals(src_aligned, target_pts):
    """
    For each aligned source point find its nearest target neighbour and
    return the corresponding Euclidean distances.

    Args:
        src_aligned: (N, 3) source points already transformed
        target_pts:  (M, 3) target points

    Returns:
        residuals: (N,) float32 array of nearest-neighbour distances
    """
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(target_pts)
    tgt_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    residuals = np.empty(len(src_aligned), dtype=np.float32)
    for i, pt in enumerate(src_aligned):
        _, _, dist_sq = tgt_tree.search_knn_vector_3d(pt, 1)
        residuals[i] = float(dist_sq[0]) ** 0.5
    return residuals


# ──────────────────────────────────────────────────────────────────────────────
# Gap-aware ICP
# ──────────────────────────────────────────────────────────────────────────────

def gap_aware_icp(source_pts, source_normals,
                   target_pts, target_normals,
                   init_T=None,
                   max_dist=ICP_DISTANCE,
                   gap_percentile=GAP_PERCENTILE,
                   gap_scale=GAP_SCALE,
                   n_rounds=GAP_ROUNDS,
                   max_iter=ICP_MAX_ITER,
                   verbose=True):
    """
    Gap-aware robust ICP for break-surface alignment.

    Alternates between:
    1. ICP with the current distance threshold
    2. Re-estimating the gap threshold from the residual distribution

    Args:
        source_pts, source_normals: break surface of the source fragment
        target_pts, target_normals: break surface of the target fragment
        init_T:        (4, 4) initial transformation (from RANSAC) or None
        max_dist:      initial max correspondence distance
        gap_percentile:percentile of residuals used to set the gap threshold
        gap_scale:     gap_threshold = gap_scale × percentile(residuals)
        n_rounds:      number of gap-aware refinement rounds
        max_iter:      total ICP iterations (split across rounds)
        verbose:       print per-round stats

    Returns:
        T:           (4, 4) final transformation
        inlier_mask: (N,) bool – True = source point has a close target match
        quality:     dict with alignment quality metrics
    """
    T            = init_T if init_T is not None else np.eye(4)
    current_dist = max_dist
    iters_per_round = max(max_iter // n_rounds, 10)

    for rnd in range(n_rounds):
        T, _ = point_to_plane_icp(
            source_pts, source_normals,
            target_pts, target_normals,
            init_T=T,
            max_dist=current_dist,
            max_iter=iters_per_round)

        src_aligned = (T[:3, :3] @ source_pts.T).T + T[:3, 3]
        residuals   = _compute_residuals(src_aligned, target_pts)

        gap_threshold = gap_scale * float(np.percentile(residuals, gap_percentile))
        current_dist  = min(gap_threshold, max_dist)

        if verbose:
            n_in = int((residuals < gap_threshold).sum())
            rms  = float(np.sqrt((residuals[residuals < gap_threshold] ** 2).mean()))
            print(f"    Round {rnd + 1}/{n_rounds}:  "
                  f"gap_thresh={gap_threshold:.3f}  "
                  f"inliers={n_in}/{len(residuals)}  "
                  f"RMS={rms:.4f}")

    # ── final ICP with converged gap threshold ────────────────────────────
    T, _ = point_to_plane_icp(
        source_pts, source_normals,
        target_pts, target_normals,
        init_T=T,
        max_dist=current_dist,
        max_iter=max_iter)

    src_aligned_final = (T[:3, :3] @ source_pts.T).T + T[:3, 3]
    residuals_final   = _compute_residuals(src_aligned_final, target_pts)
    inlier_mask       = residuals_final < current_dist

    quality = _compute_alignment_quality(
        source_pts, source_normals,
        target_pts, target_normals,
        T, inlier_mask, residuals_final)

    return T, inlier_mask, quality


# ──────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_alignment_quality(source_pts, source_normals,
                                target_pts, target_normals,
                                T, inlier_mask, residuals):
    """
    Compute alignment quality metrics.

    Returns dict with:
        overlap_fraction    – fraction of source break pts with a close target match
        rms_inliers         – RMS of inlier residuals
        mean_residual       – mean inlier residual
        median_residual     – median inlier residual
        normal_antiparallel – mean dot(rot_nA, nB) at inlier pairs; positive = anti-parallel
        normal_consistency  – mean |dot(rot_nA, nB)| (1 = perfectly aligned)
        is_valid            – bool: overlap ≥ MIN_OVERLAP and RMS reasonable
    """
    n_total   = len(source_pts)
    n_inliers = int(inlier_mask.sum())
    q = {
        'n_total'          : n_total,
        'n_inliers'        : n_inliers,
        'overlap_fraction' : n_inliers / max(n_total, 1),
    }

    if n_inliers > 0:
        inl_res = residuals[inlier_mask]
        q['rms_inliers']     = float(np.sqrt((inl_res ** 2).mean()))
        q['mean_residual']   = float(inl_res.mean())
        q['median_residual'] = float(np.median(inl_res))
    else:
        q['rms_inliers']     = float('inf')
        q['mean_residual']   = float('inf')
        q['median_residual'] = float('inf')

    # Normal compatibility at inlier correspondences
    if (source_normals is not None and target_normals is not None
            and n_inliers > 0):
        R           = T[:3, :3]
        src_aligned = (R @ source_pts.T).T + T[:3, 3]
        src_nrm_rot = (R @ source_normals.T).T

        tgt_pcd  = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target_pts)
        tgt_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

        dots = []
        for i in np.where(inlier_mask)[0]:
            _, nn_idx, _ = tgt_tree.search_knn_vector_3d(src_aligned[i], 1)
            dots.append(float(src_nrm_rot[i] @ target_normals[nn_idx[0]]))

        dots = np.array(dots)
        q['normal_antiparallel'] = float(-dots.mean())     # positive = anti-parallel
        q['normal_consistency']  = float(np.abs(dots).mean())
    else:
        q['normal_antiparallel'] = 0.0
        q['normal_consistency']  = 0.0

    q['is_valid'] = (q['overlap_fraction'] >= MIN_OVERLAP
                     and q.get('rms_inliers', float('inf')) < ICP_DISTANCE * 2)

    return q


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_aligned_ply(source_pts, source_normals,
                      target_pts, target_normals,
                      T, output_path):
    """
    Save both fragments in a single coloured PLY for visual inspection.
    Source (transformed) = red;  Target = blue.
    """
    src_aligned = (T[:3, :3] @ source_pts.T).T + T[:3, 3]

    all_pts    = np.vstack([src_aligned, target_pts])
    colors_src = np.tile([1.0, 0.2, 0.2], (len(src_aligned), 1))   # red
    colors_tgt = np.tile([0.2, 0.4, 1.0], (len(target_pts),  1))   # blue
    all_colors = np.vstack([colors_src, colors_tgt])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    if source_normals is not None and target_normals is not None:
        src_nrm_rot = (T[:3, :3] @ source_normals.T).T
        pcd.normals = o3d.utility.Vector3dVector(
            np.vstack([src_nrm_rot, target_normals]))

    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
    return os.path.getsize(output_path) / (1024 * 1024)


# ──────────────────────────────────────────────────────────────────────────────
# Main alignment function
# ──────────────────────────────────────────────────────────────────────────────

def align_fragment_pair(feat_a, feat_b, init_T=None,
                         output_dir=None, verbose=True):
    """
    Full alignment pipeline for a matched fragment pair.

    Args:
        feat_a, feat_b: feature dicts (keys: name, break_pts, break_normals, …)
        init_T:         (4, 4) initial transformation from RANSAC (or None)
        output_dir:     save directory (default: results/)
        verbose:        print progress

    Returns:
        T:       (4, 4) final transformation  (A → B)
        quality: alignment quality dict
    """
    name_a, name_b = feat_a['name'], feat_b['name']
    pair_name = f"{name_a}_to_{name_b}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Aligning: {name_a}  →  {name_b}")
        print(f"{'='*60}")

    pts_a, nrm_a = feat_a['break_pts'], feat_a['break_normals']
    pts_b, nrm_b = feat_b['break_pts'], feat_b['break_normals']

    if len(pts_a) == 0 or len(pts_b) == 0:
        print("  ERROR: empty break surface – cannot align.")
        return np.eye(4), {'is_valid': False}

    if verbose:
        print(f"  Source break pts : {len(pts_a):,}")
        print(f"  Target break pts : {len(pts_b):,}")
        if init_T is not None and not np.allclose(init_T, np.eye(4)):
            print(f"  Using RANSAC initial transformation")

    T, inlier_mask, quality = gap_aware_icp(
        pts_a, nrm_a, pts_b, nrm_b,
        init_T=init_T,
        verbose=verbose)

    if verbose:
        print(f"\n  Alignment quality:")
        print(f"    Overlap fraction  : {quality['overlap_fraction']:.3f}")
        print(f"    RMS (inliers)     : {quality.get('rms_inliers', float('inf')):.4f}")
        print(f"    Normal anti-par.  : {quality.get('normal_antiparallel', 0):.4f}")
        print(f"    Valid             : {quality['is_valid']}")

    # ── save ─────────────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.join(config.BASE_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    viz_path = os.path.join(output_dir, f"{pair_name}_aligned.ply")
    mb = save_aligned_ply(pts_a, nrm_a, pts_b, nrm_b, T, viz_path)
    if verbose:
        print(f"  Saved visualisation -> {viz_path}  ({mb:.1f} MB)")

    npz_path = os.path.join(output_dir, f"{pair_name}_alignment.npz")
    qual_save = {f"quality_{k}": np.array(v)
                 for k, v in quality.items() if not isinstance(v, bool)}
    qual_save['quality_is_valid'] = np.array(quality['is_valid'])
    np.savez(npz_path,
             transformation=T,
             inlier_mask=inlier_mask,
             **qual_save)
    if verbose:
        print(f"  Saved alignment    -> {npz_path}")

    return T, quality


# ──────────────────────────────────────────────────────────────────────────────
# Feature loader helper
# ──────────────────────────────────────────────────────────────────────────────

def _load_feat(feat_path, results_dir=None):
    """Load a *_features.npz into a feature dict."""
    if not os.path.exists(feat_path):
        return None
    data = np.load(feat_path, allow_pickle=True)
    name = os.path.basename(feat_path).replace('_features.npz', '')

    summary = {}
    summary_path = feat_path.replace('_features.npz', '_summary.npz')
    if os.path.exists(summary_path):
        sd      = np.load(summary_path, allow_pickle=True)
        summary = {k: sd[k].tolist() for k in sd.files}

    return {
        'name':          name,
        'break_pts':     data['break_pts'],
        'break_normals': data['break_normals'],
        'fpfh':          data['fpfh'],
        'pca_features':  data['pca_features'],
        'summary':       summary,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gap-aware robust alignment of matched fragment break surfaces")
    parser.add_argument("files", nargs="*",
                        help="Two *_features.npz files for direct alignment, "
                             "or omit to align top-K matches from match_results.npz")
    parser.add_argument("--match_results", type=str, default=None,
                        help="Match results NPZ from fragment_matching.py")
    parser.add_argument("--top_k",      type=int, default=5,
                        help="Align top-K matches (default: 5)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = os.path.join(config.BASE_DIR, "results")
    output_dir  = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    # ── direct pair ───────────────────────────────────────────────────────
    if len(args.files) == 2:
        feat_a = _load_feat(args.files[0])
        feat_b = _load_feat(args.files[1])
        if feat_a is None or feat_b is None:
            print("ERROR: could not load feature files.")
            sys.exit(1)
        align_fragment_pair(feat_a, feat_b, output_dir=output_dir)
        return

    # ── load match results and align top-K ───────────────────────────────
    match_file = args.match_results or os.path.join(results_dir, "match_results.npz")
    if not os.path.exists(match_file):
        print(f"No match results found at {match_file}")
        print("Run fragment_matching.py first, or provide two feature files directly.")
        sys.exit(1)

    matches = np.load(match_file, allow_pickle=True)
    scores   = matches['scores']
    names_a  = matches['names_a']
    names_b  = matches['names_b']
    Ts       = matches['transformations']

    top_k    = min(args.top_k, len(scores))
    top_idx  = np.argsort(scores)[::-1][:top_k]

    print(f"\nAligning top-{top_k} matched pairs…")

    for rank, idx in enumerate(top_idx):
        na    = str(names_a[idx])
        nb    = str(names_b[idx])
        score = float(scores[idx])
        init_T = np.array(Ts[idx])

        print(f"\n  [{rank+1}/{top_k}]  {na}  <->  {nb}  (score={score:.3f})")

        fp_a = os.path.join(results_dir, f"{na}_features.npz")
        fp_b = os.path.join(results_dir, f"{nb}_features.npz")
        feat_a = _load_feat(fp_a)
        feat_b = _load_feat(fp_b)

        if feat_a is None or feat_b is None:
            print("  WARNING: feature files not found – skipping.")
            continue

        use_init = init_T if not np.allclose(init_T, np.eye(4)) else None
        align_fragment_pair(feat_a, feat_b,
                             init_T=use_init,
                             output_dir=output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
