"""
feature_extraction.py - Geometric Feature Extraction for Break Surfaces
========================================================================
Extracts multi-scale geometric descriptors from detected break surface points.

Per-point features:
  - FPFH  (Fast Point Feature Histograms) – 33-dim, rotation-invariant
  - Local PCA shape features: linearity, planarity, sphericity, omnivariance,
    curvature estimate, normal angular deviation

Per-fragment summary (used for rapid compatibility screening in matching):
  - Break surface extent, area proxy, spread
  - Mean normal direction + consistency
  - Roughness, mean curvature, mean planarity
  - Mean / std of FPFH distribution

Input:  a *_retopo.npz produced by retopologize.py  (or a raw PLY file)
Output: *_features.npz  and  *_summary.npz  written to results/

Usage:
    python feature_extraction.py                          # all retopo NPZs in results/
    python feature_extraction.py results/frag3_retopo.npz
    python feature_extraction.py data/with_gt/frag3.ply
"""
import os
import sys
import glob
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

import config
from preprocess import preprocess, extract_gt


# ── tunable parameters ────────────────────────────────────────────────────────
FPFH_RADIUS  = 5.0   # search radius for FPFH (should be > normal-estimation radius)
FPFH_MAX_NN  = 100   # max neighbours for FPFH
PCA_RADIUS   = 3.0   # neighbourhood radius for local PCA
PCA_MAX_NN   = 50    # max neighbours for local PCA


# ──────────────────────────────────────────────────────────────────────────────
# FPFH
# ──────────────────────────────────────────────────────────────────────────────

def compute_fpfh(pcd, radius=FPFH_RADIUS, max_nn=FPFH_MAX_NN):
    """
    Compute FPFH descriptors for every point in pcd via Open3D.

    Args:
        pcd:    open3d PointCloud (must have normals)
        radius: ball-query radius
        max_nn: max neighbours

    Returns:
        (N, 33) float32 FPFH array
    """
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    return np.asarray(fpfh.data, dtype=np.float32).T  # (N, 33)


# ──────────────────────────────────────────────────────────────────────────────
# Local PCA shape features
# ──────────────────────────────────────────────────────────────────────────────

def compute_local_pca_features(pts, normals, radius=PCA_RADIUS, max_nn=PCA_MAX_NN):
    """
    Compute local PCA-based shape features for each point.

    For each point we find neighbours within `radius`, run PCA on their
    coordinates and derive eigenvalue-ratio features that encode local shape:

        linearity   = (λ1 - λ2) / λ1
        planarity   = (λ2 - λ3) / λ1
        sphericity  = λ3 / λ1
        omnivariance = (λ1 * λ2 * λ3)^(1/3)
        curvature   = λ3 / (λ1 + λ2 + λ3)
        normal_dev  = 1 - mean |n_i · n_centre|

    Args:
        pts:     (N, 3) point coordinates
        normals: (N, 3) per-point normals  (or None)
        radius:  neighbourhood radius
        max_nn:  max neighbours per point

    Returns:
        (N, 6) float32  [linearity, planarity, sphericity,
                          omnivariance, curvature, normal_dev]
    """
    tree = cKDTree(pts)
    N = len(pts)
    out = np.zeros((N, 6), dtype=np.float32)

    for i in range(N):
        nn_idx = tree.query_ball_point(pts[i], radius)
        nn_idx = nn_idx[:max_nn]

        if len(nn_idx) < 4:
            continue

        neighbors = pts[nn_idx]
        centered  = neighbors - neighbors.mean(axis=0)

        # SVD – eigenvalues are s² / (n-1) but ratios are the same as s²
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        eigs = (s ** 2)                   # descending (SVD guarantees this)
        lam1, lam2, lam3 = eigs[0], eigs[1], eigs[2]

        eps   = 1e-10
        total = lam1 + lam2 + lam3 + eps

        linearity   = (lam1 - lam2) / (lam1 + eps)
        planarity   = (lam2 - lam3) / (lam1 + eps)
        sphericity  =  lam3         / (lam1 + eps)
        omnivar     = (lam1 * lam2 * lam3 + 1e-30) ** (1.0 / 3.0)
        curvature   =  lam3 / total

        if normals is not None:
            dots       = np.abs(normals[nn_idx] @ normals[i])
            normal_dev = 1.0 - float(dots.mean())
        else:
            normal_dev = 0.0

        out[i] = [linearity, planarity, sphericity, omnivar, curvature, normal_dev]

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Fragment-level summary
# ──────────────────────────────────────────────────────────────────────────────

def compute_break_surface_summary(break_pts, break_normals, fpfh_feats, pca_feats):
    """
    Aggregate break-surface statistics into a compact fragment descriptor.
    Used for rapid pairwise compatibility screening before expensive matching.

    Returns:
        dict – all values are Python scalars or lists (serialisable with np.savez)
    """
    s = {}

    if len(break_pts) == 0:
        return s

    # Geometric extent
    bbox_min = break_pts.min(axis=0)
    bbox_max = break_pts.max(axis=0)
    extent   = bbox_max - bbox_min
    s['extent']             = extent.tolist()
    s['surface_area_proxy'] = float(np.prod(np.sort(extent)[-2:]))  # two largest dims
    s['n_points']           = len(break_pts)

    centroid      = break_pts.mean(axis=0)
    s['centroid'] = centroid.tolist()
    s['spread']   = float(np.linalg.norm(break_pts - centroid, axis=1).mean())

    # Normal statistics
    if break_normals is not None and len(break_normals) > 0:
        mean_nrm  = break_normals.mean(axis=0)
        norm_len  = np.linalg.norm(mean_nrm) + 1e-10
        mean_nrm /= norm_len
        s['mean_normal']         = mean_nrm.tolist()
        dots                     = np.abs(break_normals @ mean_nrm)
        s['normal_consistency']  = float(dots.mean())

        # Roughness: std of point projections onto mean normal
        heights      = break_pts @ mean_nrm
        s['roughness'] = float(heights.std())
    else:
        s['mean_normal']        = [0.0, 0.0, 1.0]
        s['normal_consistency'] = 0.0
        s['roughness']          = 0.0

    # PCA shape statistics
    if len(pca_feats) > 0:
        s['mean_curvature']  = float(pca_feats[:, 4].mean())
        s['std_curvature']   = float(pca_feats[:, 4].std())
        s['mean_planarity']  = float(pca_feats[:, 1].mean())
        s['mean_sphericity'] = float(pca_feats[:, 2].mean())
    else:
        s['mean_curvature']  = 0.0
        s['std_curvature']   = 0.0
        s['mean_planarity']  = 0.0
        s['mean_sphericity'] = 0.0

    # FPFH distribution statistics
    if len(fpfh_feats) > 0:
        s['fpfh_mean'] = fpfh_feats.mean(axis=0).tolist()
        s['fpfh_std']  = fpfh_feats.std(axis=0).tolist()

    return s


# ──────────────────────────────────────────────────────────────────────────────
# Top-level extraction function
# ──────────────────────────────────────────────────────────────────────────────

def extract_break_surface_features(pts, normals, labels,
                                    fpfh_radius=FPFH_RADIUS,
                                    pca_radius=PCA_RADIUS,
                                    verbose=True):
    """
    Extract all geometric features for break surface points.

    Args:
        pts:        (N, 3) all fragment points
        normals:    (N, 3) per-point normals
        labels:     (N,) binary labels  1=break / 0=original
        fpfh_radius: search radius for FPFH
        pca_radius:  neighbourhood radius for local PCA

    Returns:
        break_pts:     (M, 3) break surface points
        break_normals: (M, 3) break surface normals
        fpfh_feats:    (M, 33) FPFH descriptors
        pca_feats:     (M, 6) local PCA features
        summary:       dict with fragment-level statistics
    """
    break_mask    = labels.astype(bool)
    break_pts     = pts[break_mask]
    break_normals = normals[break_mask] if normals is not None else None

    if verbose:
        print(f"  Break surface: {len(break_pts):,} / {len(pts):,} points")

    if len(break_pts) == 0:
        return (break_pts, break_normals,
                np.zeros((0, 33), np.float32),
                np.zeros((0, 6),  np.float32),
                {})

    # Build full point cloud for FPFH (FPFH uses the full neighbourhood context)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    if verbose:
        print(f"  Computing FPFH  (radius={fpfh_radius}, max_nn={FPFH_MAX_NN})…")
    fpfh_all   = compute_fpfh(pcd, radius=fpfh_radius)
    fpfh_feats = fpfh_all[break_mask]

    if verbose:
        print(f"  Computing local PCA features  (radius={pca_radius})…")
    pca_feats = compute_local_pca_features(
        break_pts, break_normals, radius=pca_radius)

    summary = compute_break_surface_summary(
        break_pts, break_normals, fpfh_feats, pca_feats)

    if verbose:
        print(f"  Curvature  (mean/std): "
              f"{summary.get('mean_curvature', 0):.4f} / "
              f"{summary.get('std_curvature', 0):.4f}")
        print(f"  Planarity  (mean):     {summary.get('mean_planarity', 0):.4f}")
        print(f"  Roughness:             {summary.get('roughness', 0):.4f}")
        print(f"  Break-surface spread:  {summary.get('spread', 0):.4f}")

    return break_pts, break_normals, fpfh_feats, pca_feats, summary


def extract_and_save(input_path, output_dir=None, verbose=True):
    """
    Load a fragment (from a *_retopo.npz or a raw PLY), extract features, save.

    Args:
        input_path: path to *_retopo.npz  or  *.ply
        output_dir: where to write outputs (default: results/)

    Returns:
        dict with keys: name, break_pts, break_normals, fpfh, pca_features, summary
    """
    # ── strip suffixes to get clean name ─────────────────────────────────
    base = os.path.basename(input_path)
    base = base.replace('_retopo.npz', '').replace('_retopo.ply', '')
    name = os.path.splitext(base)[0]

    if output_dir is None:
        output_dir = os.path.join(config.BASE_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Feature extraction: {name}")
        print(f"{'='*60}")

    # ── load data ─────────────────────────────────────────────────────────
    if input_path.endswith('.npz'):
        data    = np.load(input_path)
        pts     = data['points'].astype(np.float32)
        normals = data['normals'].astype(np.float32) if 'normals' in data else None
        labels  = data['labels'].astype(np.int32)
    else:
        # raw PLY – preprocess then resolve labels
        pcd, _  = preprocess(input_path, voxel_size=config.VOXEL_SIZE,
                              verbose=verbose)
        pts     = np.asarray(pcd.points,  dtype=np.float32)
        normals = np.asarray(pcd.normals, dtype=np.float32)

        pred_npz = os.path.join(output_dir, f"{name}_predictions.npz")
        if os.path.exists(pred_npz):
            pd = np.load(pred_npz)
            key    = 'post_predictions' if 'post_predictions' in pd else 'raw_predictions'
            labels = pd[key].astype(np.int32)
            if verbose:
                print(f"  Labels from {os.path.basename(pred_npz)}")
        else:
            labels = extract_gt(pcd).astype(np.int32)
            if verbose:
                print("  Labels from colour GT")

    # ── extract ───────────────────────────────────────────────────────────
    break_pts, break_normals, fpfh_feats, pca_feats, summary = \
        extract_break_surface_features(pts, normals, labels, verbose=verbose)

    # ── save features NPZ ─────────────────────────────────────────────────
    feat_path = os.path.join(output_dir, f"{name}_features.npz")
    np.savez(feat_path,
             break_pts     = break_pts,
             break_normals = break_normals if break_normals is not None
                             else np.zeros((0, 3), np.float32),
             fpfh          = fpfh_feats,
             pca_features  = pca_feats,
             all_pts       = pts,
             all_normals   = normals if normals is not None
                             else np.zeros((0, 3), np.float32),
             all_labels    = labels)

    # ── save summary NPZ (quick-load for screening) ───────────────────────
    summary_path = os.path.join(output_dir, f"{name}_summary.npz")
    np.savez(summary_path,
             **{k: np.array(v) for k, v in summary.items()})

    if verbose:
        print(f"  Saved features -> {feat_path}")
        print(f"  Saved summary  -> {summary_path}")

    return {
        'name'          : name,
        'break_pts'     : break_pts,
        'break_normals' : break_normals,
        'fpfh'          : fpfh_feats,
        'pca_features'  : pca_feats,
        'summary'       : summary,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract geometric features from detected break surfaces")
    parser.add_argument("path", nargs="?", default=None,
                        help="*_retopo.npz, PLY file, or folder  "
                             "(default: all *_retopo.npz in results/)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = os.path.join(config.BASE_DIR, "results")

    if args.path is None:
        files = sorted(glob.glob(os.path.join(results_dir, "*_retopo.npz")))
        if not files:
            # fall back to raw PLYs
            files = sorted(glob.glob(os.path.join(config.DATA_PRED_DIR, "*.ply")) +
                           glob.glob(os.path.join(config.DATA_PRED_DIR, "*.PLY")))
    elif os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        files = (sorted(glob.glob(os.path.join(args.path, "*_retopo.npz"))) or
                 sorted(glob.glob(os.path.join(args.path, "*.ply"))))
    else:
        print(f"ERROR: '{args.path}' not found.")
        sys.exit(1)

    if not files:
        print("No files found. Run retopologize.py first.")
        sys.exit(1)

    print(f"\nExtracting features from {len(files)} fragment(s)…")
    for fp in files:
        extract_and_save(fp, output_dir=args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
