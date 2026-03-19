"""
preprocess.py - Light Preprocessing for PointNet++
====================================================
Minimal preprocessing: dedup, optional light downsample, normals.
Saves preprocessed files to data/preprocessed/ to avoid re-processing.
"""
import os
import time
import numpy as np
import open3d as o3d


# Default preprocessed directory (relative to base_dir)
def get_preprocessed_dir():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(base_dir, "data", "preprocessed")
    os.makedirs(d, exist_ok=True)
    return d


def get_cached_path(filepath, voxel_size):
    """Get the cached preprocessed file path for a given input."""
    frag_name = os.path.splitext(os.path.basename(filepath))[0]
    voxel_str = f"v{voxel_size}" if voxel_size and voxel_size > 0 else "v0"
    cached_name = f"{frag_name}_{voxel_str}.ply"
    return os.path.join(get_preprocessed_dir(), cached_name)


def preprocess(filepath, voxel_size=0.5, verbose=True, use_cache=True):
    """
    Light preprocessing pipeline with caching.

    Args:
        filepath: Path to PLY file
        voxel_size: Voxel downsample size (None or 0 to skip)
        verbose: Print progress
        use_cache: Load from data/preprocessed/ if available

    Returns:
        pcd: Preprocessed point cloud with normals
        log: Dict with stats
    """
    log = {}

    # Check cache
    cached_path = get_cached_path(filepath, voxel_size)
    if use_cache and os.path.exists(cached_path):
        if verbose:
            print(f"\n  Loading cached: {os.path.basename(cached_path)}...")
        t0 = time.time()
        pcd = o3d.io.read_point_cloud(cached_path)
        log['n_final'] = len(pcd.points)
        log['cached'] = True
        log['preprocess_time'] = time.time() - t0
        if verbose:
            print(f"  Loaded {len(pcd.points):,} points ({log['preprocess_time']:.1f}s)")
        return pcd, log

    # Full preprocessing
    t0 = time.time()

    if verbose:
        print(f"\n  Loading {os.path.basename(filepath)}...")

    pcd = o3d.io.read_point_cloud(filepath)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    n_raw = len(pts)
    log['n_raw'] = n_raw

    if verbose:
        print(f"  Raw: {n_raw:,} points")

    # Deduplication
    _, uidx = np.unique(pts, axis=0, return_index=True)
    uidx = np.sort(uidx)
    pcd_c = o3d.geometry.PointCloud()
    pcd_c.points = o3d.utility.Vector3dVector(pts[uidx])
    if cols is not None:
        pcd_c.colors = o3d.utility.Vector3dVector(cols[uidx])
    n_dedup = len(uidx)
    log['n_dedup'] = n_dedup

    if verbose:
        print(f"  After dedup: {n_dedup:,}")

    # Voxel downsample
    if voxel_size is not None and voxel_size > 0:
        pcd_d = pcd_c.voxel_down_sample(voxel_size)
        n_voxel = len(pcd_d.points)
        log['n_voxel'] = n_voxel
        if verbose:
            print(f"  After voxel ({voxel_size}): {n_voxel:,}")
    else:
        pcd_d = pcd_c
        log['n_voxel'] = n_dedup

    # Statistical outlier removal
    cl, ind = pcd_d.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_f = pcd_d.select_by_index(ind)
    n_final = len(pcd_f.points)
    log['n_final'] = n_final

    if verbose:
        print(f"  After SOR: {n_final:,}")

    # Normal estimation
    if verbose:
        print(f"  Estimating normals...")
    pcd_f.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd_f.orient_normals_consistent_tangent_plane(k=15)

    log['preprocess_time'] = time.time() - t0
    log['cached'] = False

    if verbose:
        print(f"  Done ({log['preprocess_time']:.1f}s)")

    # Save to cache
    o3d.io.write_point_cloud(cached_path, pcd_f, write_ascii=False)
    size_mb = os.path.getsize(cached_path) / (1024 * 1024)
    if verbose:
        print(f"  Cached: {cached_path} ({size_mb:.1f} MB)")

    return pcd_f, log


def extract_gt(pcd):
    """
    Extract ground truth labels from green-painted vertices.
    1 = break, 0 = original.
    """
    colors = np.asarray(pcd.colors)
    labels = ((colors[:, 1] > 0.8) &
              (colors[:, 0] < 0.3) &
              (colors[:, 2] < 0.3)).astype(np.int64)
    return labels