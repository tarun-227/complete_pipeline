"""
retopologize.py - Adaptive Data Reduction / Re-topologizing
============================================================
Reduces point cloud density while preserving topology of break surfaces.
Break surface regions are kept at fine resolution; original surfaces are
aggressively downsampled. This avoids losing critical geometric detail at
break edges while making downstream feature extraction faster.

Usage:
    python retopologize.py                            # all fragments in data/without_gt/
    python retopologize.py path/to/fragment.ply
    python retopologize.py path/to/folder/
    python retopologize.py fragment.ply --break_voxel 0.2 --orig_voxel 2.0
"""
import os
import sys
import glob
import argparse
import numpy as np
import open3d as o3d

import config
from preprocess import preprocess, extract_gt


def adaptive_retopologize(pcd, break_mask, break_voxel=0.3, orig_voxel=1.5,
                           verbose=True):
    """
    Adaptive decimation: fine voxel for break surfaces, coarse for original.

    Args:
        pcd:         open3d PointCloud with normals
        break_mask:  bool/int array, 1=break, 0=original (same length as pcd.points)
        break_voxel: voxel size for break regions  (smaller -> denser)
        orig_voxel:  voxel size for original regions (larger -> sparser)
        verbose:     print stats

    Returns:
        pcd_out:  reduced PointCloud with normals preserved
        labels:   (N,) int array on output points, 1=break / 0=original
        info:     dict with reduction statistics
    """
    pts  = np.asarray(pcd.points)
    mask = np.asarray(break_mask, dtype=bool)
    n_total = len(pts)

    break_idx = np.where(mask)[0]
    orig_idx  = np.where(~mask)[0]

    has_normals = pcd.has_normals()
    nrms = np.asarray(pcd.normals) if has_normals else None

    if verbose:
        print(f"  Input : {n_total:,} pts  "
              f"({len(break_idx):,} break | {len(orig_idx):,} original)")

    out_pts  = []
    out_nrms = []
    out_lbl  = []

    def _voxel_region(idx_set, voxel, label):
        if len(idx_set) == 0:
            return
        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(pts[idx_set])
        if has_normals:
            sub.normals = o3d.utility.Vector3dVector(nrms[idx_set])
        sub_d = sub.voxel_down_sample(voxel)
        n_out = len(sub_d.points)
        out_pts.append(np.asarray(sub_d.points))
        if has_normals:
            out_nrms.append(np.asarray(sub_d.normals))
        out_lbl.append(np.full(n_out, label, dtype=np.int32))
        if verbose:
            region = "Break   " if label == 1 else "Original"
            print(f"  {region}: {len(idx_set):,} -> {n_out:,} pts  (voxel={voxel})")

    _voxel_region(break_idx, break_voxel, label=1)
    _voxel_region(orig_idx,  orig_voxel,  label=0)

    all_pts = np.vstack(out_pts)
    all_lbl = np.concatenate(out_lbl)

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(all_pts)

    if has_normals and out_nrms:
        pcd_out.normals = o3d.utility.Vector3dVector(np.vstack(out_nrms))

    reduction = 100.0 * (1.0 - len(all_pts) / n_total)
    info = {
        'n_input'       : n_total,
        'n_output'      : len(all_pts),
        'n_break_out'   : int((all_lbl == 1).sum()),
        'n_orig_out'    : int((all_lbl == 0).sum()),
        'reduction_pct' : reduction,
    }

    if verbose:
        print(f"  Output: {len(all_pts):,} pts  ({reduction:.1f}% reduction)")

    return pcd_out, all_lbl, info


def retopologize_fragment(ply_path, npz_path=None,
                           break_voxel=0.3, orig_voxel=1.5,
                           output_dir=None, verbose=True):
    """
    Full retopologize pipeline for a single fragment.

    Loads the PLY, finds the break-surface mask (from predict.py's NPZ output
    or falls back to colour-based GT), then runs adaptive decimation.

    Args:
        ply_path:    Input PLY file
        npz_path:    Prediction NPZ from predict.py  (auto-detected if None)
        break_voxel: Voxel size for break regions
        orig_voxel:  Voxel size for original regions
        output_dir:  Save dir (default: results/)
        verbose:     Print progress

    Returns:
        pcd_out: Reduced open3d PointCloud
        labels:  (N,) int array  1=break / 0=original
        info:    stats dict
    """
    name = os.path.splitext(os.path.basename(ply_path))[0]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Retopologizing: {name}")
        print(f"{'='*60}")

    pcd, _ = preprocess(ply_path, voxel_size=config.VOXEL_SIZE, verbose=verbose)
    pts     = np.asarray(pcd.points)

    # ── resolve break mask ────────────────────────────────────────────────
    if npz_path is None:
        results_dir = os.path.join(config.BASE_DIR, "results")
        npz_path    = os.path.join(results_dir, f"{name}_predictions.npz")

    if os.path.exists(npz_path):
        data = np.load(npz_path)
        key  = 'post_predictions' if 'post_predictions' in data else 'raw_predictions'
        break_mask = data[key].astype(bool)
        if verbose:
            print(f"  Predictions from {os.path.basename(npz_path)}  "
                  f"({break_mask.sum():,} break pts)")
    else:
        # fall back to colour GT
        colors = np.asarray(pcd.colors)
        if len(colors) > 0:
            break_mask = extract_gt(pcd).astype(bool)
            if verbose:
                print(f"  No NPZ found – using colour GT "
                      f"({break_mask.sum():,} break pts)")
        else:
            if verbose:
                print("  WARNING: no predictions or colours – treating all as original")
            break_mask = np.zeros(len(pts), dtype=bool)

    # guard against size mismatch after preprocessing
    if len(break_mask) != len(pts):
        if verbose:
            print(f"  WARNING: mask length {len(break_mask)} != pts {len(pts)} "
                  f"– skipping adaptive step")
        break_mask = np.zeros(len(pts), dtype=bool)

    pcd_out, labels, info = adaptive_retopologize(
        pcd, break_mask,
        break_voxel=break_voxel, orig_voxel=orig_voxel,
        verbose=verbose)

    # ── save ─────────────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = os.path.join(config.BASE_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    # colour PLY  (blue=break, grey=original)
    colors_out = np.where(
        labels[:, None] == 1,
        np.array([[0.0, 0.4, 1.0]]),
        np.array([[0.75, 0.75, 0.75]])
    ).astype(np.float64)
    pcd_out.colors = o3d.utility.Vector3dVector(colors_out)

    out_ply = os.path.join(output_dir, f"{name}_retopo.ply")
    o3d.io.write_point_cloud(out_ply, pcd_out, write_ascii=False)

    # NPZ with points + normals + labels for downstream stages
    out_npz = os.path.join(output_dir, f"{name}_retopo.npz")
    np.savez(out_npz,
             points  = np.asarray(pcd_out.points).astype(np.float32),
             normals = np.asarray(pcd_out.normals).astype(np.float32)
                       if pcd_out.has_normals() else np.zeros((0, 3), np.float32),
             labels  = labels)

    if verbose:
        print(f"  Saved PLY -> {out_ply}")
        print(f"  Saved NPZ -> {out_npz}")

    return pcd_out, labels, info


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive point cloud data reduction preserving break surface topology")
    parser.add_argument("path", nargs="?", default=None,
                        help="PLY file or folder (default: data/without_gt/)")
    parser.add_argument("--npz",         type=str,   default=None,
                        help="Predictions NPZ (auto-detected if omitted)")
    parser.add_argument("--break_voxel", type=float, default=0.3,
                        help="Voxel size for break regions (default: 0.3)")
    parser.add_argument("--orig_voxel",  type=float, default=1.5,
                        help="Voxel size for original regions (default: 1.5)")
    parser.add_argument("--output_dir",  type=str,   default=None)
    args = parser.parse_args()

    if args.path is None:
        folder = config.DATA_PRED_DIR
        files  = sorted(glob.glob(os.path.join(folder, "*.ply")) +
                        glob.glob(os.path.join(folder, "*.PLY")))
    elif os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.ply")) +
                       glob.glob(os.path.join(args.path, "*.PLY")))
    else:
        print(f"ERROR: '{args.path}' is not a valid file or directory.")
        sys.exit(1)

    if not files:
        print("No PLY files found.")
        sys.exit(1)

    print(f"\nRetopologizing {len(files)} fragment(s)…")
    for fp in files:
        retopologize_fragment(
            fp,
            npz_path    = args.npz if len(files) == 1 else None,
            break_voxel = args.break_voxel,
            orig_voxel  = args.orig_voxel,
            output_dir  = args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
