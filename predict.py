"""
predict.py  -  Break Surface Detection with PointNet++
=======================================================
Detects break surfaces on 3D stone fragment point clouds (.ply / .PLY).
Uses a pretrained PointNet++ model.  No user intervention required.

Usage
-----
  # Default - runs on data/without_gt/
  python3 predict.py

  # Single file
  python3 predict.py path/to/fragment.ply

  # Folder (all .ply files inside)
  python3 predict.py path/to/folder/

Optional flags
  --checkpoint   path to model checkpoint  (default: checkpoints/best_model.pt)
  --threshold    break probability cutoff  (default: 0.5)
  --batch_size   patches per GPU per step  (default: from config)
  --voxel        voxel downsample size     (default: from config)

Outputs (saved to results/)
  <name>_raw.ply            Blue=break, Grey=original  (threshold only)
  <name>_postprocessed.ply  After iterative fill + cluster removal
  <name>_predictions.npz   raw_predictions, post_predictions, proba arrays
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from collections import Counter

import config
from preprocess import preprocess
from model.pointnet2 import PointNet2Classifier


# -----------------------------------------------------------------------------
# INFERENCE
# -----------------------------------------------------------------------------

def predict_fragment(points, normals, model, devices, num_points=4096,
                     batch_size=64):
    """
    Predict break probability for every point in a fragment.
    Uses nn.DataParallel so all available GPUs are utilised automatically.
    """
    model.eval()
    primary = devices[0]
    n_total = len(points)
    proba   = np.zeros(n_total, dtype=np.float32)

    pcd_cpu = o3d.geometry.PointCloud()
    pcd_cpu.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(pcd_cpu)

    effective_batch = batch_size * len(devices)
    n_batches = (n_total + effective_batch - 1) // effective_batch

    print(f"  GPUs          : {[str(d) for d in devices]}")
    print(f"  Batch/GPU     : {batch_size}  ->  effective: {effective_batch}")
    print(f"  Total batches : {n_batches:,}")

    t0 = time.time()

    with torch.no_grad():
        for bi in range(n_batches):
            start = bi * effective_batch
            end   = min(start + effective_batch, n_total)

            batch_features = []
            for idx in range(start, end):
                center = points[idx]
                _, nn_idx, _ = tree.search_knn_vector_3d(center, num_points + 1)
                nn_idx = np.array(nn_idx[1:])
                if len(nn_idx) < num_points:
                    pad = np.random.choice(nn_idx, num_points - len(nn_idx), replace=True)
                    nn_idx = np.concatenate([nn_idx, pad])
                nn_idx = nn_idx[:num_points]
                patch_pts = points[nn_idx] - center
                patch_nrm = normals[nn_idx]
                feat = np.concatenate([patch_pts, patch_nrm], axis=1)
                batch_features.append(feat.T.astype(np.float32))

            batch_tensor = torch.from_numpy(np.stack(batch_features)).to(primary)
            logits, _ = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            proba[start:end] = probs

            if bi % 20 == 0 or bi == n_batches - 1:
                elapsed = time.time() - t0
                pct = 100.0 * end / n_total
                eta = (elapsed / max(end, 1)) * (n_total - end)
                print(f"    {pct:5.1f}%  ({end:,}/{n_total:,})  "
                      f"{elapsed/60:.1f} min elapsed  ~{eta/60:.1f} min left")

    print(f"  Inference done in {(time.time()-t0)/60:.1f} min")
    return proba


# -----------------------------------------------------------------------------
# POST-PROCESSING
# -----------------------------------------------------------------------------

def iterative_fill(points, preds, tree, k=50, ratio=0.5, max_iter=10):
    """Flip original->break for points whose break-neighbour fraction >= ratio."""
    p = preds.copy()
    total_filled = 0
    for it in range(max_iter):
        orig_idx = np.where(p == 0)[0]
        flips = []
        for idx in orig_idx:
            _, nn, _ = tree.search_knn_vector_3d(points[idx], k + 1)
            nn_arr = np.array(nn[1:])
            if p[nn_arr].sum() / k >= ratio:
                flips.append(idx)
        for idx in flips:
            p[idx] = 1
        total_filled += len(flips)
        print(f"      iter {it+1}: flipped {len(flips):,} points")
        if len(flips) == 0:
            break
    return p, total_filled


def postprocess(pcd, proba, threshold=0.5):
    """
    Post-processing pipeline:
      1. Threshold raw probabilities -> binary predictions
      2. Probability pull-in: borderline points (0.3 to threshold) promoted
         if >= 40% of k=30 neighbours have prob > 0.7
      3. Three-pass iterative fill (tight -> medium -> loose)
      4. Erosion: remove isolated break points with < 30% break neighbours
      5. Dynamic DBSCAN cluster removal (eps from actual point spacing)
    """
    points = np.asarray(pcd.points)
    tree   = o3d.geometry.KDTreeFlann(pcd)

    # 1. Threshold
    preds = (proba >= threshold).astype(int)
    print(f"    [1] Threshold ({threshold:.2f}): {preds.sum():,} break points")

    # 2. Probability pull-in
    borderline = np.where((proba >= 0.3) & (proba < threshold))[0]
    pulled = 0
    K_PULL = 30
    for idx in borderline:
        _, nn, _ = tree.search_knn_vector_3d(points[idx], K_PULL + 1)
        nn_arr = np.array(nn[1:])
        if (proba[nn_arr] > 0.7).sum() / K_PULL >= 0.4:
            preds[idx] = 1
            pulled += 1
    print(f"    [2] Probability pull-in: +{pulled:,} points")

    # 3. Iterative fill
    print(f"    [3] Iterative fill...")
    preds, n1 = iterative_fill(points, preds, tree, k=50, ratio=0.6, max_iter=10)
    preds, n2 = iterative_fill(points, preds, tree, k=40, ratio=0.5, max_iter=10)
    preds, n3 = iterative_fill(points, preds, tree, k=30, ratio=0.4, max_iter=5)
    print(f"    [3] Fill added {n1+n2+n3:,} points total")

    # 4. Erosion
    K_ERODE, ERODE_RATIO = 20, 0.30
    break_idx_pre = np.where(preds == 1)[0]
    eroded = 0
    for idx in break_idx_pre:
        _, nn, _ = tree.search_knn_vector_3d(points[idx], K_ERODE + 1)
        nn_arr = np.array(nn[1:])
        if preds[nn_arr].sum() / K_ERODE < ERODE_RATIO:
            preds[idx] = 0
            eroded += 1
    print(f"    [4] Erosion removed {eroded:,} isolated points")

    # 5. Dynamic DBSCAN cluster removal
    break_idx = np.where(preds == 1)[0]
    if len(break_idx) > 50:
        bp = o3d.geometry.PointCloud()
        bp.points = o3d.utility.Vector3dVector(points[break_idx])
        sample_idx = np.random.choice(len(points), min(2000, len(points)), replace=False)
        dists = []
        for i in sample_idx:
            _, _, dsq = tree.search_knn_vector_3d(points[i], 2)
            dists.append(np.sqrt(dsq[1]))
        spacing = float(np.median(dists))
        eps = spacing * 5
        min_cluster = max(30, int(0.001 * len(break_idx)))
        print(f"    [5] DBSCAN eps={eps:.4f}  min_cluster={min_cluster}")
        cls = np.array(bp.cluster_dbscan(eps=eps, min_points=10))
        cc  = Counter(cls[cls >= 0])
        small = set(c for c, cnt in cc.items() if cnt < min_cluster)
        removed = 0
        for i, cidx in enumerate(break_idx):
            if cls[i] in small or cls[i] == -1:
                preds[cidx] = 0
                removed += 1
        print(f"    [5] Removed {removed:,} noise/small-cluster points")

    n_final = int(preds.sum())
    print(f"    Final: {n_final:,} break points ({100*n_final/len(preds):.1f}%)")
    return preds


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def save_result_ply(pcd, preds, output_path):
    """Save coloured PLY. Blue=break, Grey=original."""
    preds_arr = np.asarray(preds)
    colors = np.where(
        preds_arr[:, None] == 1,
        np.array([[0.0, 0.4, 1.0]]),
        np.array([[0.75, 0.75, 0.75]])
    ).astype(np.float64)
    result = o3d.geometry.PointCloud()
    result.points  = pcd.points
    result.colors  = o3d.utility.Vector3dVector(colors)
    if pcd.has_normals():
        result.normals = pcd.normals
    o3d.io.write_point_cloud(output_path, result, write_ascii=False)
    return os.path.getsize(output_path) / (1024 * 1024)


def resolve_input_files(path_arg):
    """
    Resolve <path> argument to a list of PLY file paths.
      None/omitted -> config.DATA_PRED_DIR
      file path    -> [that file]
      folder path  -> all PLY files inside
    """
    if path_arg is None:
        folder = config.DATA_PRED_DIR
        files = sorted(glob.glob(os.path.join(folder, "*.ply")) +
                       glob.glob(os.path.join(folder, "*.PLY")))
        print(f"  No path given - using default folder: {folder}")
        return files

    path_arg = os.path.abspath(path_arg)

    if os.path.isfile(path_arg):
        return [path_arg]

    if os.path.isdir(path_arg):
        return sorted(glob.glob(os.path.join(path_arg, "*.ply")) +
                      glob.glob(os.path.join(path_arg, "*.PLY")))

    print(f"  ERROR: '{path_arg}' is not a valid file or directory.")
    sys.exit(1)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Break surface detection with PointNet++",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 predict.py
  python3 predict.py data/without_gt/frag3.PLY
  python3 predict.py data/without_gt/
  python3 predict.py /absolute/path/to/folder/
        """)

    parser.add_argument(
        "path", nargs="?", default=None,
        help="Path to a .ply file or folder of .ply files. "
             "Defaults to data/without_gt/ if omitted.")
    parser.add_argument("--checkpoint", type=str,   default=None)
    parser.add_argument("--voxel",      type=float, default=config.VOXEL_SIZE)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--batch_size", type=int,   default=config.PRED_BATCH_SIZE)
    args = parser.parse_args()

    # GPU setup
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("  WARNING: No CUDA GPUs found - running on CPU (will be slow)")
        devices = [torch.device("cpu")]
    else:
        devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
        print(f"\n  Found {n_gpus} GPU(s):")
        for d in devices:
            print(f"    {d}  -  {torch.cuda.get_device_name(d)}")
    primary = devices[0]

    # Load checkpoint
    ckpt_path = args.checkpoint or os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"\n  ERROR: Checkpoint not found at '{ckpt_path}'")
        print(f"  Place best_model.pt in the checkpoints/ directory.")
        sys.exit(1)

    print(f"\n{'#'*70}")
    print("  POINTNET++ BREAK SURFACE DETECTION")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Threshold  : {args.threshold}")
    print(f"{'#'*70}")

    checkpoint   = torch.load(ckpt_path, map_location=primary, weights_only=False)
    model_config = checkpoint['config']

    base_model = PointNet2Classifier(
        input_channels=model_config['input_channels'],
        dropout=0.0).to(primary)
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()

    if n_gpus > 1:
        model = nn.DataParallel(base_model, device_ids=list(range(n_gpus)))
        print(f"  DataParallel across {n_gpus} GPUs")
    else:
        model = base_model

    print(f"  Model      : epoch {checkpoint['epoch']}, val F1 = {checkpoint['val_f1']:.4f}")
    print(f"  Trained on : {model_config['training_fragments']}")

    # Resolve files
    files = resolve_input_files(args.path)
    if not files:
        print("\n  ERROR: No .ply / .PLY files found.")
        sys.exit(1)

    print(f"\n  Fragments to process: {len(files)}")
    for f in files:
        print(f"    {os.path.basename(f)}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    all_stats = []

    for fi, fp in enumerate(files):
        frag_name = os.path.splitext(os.path.basename(fp))[0]

        print(f"\n{'='*70}")
        print(f"  [{fi+1}/{len(files)}]  {frag_name}")
        print(f"{'='*70}")

        t_total = time.time()

        pcd, log = preprocess(fp, voxel_size=args.voxel)
        points  = np.asarray(pcd.points).astype(np.float32)
        normals = np.asarray(pcd.normals).astype(np.float32)
        print(f"  Points after preprocessing: {len(points):,}")

        print(f"\n  Running inference...")
        proba = predict_fragment(points, normals, model, devices,
                                 num_points=model_config['num_points'],
                                 batch_size=args.batch_size)

        # RAW output
        raw_preds = (proba >= args.threshold).astype(int)
        raw_path  = os.path.join(config.RESULTS_DIR, f"{frag_name}_raw.ply")
        mb = save_result_ply(pcd, raw_preds, raw_path)
        n_raw = int(raw_preds.sum())
        print(f"\n  [RAW]  {n_raw:,} break pts ({100*n_raw/len(raw_preds):.1f}%)  ->  {raw_path} ({mb:.1f} MB)")

        # Postprocessed output
        print(f"\n  Post-processing...")
        post_preds = postprocess(pcd, proba, threshold=args.threshold)
        post_path  = os.path.join(config.RESULTS_DIR, f"{frag_name}_postprocessed.ply")
        mb = save_result_ply(pcd, post_preds, post_path)
        n_post = int(post_preds.sum())
        print(f"\n  [POST] {n_post:,} break pts ({100*n_post/len(post_preds):.1f}%)  ->  {post_path} ({mb:.1f} MB)")

        # NPZ arrays
        npz_path = os.path.join(config.RESULTS_DIR, f"{frag_name}_predictions.npz")
        np.savez(npz_path, raw_predictions=raw_preds,
                 post_predictions=post_preds, probabilities=proba)
        print(f"  [NPZ]  {npz_path}")

        elapsed = time.time() - t_total
        all_stats.append({'name': frag_name, 'n_pts': len(post_preds),
                          'n_raw': n_raw, 'n_post': n_post, 'time': elapsed})
        print(f"\n  Done in {elapsed/60:.1f} min")

    # Summary
    print(f"\n{'#'*70}")
    print(f"  COMPLETE  -  {len(all_stats)} fragment(s) processed")
    print(f"{'#'*70}")
    print(f"\n  {'Fragment':<38} {'Points':>10} {'Raw%':>7} {'Post%':>7} {'Time':>8}")
    print(f"  {'-'*72}")
    for s in all_stats:
        print(f"  {s['name']:<38} {s['n_pts']:>10,} "
              f"{100*s['n_raw']/s['n_pts']:>6.1f}% "
              f"{100*s['n_post']/s['n_pts']:>6.1f}% "
              f"{s['time']/60:>7.1f}m")
    print()


if __name__ == "__main__":
    main()