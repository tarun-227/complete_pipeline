"""
pipeline.py - End-to-End Fragment Reconstruction Pipeline
==========================================================
Ties all stages together into a single runnable script:

  Stage 1 – Preprocess      (preprocess.py)
  Stage 2 – Break detection (predict.py)
  Stage 3 – Retopologize    (retopologize.py)
  Stage 4 – Feature extract (feature_extraction.py)
  Stage 5 – Match           (fragment_matching.py)
  Stage 6 – Align           (alignment.py)

Each stage can be skipped if its outputs already exist (--force to re-run).

Usage:
    python pipeline.py                            # all PLY files in data/without_gt/
    python pipeline.py path/to/folder/
    python pipeline.py frag1.ply frag2.ply frag3.ply
    python pipeline.py data/without_gt/ --stages 3 4 5 6   # skip predict if done
    python pipeline.py data/without_gt/ --force             # re-run everything
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

import config
from preprocess import preprocess
from predict import predict_fragment, postprocess, save_result_ply
from retopologize import retopologize_fragment
from feature_extraction import extract_and_save
from fragment_matching import match_all_fragments, print_match_report
from alignment import align_fragment_pair, _load_feat
from model.pointnet2 import PointNet2Classifier


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _header(title):
    print(f"\n{'#'*70}")
    print(f"  {title}")
    print(f"{'#'*70}")


def _section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def resolve_ply_files(paths):
    """Expand a list of files / folders to a sorted list of PLY paths."""
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.ply")) +
                          glob.glob(os.path.join(p, "*.PLY")))
        else:
            print(f"  WARNING: '{p}' not found – skipping.")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 – model loader (shared across all fragments)
# ──────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path):
    """Load the PointNet++ checkpoint and return (model, devices, model_config)."""
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("  No CUDA GPUs – running on CPU (slow).")
        devices = [torch.device("cpu")]
    else:
        devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)]
        print(f"  {n_gpus} GPU(s) detected.")
    primary = devices[0]

    if not os.path.exists(ckpt_path):
        print(f"  ERROR: checkpoint not found at '{ckpt_path}'")
        sys.exit(1)

    ckpt        = torch.load(ckpt_path, map_location=primary, weights_only=False)
    model_cfg   = ckpt['config']
    base_model  = PointNet2Classifier(
        input_channels=model_cfg['input_channels'],
        dropout=0.0).to(primary)
    base_model.load_state_dict(ckpt['model_state_dict'])
    base_model.eval()

    if n_gpus > 1:
        model = nn.DataParallel(base_model, device_ids=list(range(n_gpus)))
    else:
        model = base_model

    print(f"  Loaded checkpoint: epoch {ckpt['epoch']}, "
          f"val F1 = {ckpt['val_f1']:.4f}")
    return model, devices, model_cfg


# ──────────────────────────────────────────────────────────────────────────────
# Stage runners
# ──────────────────────────────────────────────────────────────────────────────

def run_stage1_preprocess(ply_files, voxel_size):
    """Preprocess all PLY files (with caching). Returns list of (name, pcd) tuples."""
    _header("STAGE 1 – PREPROCESSING")
    results = []
    for fp in ply_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n  {name}")
        pcd, log = preprocess(fp, voxel_size=voxel_size, verbose=True)
        results.append((name, fp, pcd))
    return results


def run_stage2_detect(preprocessed, model, devices, model_cfg,
                       results_dir, threshold, batch_size, force):
    """Run break surface detection for each fragment."""
    _header("STAGE 2 – BREAK SURFACE DETECTION")
    out = []
    for name, fp, pcd in preprocessed:
        npz_path = os.path.join(results_dir, f"{name}_predictions.npz")
        if not force and os.path.exists(npz_path):
            print(f"\n  {name}: predictions found – skipping  ({npz_path})")
            data = np.load(npz_path)
            out.append((name, fp, pcd,
                        data['post_predictions'],
                        data['probabilities']))
            continue

        _section(f"Detecting: {name}")
        pts     = np.asarray(pcd.points,  dtype=np.float32)
        normals = np.asarray(pcd.normals, dtype=np.float32)

        proba     = predict_fragment(pts, normals, model, devices,
                                     num_points=model_cfg['num_points'],
                                     batch_size=batch_size)
        raw_preds  = (proba >= threshold).astype(int)
        post_preds = postprocess(pcd, proba, threshold=threshold)

        # save raw coloured PLY
        raw_path  = os.path.join(results_dir, f"{name}_raw.ply")
        post_path = os.path.join(results_dir, f"{name}_postprocessed.ply")
        save_result_ply(pcd, raw_preds,  raw_path)
        save_result_ply(pcd, post_preds, post_path)

        np.savez(npz_path,
                 raw_predictions=raw_preds,
                 post_predictions=post_preds,
                 probabilities=proba)
        print(f"  Break pts (post): {int(post_preds.sum()):,} / {len(post_preds):,}")
        out.append((name, fp, pcd, post_preds, proba))
    return out


def run_stage3_retopologize(detected, results_dir,
                              break_voxel, orig_voxel, force):
    """Adaptive decimation preserving break surface topology."""
    _header("STAGE 3 – RETOPOLOGIZE")
    out = []
    for name, fp, pcd, post_preds, proba in detected:
        npz_path = os.path.join(results_dir, f"{name}_retopo.npz")
        if not force and os.path.exists(npz_path):
            print(f"\n  {name}: retopo found – skipping  ({npz_path})")
            out.append(name)
            continue
        retopologize_fragment(fp,
                               npz_path=os.path.join(results_dir,
                                                      f"{name}_predictions.npz"),
                               break_voxel=break_voxel,
                               orig_voxel=orig_voxel,
                               output_dir=results_dir)
        out.append(name)
    return out


def run_stage4_features(names, results_dir, force):
    """Extract FPFH + PCA geometric features from break surfaces."""
    _header("STAGE 4 – FEATURE EXTRACTION")
    feat_files = []
    for name in names:
        feat_path  = os.path.join(results_dir, f"{name}_features.npz")
        retopo_npz = os.path.join(results_dir, f"{name}_retopo.npz")
        if not force and os.path.exists(feat_path):
            print(f"\n  {name}: features found – skipping  ({feat_path})")
            feat_files.append(feat_path)
            continue
        if not os.path.exists(retopo_npz):
            print(f"  WARNING: {retopo_npz} not found – skipping {name}")
            continue
        extract_and_save(retopo_npz, output_dir=results_dir)
        feat_files.append(feat_path)
    return feat_files


def run_stage5_match(feat_files, results_dir, force):
    """Find matches between all fragment pairs."""
    _header("STAGE 5 – FRAGMENT MATCHING")
    match_path = os.path.join(results_dir, "match_results.npz")

    if not force and os.path.exists(match_path):
        print(f"\n  Match results found – skipping  ({match_path})")
        matches = np.load(match_path, allow_pickle=True)
        # re-build a minimal results list for Stage 6
        results = []
        for i in range(len(matches['scores'])):
            results.append({
                'name_a':         str(matches['names_a'][i]),
                'name_b':         str(matches['names_b'][i]),
                'final_score':    float(matches['scores'][i]),
                'n_inliers':      int(matches['n_inliers'][i]),
                'transformation': matches['transformations'][i].tolist(),
            })
        results.sort(key=lambda r: r['final_score'], reverse=True)
        print_match_report(results)
        return results

    if len(feat_files) < 2:
        print("  Need ≥ 2 fragments for matching – skipping.")
        return []

    results = match_all_fragments(feat_files, verbose=True)
    print_match_report(results)

    np.savez(match_path,
             names_a        = np.array([r['name_a']         for r in results]),
             names_b        = np.array([r['name_b']         for r in results]),
             scores         = np.array([r['final_score']    for r in results]),
             n_inliers      = np.array([r['n_inliers']      for r in results]),
             transformations= np.array([r['transformation'] for r in results]))
    print(f"\n  Match results saved to {match_path}")
    return results


def run_stage6_align(match_results, results_dir, top_k, force):
    """Align the top-K matched pairs with gap-aware ICP."""
    _header("STAGE 6 – ALIGNMENT")
    if not match_results:
        print("  No match results – skipping alignment.")
        return

    top_k = min(top_k, len(match_results))
    print(f"\n  Aligning top-{top_k} pairs…")

    for rank, res in enumerate(match_results[:top_k]):
        na, nb = res['name_a'], res['name_b']
        score  = res['final_score']
        pair   = f"{na}_to_{nb}"

        npz_path = os.path.join(results_dir, f"{pair}_alignment.npz")
        if not force and os.path.exists(npz_path):
            print(f"\n  [{rank+1}] {na} <-> {nb}: alignment found – skipping")
            continue

        print(f"\n  [{rank+1}/{top_k}]  {na}  <->  {nb}  (score={score:.3f})")
        fp_a = os.path.join(results_dir, f"{na}_features.npz")
        fp_b = os.path.join(results_dir, f"{nb}_features.npz")

        feat_a = _load_feat(fp_a)
        feat_b = _load_feat(fp_b)
        if feat_a is None or feat_b is None:
            print("  WARNING: feature files missing – skipping this pair.")
            continue

        init_T = np.array(res['transformation'])
        use_T  = init_T if not np.allclose(init_T, np.eye(4)) else None
        align_fragment_pair(feat_a, feat_b,
                             init_T=use_T,
                             output_dir=results_dir)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end fragment reconstruction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  1 – Preprocess      (dedup / voxel downsample / normals)
  2 – Break detect    (PointNet++ inference + post-processing)
  3 – Retopologize    (adaptive decimation preserving break surfaces)
  4 – Feature extract (FPFH + local PCA shape descriptors)
  5 – Match           (FPFH correspondences + RANSAC + scoring)
  6 – Align           (gap-aware point-to-plane ICP)

Examples:
  python pipeline.py data/without_gt/
  python pipeline.py frag1.ply frag2.ply frag3.ply
  python pipeline.py data/without_gt/ --stages 3 4 5 6
  python pipeline.py data/without_gt/ --force
        """)

    parser.add_argument("paths", nargs="*",
                        help="PLY files or folder(s)  (default: data/without_gt/)")
    parser.add_argument("--stages", nargs="+", type=int,
                        choices=[1, 2, 3, 4, 5, 6], default=[1, 2, 3, 4, 5, 6],
                        help="Which stages to run (default: all)")
    parser.add_argument("--force",       action="store_true",
                        help="Re-run stages even if outputs exist")
    parser.add_argument("--checkpoint",  type=str, default=None,
                        help="Path to model checkpoint  (default: checkpoints/best_model.pt)")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Break probability threshold  (default: 0.5)")
    parser.add_argument("--batch_size",  type=int, default=config.PRED_BATCH_SIZE,
                        help="Inference batch size per GPU")
    parser.add_argument("--voxel",       type=float, default=config.VOXEL_SIZE,
                        help="Voxel size for preprocessing")
    parser.add_argument("--break_voxel", type=float, default=0.3,
                        help="Retopo voxel size for break regions")
    parser.add_argument("--orig_voxel",  type=float, default=1.5,
                        help="Retopo voxel size for original regions")
    parser.add_argument("--top_k",       type=int, default=5,
                        help="Number of top matches to align  (default: 5)")
    args = parser.parse_args()

    t_start     = time.time()
    stages      = set(args.stages)
    results_dir = os.path.join(config.BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    ckpt_path = args.checkpoint or os.path.join(
        config.BASE_DIR, config.CHECKPOINT_DIR, "best_model.pt")

    # ── resolve input files ───────────────────────────────────────────────
    raw_paths = args.paths or [config.DATA_PRED_DIR]
    ply_files = resolve_ply_files(raw_paths)
    if not ply_files:
        print("ERROR: no PLY files found.")
        sys.exit(1)

    _header("HEALING STONES – FRAGMENT RECONSTRUCTION PIPELINE")
    print(f"\n  Fragments  : {len(ply_files)}")
    for f in ply_files:
        print(f"    {os.path.basename(f)}")
    print(f"  Stages     : {sorted(stages)}")
    print(f"  Results dir: {results_dir}")
    print(f"  Force re-run: {args.force}")

    # ── stage 1: preprocess ───────────────────────────────────────────────
    preprocessed = []
    if 1 in stages:
        preprocessed = run_stage1_preprocess(ply_files, args.voxel)
    else:
        # build minimal preprocessed list for downstream stages
        for fp in ply_files:
            name = os.path.splitext(os.path.basename(fp))[0]
            pcd, _ = preprocess(fp, voxel_size=args.voxel, verbose=False)
            preprocessed.append((name, fp, pcd))

    # ── stage 2: break surface detection ─────────────────────────────────
    detected = []
    if 2 in stages:
        model, devices, model_cfg = load_model(ckpt_path)
        detected = run_stage2_detect(preprocessed, model, devices, model_cfg,
                                      results_dir, args.threshold,
                                      args.batch_size, args.force)
    else:
        for name, fp, pcd in preprocessed:
            npz_path = os.path.join(results_dir, f"{name}_predictions.npz")
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                detected.append((name, fp, pcd,
                                  data['post_predictions'],
                                  data['probabilities']))
            else:
                print(f"  WARNING: no predictions for {name} – using zeros")
                n = len(pcd.points)
                detected.append((name, fp, pcd,
                                  np.zeros(n, np.int32),
                                  np.zeros(n, np.float32)))

    # ── stage 3: retopologize ─────────────────────────────────────────────
    names = []
    if 3 in stages:
        names = run_stage3_retopologize(detected, results_dir,
                                         args.break_voxel, args.orig_voxel,
                                         args.force)
    else:
        names = [name for name, *_ in detected]

    # ── stage 4: feature extraction ───────────────────────────────────────
    feat_files = []
    if 4 in stages:
        feat_files = run_stage4_features(names, results_dir, args.force)
    else:
        feat_files = [os.path.join(results_dir, f"{n}_features.npz")
                      for n in names
                      if os.path.exists(
                          os.path.join(results_dir, f"{n}_features.npz"))]

    # ── stage 5: matching ─────────────────────────────────────────────────
    match_results = []
    if 5 in stages:
        match_results = run_stage5_match(feat_files, results_dir, args.force)
    else:
        match_path = os.path.join(results_dir, "match_results.npz")
        if os.path.exists(match_path):
            matches = np.load(match_path, allow_pickle=True)
            for i in range(len(matches['scores'])):
                match_results.append({
                    'name_a':         str(matches['names_a'][i]),
                    'name_b':         str(matches['names_b'][i]),
                    'final_score':    float(matches['scores'][i]),
                    'n_inliers':      int(matches['n_inliers'][i]),
                    'transformation': matches['transformations'][i].tolist(),
                })
            match_results.sort(key=lambda r: r['final_score'], reverse=True)

    # ── stage 6: alignment ────────────────────────────────────────────────
    if 6 in stages:
        run_stage6_align(match_results, results_dir, args.top_k, args.force)

    # ── final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    _header(f"PIPELINE COMPLETE  ({elapsed/60:.1f} min)")

    print(f"\n  Fragments processed : {len(ply_files)}")
    print(f"  Results written to  : {results_dir}")

    if match_results:
        print(f"\n  Top matches:")
        print(f"  {'Pair':<40} {'Score':>7} {'Inliers':>8}")
        print(f"  {'-'*58}")
        for r in match_results[:args.top_k]:
            pair = f"{r['name_a']}  <->  {r['name_b']}"
            print(f"  {pair:<40} {r['final_score']:>7.3f} {r['n_inliers']:>8}")

    print()


if __name__ == "__main__":
    main()