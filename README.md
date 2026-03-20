# Healing Stones — Break Surface Detection & Fragment Reconstruction Pipeline

> **Google Summer of Code / HSF Project**
> Reconstructing digitized cultural heritage artifacts using AI/ML — applied to a fragmented ancient Maya monument.

---

## Overview

This pipeline takes raw 3D scans of stone fragments, identifies which surfaces are **break surfaces** (where the stone was fractured), and uses those surfaces to find which fragments fit together — ultimately producing a reconstruction proposal for the assembled monument.

The pipeline is built around a **PointNet++ classifier** trained on green-painted ground-truth annotations, followed by geometry-based matching and ICP alignment. No manual feature engineering step is required; the model learns directly from point clouds.

```
Raw PLY scans
    │
    ▼  Stage 1 — Preprocess
Dedup · Voxel downsample · SOR outlier removal · Normal estimation
    │
    ▼  Stage 2 — Detect break surfaces
PointNet++ inference → per-point break/original probability → 5-stage post-processing
    │
    ▼  Stage 3 — Retopologize
Adaptive voxel decimation (fine on break surfaces, coarse elsewhere)
    │
    ▼  Stage 4 — Feature extraction
FPFH descriptors + local PCA shape features on break surface points
    │
    ▼  Stage 5 — Fragment matching
MNN matching · Lowe ratio test · RANSAC · ML re-ranking (LogReg)
    │
    ▼  Stage 6 — Alignment
Gap-aware point-to-plane ICP · overlap & quality scoring
    │
    ▼  results/   — ranked match list + aligned PLY visualizations
```

---

## Repository Layout

```
break_surface/
├── data/
│   ├── with_gt/          # GT fragments for training (PLY with green vertex colors)
│   └── without_gt/       # Fragments to reconstruct (plain PLY scans)
├── model/
│   ├── pointnet2.py      # PointNet++ architecture (3×SA + global SA + MLP head)
│   └── dataset.py        # BreakSurfaceDataset — local KNN patches, balanced sampling
├── checkpoints/
│   └── best_model.pt     # Saved after training (required for predict / pipeline)
├── results/              # All outputs land here
├── config.py             # All hyperparameters and paths
├── preprocess.py         # Voxel downsample, SOR, normal estimation, GT label extraction
├── train.py              # PointNet++ training loop
├── predict.py            # Inference + 5-stage post-processing
├── retopologize.py       # Adaptive voxel decimation (break vs original regions)
├── feature_extraction.py # FPFH + local PCA features on break surface points
├── fragment_matching.py  # Mutual-NN matching, RANSAC, ML scorer, ranked pair list
├── alignment.py          # Gap-aware ICP, quality scoring, PLY visualizations
├── pipeline.py           # End-to-end runner (Stages 1–6 with caching)
├── analyze_adjacency.py  # World-space adjacency map from Blender positions + PLY data
└── export_transforms.py  # Blender script: exports fragment world transforms to JSON
```

---

## Setup

### Requirements

- Python 3.10 – 3.12
- CUDA-capable GPU strongly recommended for training and inference (CPU works but is slow)

### Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118   # GPU (CUDA 11.8)
# or
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu     # CPU only

pip install open3d scipy scikit-learn numpy tqdm
```

### Directory setup

```bash
mkdir -p data/with_gt data/without_gt checkpoints results
```

Place your GT-labeled PLY files (with green vertex colors on break surfaces) in `data/with_gt/`.
Place your unlabeled PLY scans in `data/without_gt/`.

---

## Quick Start — Full Pipeline

```bash
# Run all 6 stages on the fragments in data/without_gt/
python pipeline.py data/without_gt/

# Run only specific stages
python pipeline.py data/without_gt/ --stages 1 2 3

# Force re-run (ignore cached results)
python pipeline.py data/without_gt/ --force

# Align only the top-3 matches instead of top-5
python pipeline.py data/without_gt/ --top_k 3
```

Outputs are cached between runs — stages skip automatically if their output already exists.

---

## Step-by-Step Commands

### Stage 1 — Preprocess

Cleans each PLY: removes duplicate points, applies voxel downsampling, SOR outlier removal, and estimates normals.

```bash
# Runs automatically as part of pipeline.py Stage 1
# Cached results: data/without_gt/*_preprocessed.npz
```

Key hyperparameters in `config.py`:

| Parameter        | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `VOXEL_SIZE`    | 0.5     | Voxel size for downsampling          |
| `SOR_NEIGHBORS` | 20      | KNN count for outlier removal        |
| `SOR_STD`       | 2.0     | Std-dev multiplier threshold         |
| `KNN_NORMALS`   | 30      | Neighbours for normal estimation     |

---

### Stage 2 — Train & Detect Break Surfaces

**Train** (requires GT-labeled fragments in `data/with_gt/`):

```bash
python train.py
python train.py --epochs 200 --batch_size 64 --gpu cuda:0
```

Saves `checkpoints/best_model.pt`. Training flags:

| Flag                  | Default | Description                         |
|-----------------------|---------|-------------------------------------|
| `--epochs`            | 10      | Training epochs                     |
| `--batch_size`        | 32      | Batch size                          |
| `--lr`                | 0.001   | Learning rate                       |
| `--gpu`               | cuda:0  | GPU device string                   |
| `--num_points`        | 4096    | Points per patch sample             |
| `--samples_per_frag`  | 200     | Patch samples per fragment          |
| `--voxel`             | 0.5     | Voxel downsample size               |

**Predict** on new fragments (requires `checkpoints/best_model.pt`):

```bash
python predict.py data/without_gt/
python predict.py data/without_gt/ --threshold 0.4   # lower = more break surface
```

Post-processing stages (applied automatically after raw inference):

1. Threshold: keep points with `P(break) > threshold`
2. Probability pull-in: expand into nearby high-probability points
3. Iterative fill: grow break region into the connected neighbourhood
4. Erosion: remove isolated noise at the boundary
5. DBSCAN: drop tiny disconnected clusters

Outputs: `results/*_predictions.npz` (per-point probabilities + binary labels)

---

### Stage 3 — Retopologize

Decimates each fragment with two different voxel sizes: fine resolution on break surfaces, coarse on the original (non-break) surface. This keeps break surface detail while reducing total point count.

```bash
python retopologize.py data/without_gt/FR_01.ply results/FR_01_predictions.npz
python retopologize.py data/without_gt/FR_01.ply results/FR_01_predictions.npz \
    --break_voxel 0.3 --orig_voxel 1.5
```

| Flag            | Default | Description                                    |
|----------------|---------|------------------------------------------------|
| `--break_voxel` | 0.3     | Voxel size on break surface (finer = more pts) |
| `--orig_voxel`  | 1.5     | Voxel size on original surface                 |

Output: `*_retopo.ply` + `*_retopo.npz`

---

### Stage 4 — Feature Extraction

Computes per-point descriptors on break surface points:

- **FPFH** (33-dim): rotation-invariant histogram of normal angle differences
- **Local PCA** (6-dim): linearity, planarity, sphericity, omnivariance, curvature, normal deviation

Also computes a fragment-level summary descriptor (mean normal, roughness, area proxy, etc.) used for fast compatibility screening.

```bash
python feature_extraction.py results/FR_01_retopo.npz
```

Output: `results/*_features.npz` + `results/*_summary.npz`

---

### Stage 5 — Fragment Matching

For each pair of fragments:

1. **Compatibility screen** — fast scoring from normal anti-parallelism, size ratio, roughness, curvature (eliminates clearly incompatible pairs)
2. **FPFH descriptor matching** — mutual nearest-neighbour + Lowe's ratio test (ratio=0.9)
3. **RANSAC** — geometric consistency check on correspondences (50 000 iterations, 2-unit inlier threshold)
4. **ML re-ranking** — logistic regression on 11-dim match feature vector (re-ranks after RANSAC)

```bash
python fragment_matching.py results/
```

Output: `results/match_results.npz` — ranked list of all pairs with scores

---

### Stage 6 — Alignment

Takes the top-K matches and refines them with **gap-aware point-to-plane ICP**:

- 3 rounds of ICP alternating with dynamic gap threshold updates
- Gap threshold = 1.5 × 85th-percentile of residuals (adapts to surface irregularity)
- Outputs overlap fraction, RMS inlier distance, normal anti-parallelism score

```bash
python alignment.py results/ --top_k 5
```

Output per aligned pair:
- `results/*_aligned.ply` — combined point cloud (source=red, target=blue)
- `results/*_alignment.npz` — 4×4 transform matrix + quality metrics

---

## Adjacency Analysis (Blender-Assisted)

If you have a rough assembly in Blender with fragment world-space positions, you can generate an adjacency map before running the full pipeline:

**Step 1** — Export fragment transforms from Blender:

```
# In Blender: Text Editor → Open export_transforms.py → Run Script
# Output: fragment_transforms.json  (world-space bboxes and centroids)
```

**Step 2** — Run adjacency analysis:

```bash
python analyze_adjacency.py
python analyze_adjacency.py --threshold 2.0   # max point distance to count as adjacent
```

Output: `results/adjacency_map.json` — which fragments share a break surface, with minimum point-to-point distances.

Use the adjacency map to seed the ML scorer with positive/negative pair labels for more accurate match re-ranking.

---

## Model Architecture

**PointNet++** with three Set Abstraction layers followed by a global aggregation and MLP classifier:

```
Input (B, 6, N)  — XYZ + surface normals
  │
  ├─ SA1: radius=2.0,  K=32,  MLP [64, 64, 128],    npoint=1024
  ├─ SA2: radius=4.0,  K=64,  MLP [128, 128, 256],  npoint=256
  ├─ SA3: radius=8.0,  K=128, MLP [256, 512, 1024], npoint=64
  └─ GlobalSA:         MLP [256, 512, 1024]
        │
        └─ MLP [512, 256] → Dropout(0.4) → Linear(2) → logits
```

Output: per-sample logits `(B, 2)` for [original, break] + global feature `(B, 1024)`.
Class imbalance handled via weighted cross-entropy (`BREAK_WEIGHT = 7.0`).

---

## Configuration Reference

All hyperparameters live in `config.py`:

```python
# Paths
DATA_GT_DIR    = "data/with_gt/"       # GT-labeled training fragments
DATA_PRED_DIR  = "data/without_gt/"    # Fragments to reconstruct
CHECKPOINT_DIR = "checkpoints/"        # Saved model weights
RESULTS_DIR    = "results/"            # All outputs

# Preprocessing
VOXEL_SIZE   = 0.5    # Downsample voxel size (None = no downsampling)
SOR_NEIGHBORS = 20    # SOR KNN count
SOR_STD       = 2.0   # SOR std-dev threshold

# Training
BATCH_SIZE    = 32
LEARNING_RATE = 0.001
EPOCHS        = 10
PATIENCE      = 15    # Early stopping
BREAK_WEIGHT  = 7.0   # Upweight minority break class

# Model
INPUT_CHANNELS = 6    # XYZ + normals
DROPOUT        = 0.4

# Prediction
PRED_BATCH_SIZE = 64
PRED_THRESHOLD  = 0.5  # Break surface probability cutoff
```

---

## Outputs Reference

| File | Stage | Description |
|------|-------|-------------|
| `data/without_gt/*_preprocessed.npz` | 1 | Cleaned points, normals |
| `results/*_predictions.npz` | 2 | Per-point break probabilities + labels |
| `results/*_retopo.ply` | 3 | Adaptive-resolution point cloud |
| `results/*_retopo.npz` | 3 | Same with normals + labels |
| `results/*_features.npz` | 4 | Per-point FPFH + PCA descriptors |
| `results/*_summary.npz` | 4 | Fragment-level summary descriptor |
| `results/match_results.npz` | 5 | Ranked pair list with match scores |
| `results/*_aligned.ply` | 6 | Source (red) + target (blue) combined cloud |
| `results/*_alignment.npz` | 6 | 4×4 transform + quality metrics |
| `results/adjacency_map.json` | — | World-space fragment adjacency map |

---

## Server / GPU Workflow

The pipeline is designed to run on a remote GPU server. Recommended sequence:

```bash
# 1. Clone/copy repo to server, install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open3d scipy scikit-learn numpy tqdm

# 2. Copy PLY scans to data/
scp data/without_gt/*.ply user@server:~/break_surface/data/without_gt/
scp data/with_gt/*.ply    user@server:~/break_surface/data/with_gt/

# 3. Train on GT fragments (fast with GPU)
python train.py --epochs 100 --batch_size 64 --gpu cuda:0

# 4. Run full pipeline on unlabeled fragments
python pipeline.py data/without_gt/ --top_k 5

# 5. Copy results back
scp -r user@server:~/break_surface/results/ ./results/
```

Approximate GPU memory usage: ~4 GB for batch_size=64, NUM_POINTS=4096.
CPU fallback works but Stage 2 inference will be significantly slower.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
Install PyTorch for your platform — see Setup section above.

**`ERROR: Checkpoint not found at 'checkpoints/best_model.pt'`**
Run `python train.py` first, or copy a pre-trained checkpoint into `checkpoints/`.

**`WARNING: No CUDA GPUs found — running on CPU`**
Normal on a CPU-only machine. For training, a GPU is strongly recommended.
For inference only, CPU is usable but slow (~10–30 min per fragment).

**Blender export `NameError: name 'math' is not defined`**
Add `import math` at the top of `export_transforms.py` in the Blender text editor.

**Adjacency analysis matches fewer PLY files than expected**
The PLY filenames (e.g. `FR_01.ply`) and JSON fragment names (e.g. `frag_1`) are matched by 3D centroid proximity, not by name. If matching fails, verify that the PLY files are in world-space coordinates (not local Blender object space).

---

## Citation / Acknowledgements

This project is developed as part of the **HSF / Google Summer of Code** programme.
Model architecture based on [PointNet++](https://arxiv.org/abs/1706.02413) (Qi et al., 2017).
Point cloud I/O via [Open3D](http://www.open3d.org/).

---

*For questions or issues, open a ticket in the project repository.*
