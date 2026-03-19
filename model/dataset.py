"""
dataset.py - Point Cloud Dataset for PointNet++
================================================
Extracts local neighborhood patches around points for training.
Each sample = a local patch of NUM_POINTS_PER_SAMPLE points with label.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

from preprocess import preprocess, extract_gt
import config


class BreakSurfaceDataset(Dataset):
    """
    Dataset that extracts local patches around points.

    For each sample:
    - Pick a center point
    - Extract its neighborhood (NUM_POINTS_PER_SAMPLE nearest neighbors)
    - Center the patch (subtract centroid) -> rotation invariant XYZ
    - Return (points, label) where points is (N, C) and label is 0 or 1
    """

    def __init__(self, fragments, num_points=config.NUM_POINTS_PER_SAMPLE,
                 use_normals=config.USE_NORMALS, augment=True, balance=True,
                 samples_per_fragment=50000):
        """
        Args:
            fragments: List of (points, normals, labels) tuples
            num_points: Points per local patch
            use_normals: Include normals as features
            augment: Apply data augmentation
            balance: Balance break/original AND across fragments
            samples_per_fragment: Samples per fragment per class
        """
        self.num_points = num_points
        self.use_normals = use_normals
        self.augment = augment

        # Combine all fragments, tracking boundaries
        all_points = []
        all_normals = []
        all_labels = []
        frag_boundaries = [0]

        for pts, norms, labs in fragments:
            all_points.append(pts)
            all_normals.append(norms)
            all_labels.append(labs)
            frag_boundaries.append(frag_boundaries[-1] + len(pts))

        self.points = np.vstack(all_points).astype(np.float32)
        self.normals = np.vstack(all_normals).astype(np.float32)
        self.labels = np.concatenate(all_labels).astype(np.int64)

        # Build KDTree on full point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)
        self.tree = o3d.geometry.KDTreeFlann(pcd)

        # Balance ACROSS FRAGMENTS and across classes
        # Each fragment contributes equal samples, each class within fragment is balanced
        n_per_class_per_frag = samples_per_fragment // 2
        all_selected = []

        for fi in range(len(fragments)):
            start = frag_boundaries[fi]
            end = frag_boundaries[fi + 1]
            frag_labels = self.labels[start:end]

            frag_break = np.where(frag_labels == 1)[0] + start  # global indices
            frag_orig = np.where(frag_labels == 0)[0] + start

            # Sample break points from this fragment
            if len(frag_break) >= n_per_class_per_frag:
                break_sel = np.random.choice(frag_break, n_per_class_per_frag, replace=False)
            else:
                break_sel = np.random.choice(frag_break, n_per_class_per_frag, replace=True)

            # Sample original points from this fragment
            if len(frag_orig) >= n_per_class_per_frag:
                orig_sel = np.random.choice(frag_orig, n_per_class_per_frag, replace=False)
            else:
                orig_sel = np.random.choice(frag_orig, n_per_class_per_frag, replace=True)

            all_selected.extend(break_sel)
            all_selected.extend(orig_sel)

            n_b = len(frag_break)
            n_o = len(frag_orig)
            print(f"    Frag {fi+1}: {n_b:,} break, {n_o:,} orig -> "
                  f"sampled {n_per_class_per_frag:,} each")

        self.indices = np.array(all_selected)
        np.random.shuffle(self.indices)

        n_break = (self.labels[self.indices] == 1).sum()
        n_orig = len(self.indices) - n_break
        print(f"  Dataset: {len(self.indices):,} samples "
              f"(break={n_break:,}, orig={n_orig:,}, "
              f"{len(fragments)} fragments equally weighted)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        center_idx = self.indices[idx]
        center = self.points[center_idx]
        label = self.labels[center_idx]

        # Find neighborhood
        k, nn_idx, nn_dist = self.tree.search_knn_vector_3d(
            center, self.num_points + 1)
        nn_idx = np.array(nn_idx[1:])  # exclude self

        # Pad if not enough neighbors
        if len(nn_idx) < self.num_points:
            pad = np.random.choice(nn_idx, self.num_points - len(nn_idx), replace=True)
            nn_idx = np.concatenate([nn_idx, pad])
        nn_idx = nn_idx[:self.num_points]

        # Get neighbor points and center them
        patch_points = self.points[nn_idx] - center  # centered at origin
        patch_normals = self.normals[nn_idx]

        # Data augmentation
        if self.augment:
            patch_points, patch_normals = self._augment(patch_points, patch_normals)

        # Build feature vector
        if self.use_normals:
            features = np.concatenate([patch_points, patch_normals], axis=1)  # (N, 6)
        else:
            features = patch_points  # (N, 3)

        # Transpose for PointNet++ (C, N) format
        features = features.T.astype(np.float32)  # (C, N)

        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

    def _augment(self, points, normals):
        """Random rotation around Z + jitter + random scale."""
        # Random rotation around vertical axis (Z)
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0],
                       [sin_t, cos_t, 0],
                       [0, 0, 1]], dtype=np.float32)
        points = points @ R.T
        normals = normals @ R.T

        # Random full 3D rotation (small angle)
        for axis in range(3):
            angle = np.random.uniform(-0.1, 0.1)  # small rotation
            c, s = np.cos(angle), np.sin(angle)
            if axis == 0:
                R2 = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
            elif axis == 1:
                R2 = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
            else:
                R2 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            points = points @ R2.T
            normals = normals @ R2.T

        # Random jitter
        points += np.random.normal(0, 0.005, points.shape).astype(np.float32)

        # Random scale
        scale = np.random.uniform(0.9, 1.1)
        points *= scale

        return points, normals


def load_fragments(gt_dir, voxel_size=config.VOXEL_SIZE):
    """
    Load all GT fragments from directory.

    Returns:
        fragments: List of (points, normals, labels) tuples
        frag_names: List of fragment names
    """
    files = sorted(glob.glob(os.path.join(gt_dir, "*.ply"))
                   + glob.glob(os.path.join(gt_dir, "*.PLY")))

    if not files:
        print(f"  No PLY files in {gt_dir}")
        return [], []

    fragments = []
    frag_names = []

    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n  Loading {name}...")

        pcd, log = preprocess(fp, voxel_size=voxel_size)
        points = np.asarray(pcd.points).astype(np.float32)
        normals = np.asarray(pcd.normals).astype(np.float32)
        labels = extract_gt(pcd)

        n_break = labels.sum()
        n_total = len(labels)
        print(f"  {name}: {n_total:,} pts, {n_break:,} break ({100*n_break/n_total:.1f}%)")

        fragments.append((points, normals, labels))
        frag_names.append(name)

    return fragments, frag_names


def load_predict_fragments(pred_dir, voxel_size=config.VOXEL_SIZE):
    """
    Load fragments for prediction (no GT needed).

    Returns:
        fragments: List of (points, normals) tuples
        frag_names: List of names
    """
    files = sorted(glob.glob(os.path.join(pred_dir, "*.ply"))
                   + glob.glob(os.path.join(pred_dir, "*.PLY")))

    if not files:
        print(f"  No PLY files in {pred_dir}")
        return [], []

    fragments = []
    frag_names = []

    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        print(f"\n  Loading {name}...")

        pcd, log = preprocess(fp, voxel_size=voxel_size)
        points = np.asarray(pcd.points).astype(np.float32)
        normals = np.asarray(pcd.normals).astype(np.float32)

        print(f"  {name}: {len(points):,} pts")

        fragments.append((points, normals, pcd))
        frag_names.append(name)

    return fragments, frag_names