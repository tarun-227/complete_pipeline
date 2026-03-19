"""
pointnet2.py - PointNet++ Architecture for Point Classification
===============================================================
Simplified PointNet++ that classifies individual points as break/original
based on their local neighborhood.

Architecture:
  Input: (B, C, N) where C=6 (XYZ+normals), N=4096 points
  -> Set Abstraction Layer 1: N=1024, radius=0.5, K=32
  -> Set Abstraction Layer 2: N=256, radius=1.0, K=64
  -> Set Abstraction Layer 3: N=64, radius=2.0, K=128
  -> Global feature from all 64 points
  -> MLP classifier -> break/original
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between each pair of points.
    src: (B, N, C)
    dst: (B, M, C)
    Returns: (B, N, M)
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Farthest point sampling.
    xyz: (B, N, 3)
    npoint: number of points to sample
    Returns: (B, npoint) indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]

    return centroids


def index_points(points, idx):
    """
    Index into points tensor.
    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    Returns: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query: find nsample points within radius for each center.
    xyz: (B, N, 3) all points
    new_xyz: (B, S, 3) center points
    Returns: (B, S, nsample) indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)  # (B, S, N)

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # Handle case where fewer than nsample points in ball
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class SetAbstraction(nn.Module):
    """
    Set Abstraction layer: samples + groups + applies PointNet on each group.
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels):
        """
        Args:
            npoint: Number of points to sample (output size)
            radius: Ball query radius
            nsample: Max points per ball
            in_channel: Input feature channels (includes XYZ=3)
            mlp_channels: List of MLP layer sizes
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) coordinates
            points: (B, N, D) features (or None)

        Returns:
            new_xyz: (B, npoint, 3) sampled coordinates
            new_points: (B, npoint, D') features
        """
        # Sample
        fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        # Group
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
        grouped_xyz -= new_xyz.unsqueeze(2)  # center each group

        if points is not None:
            grouped_points = index_points(points, idx)  # (B, npoint, nsample, D)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz

        # (B, npoint, nsample, C) -> (B, C, nsample, npoint) for Conv2d
        grouped_points = grouped_points.permute(0, 3, 2, 1)

        # Apply MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            grouped_points = F.relu(bn(conv(grouped_points)))

        # Max pool over each group
        new_points = torch.max(grouped_points, 2)[0]  # (B, C', npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, C')

        return new_xyz, new_points


class GlobalSetAbstraction(nn.Module):
    """
    Global set abstraction: aggregates ALL points into a single feature.
    """

    def __init__(self, in_channel, mlp_channels):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        last_channel = in_channel
        for out_channel in mlp_channels:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3)
            points: (B, N, D)

        Returns:
            new_points: (B, D') global feature
        """
        if points is not None:
            features = torch.cat([xyz, points], dim=-1)
        else:
            features = xyz

        features = features.permute(0, 2, 1)  # (B, C, N)

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            features = F.relu(bn(conv(features)))

        # Global max pool
        features = torch.max(features, 2)[0]  # (B, C')

        return features


class PointNet2Classifier(nn.Module):
    """
    PointNet++ for binary point classification (break vs original).

    Input: (B, C, N) where C=6 (XYZ+normals), N=4096
    Output: (B, 2) class logits
    """

    def __init__(self, input_channels=6, dropout=0.4):
        super().__init__()

        # Extra feature channels (input_channels - 3 for XYZ)
        extra_channels = input_channels - 3

        # Set Abstraction layers
        # SA1: 4096 -> 1024 points
        self.sa1 = SetAbstraction(
            npoint=1024, radius=0.5, nsample=32,
            in_channel=3 + extra_channels,
            mlp_channels=[64, 64, 128])

        # SA2: 1024 -> 256 points
        self.sa2 = SetAbstraction(
            npoint=256, radius=1.0, nsample=64,
            in_channel=128 + 3,
            mlp_channels=[128, 128, 256])

        # SA3: 256 -> 64 points
        self.sa3 = SetAbstraction(
            npoint=64, radius=2.0, nsample=128,
            in_channel=256 + 3,
            mlp_channels=[256, 256, 512])

        # Global aggregation
        self.global_sa = GlobalSetAbstraction(
            in_channel=512 + 3,
            mlp_channels=[512, 1024])

        # Classifier
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        """
        Args:
            x: (B, C, N) input point cloud with features

        Returns:
            logits: (B, 2) class logits
            features: (B, 1024) global feature (for visualization)
        """
        B, C, N = x.shape

        # Split XYZ and features
        xyz = x[:, :3, :].permute(0, 2, 1)  # (B, N, 3)
        if C > 3:
            features = x[:, 3:, :].permute(0, 2, 1)  # (B, N, C-3)
        else:
            features = None

        # Set Abstraction layers
        xyz1, points1 = self.sa1(xyz, features)    # (B, 1024, 128)
        xyz2, points2 = self.sa2(xyz1, points1)     # (B, 256, 256)
        xyz3, points3 = self.sa3(xyz2, points2)     # (B, 64, 512)

        # Global feature
        global_feat = self.global_sa(xyz3, points3)  # (B, 1024)

        # Classifier
        out = self.drop1(F.relu(self.bn1(self.fc1(global_feat))))
        out = self.drop2(F.relu(self.bn2(self.fc2(out))))
        logits = self.fc3(out)  # (B, 2)

        return logits, global_feat


if __name__ == "__main__":
    # Quick test
    model = PointNet2Classifier(input_channels=6)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(4, 6, 4096)
    logits, feat = model(x)
    print(f"Input: {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Features: {feat.shape}")