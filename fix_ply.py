"""
fix_ply.py - Fix corrupted PLY files
Uses plyfile library (tolerant reader) to extract data,
then re-saves as clean PLY that Open3D can read.

Usage:
  pip install plyfile
  python3 fix_ply.py input.ply
  python3 fix_ply.py --all
"""
import os
import sys
import glob
import argparse
import numpy as np


def fix_single(input_path, output_path=None):
    try:
        from plyfile import PlyData
    except ImportError:
        print("  Install plyfile first: pip install plyfile")
        return False

    if output_path is None:
        output_path = input_path

    print(f"\n  Reading {input_path} with plyfile...")
    try:
        plydata = PlyData.read(input_path)
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    vertex = plydata['vertex']
    n = len(vertex)
    print(f"  Vertices: {n:,}")

    x = np.array(vertex['x'], dtype=np.float64)
    y = np.array(vertex['y'], dtype=np.float64)
    z = np.array(vertex['z'], dtype=np.float64)

    has_colors = 'red' in vertex.data.dtype.names
    if has_colors:
        r = np.array(vertex['red'], dtype=np.float64) / 255.0
        g = np.array(vertex['green'], dtype=np.float64) / 255.0
        b = np.array(vertex['blue'], dtype=np.float64) / 255.0
        print(f"  Has colors: YES")
    else:
        print(f"  Has colors: NO")

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack([x, y, z]))
    if has_colors:
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack([r, g, b]))

    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved: {output_path} ({size_mb:.1f} MB)")

    pcd2 = o3d.io.read_point_cloud(output_path)
    print(f"  Verify: Open3D reads {len(pcd2.points):,} points OK")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs='?', default=None)
    parser.add_argument("output", nargs='?', default=None)
    parser.add_argument("--all", action='store_true')
    args = parser.parse_args()

    if args.all:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for d in ["data/with_gt", "data/without_gt"]:
            full = os.path.join(base_dir, d)
            if not os.path.exists(full):
                continue
            files = sorted(glob.glob(os.path.join(full, "*.ply"))
                           + glob.glob(os.path.join(full, "*.PLY")))
            for fp in files:
                fix_single(fp)
    elif args.input:
        fix_single(args.input, args.output)
    else:
        print("Usage: python3 fix_ply.py input.ply [output.ply]")
        print("       python3 fix_ply.py --all")


if __name__ == "__main__":
    main()