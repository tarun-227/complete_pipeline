"""
export_transforms.py  –  Run this INSIDE Blender's Script Editor
=================================================================
Steps:
  1. Open Blender
  2. Open final.blend1  (File > Open)
  3. Switch to the Scripting workspace (top tab bar)
  4. Click "New", paste this entire file, click "Run Script"
  5. A file called  fragment_transforms.json  will be saved next to the .blend

What it exports:
  - Name of every mesh object in the scene
  - 4x4 world-space transformation matrix
  - World-space bounding box corners  (8 corners × 3 coords)
  - World-space centroid
"""

import bpy
import json
import os
import mathutils

# ── output path: same folder as the blend file ────────────────────────────────
blend_path = bpy.data.filepath
output_dir  = os.path.dirname(blend_path) if blend_path else os.path.expanduser("~")
output_path = os.path.join(output_dir, "fragment_transforms.json")

# ── collect data ──────────────────────────────────────────────────────────────
objects_data = []

for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue

    # 4×4 world matrix (row-major)
    T = obj.matrix_world
    matrix_list = [list(row) for row in T]

    # World-space bounding box (8 corners)
    bbox_world = [list(obj.matrix_world @ mathutils.Vector(corner))
                  for corner in obj.bound_box]

    # World-space centroid  (mean of bbox corners)
    cx = sum(c[0] for c in bbox_world) / 8
    cy = sum(c[1] for c in bbox_world) / 8
    cz = sum(c[2] for c in bbox_world) / 8

    # Location, rotation (Euler XYZ degrees), scale
    loc   = list(obj.location)
    rot   = [math.degrees(a) for a in obj.rotation_euler]
    scale = list(obj.scale)

    objects_data.append({
        "name"       : obj.name,
        "matrix"     : matrix_list,          # 4×4 world transform
        "location"   : loc,                  # world location
        "bbox_world" : bbox_world,           # 8 corners in world space
        "centroid"   : [cx, cy, cz],         # bbox centroid
    })

    print(f"  Exported: {obj.name}  loc={[round(x,2) for x in loc]}")

# ── save ──────────────────────────────────────────────────────────────────────
import math   # needed for math.degrees above

with open(output_path, "w") as f:
    json.dump({"objects": objects_data}, f, indent=2)

print(f"\n✓  Saved {len(objects_data)} objects to:\n   {output_path}")
