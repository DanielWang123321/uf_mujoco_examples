#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, UFACTORY Inc.
"""Generate 1:1 3D meshes (OBJ/STL) and MuJoCo MJCF model for a paper cup.

Dimensions:
- Height: 88 mm (0.088 m)
- Bottom outer diameter: 53 mm (0.053 m)
- Top outer diameter: 75 mm (0.075 m)
- Wall thickness: 1 mm (0.001 m)
- Total mass: 5.74 g (0.00574 kg)
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

HEIGHT = 0.088
D_BOT_OUT = 0.053
D_TOP_OUT = 0.075
THICKNESS = 0.001
TOTAL_MASS = 0.00574  # 5.74g disposable paper cup
BOT_MASS = 0.001180
WALL_MASS = 0.000190  # 0.001180 + 24 * 0.000190 == 0.005740 kg


def build_paper_cup_mesh(n_seg: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Build watertight triangle mesh (vertices, faces) for the hollow paper cup."""
    r_bot_out = D_BOT_OUT / 2.0
    r_top_out = D_TOP_OUT / 2.0
    slope = (r_top_out - r_bot_out) / HEIGHT
    # Normal thickness offset along slanted wall
    sec_alpha = math.hypot(1.0, slope)
    r_top_in = r_top_out - THICKNESS * sec_alpha
    z_bot_in = THICKNESS
    r_bot_in = r_bot_out + slope * z_bot_in - THICKNESS * sec_alpha

    angles = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # 4 rings of n_seg vertices + 2 center vertices (bottom outer, bottom inner)
    # Ring 0: outer bottom (z = 0, r = r_bot_out)
    # Ring 1: outer top    (z = HEIGHT, r = r_top_out)
    # Ring 2: inner top    (z = HEIGHT, r = r_top_in)
    # Ring 3: inner bottom (z = z_bot_in, r = r_bot_in)
    # Vertex 4*n_seg:     outer bottom center (0, 0, 0)
    # Vertex 4*n_seg + 1: inner bottom center (0, 0, z_bot_in)
    verts = []
    for r, z in [
        (r_bot_out, 0.0),
        (r_top_out, HEIGHT),
        (r_top_in, HEIGHT),
        (r_bot_in, z_bot_in),
    ]:
        ring = np.column_stack([r * cos_a, r * sin_a, np.full_like(angles, z)])
        verts.append(ring)

    verts.append(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, z_bot_in]]))
    vertices = np.vstack(verts)

    faces = []
    c_out = 4 * n_seg
    c_in = 4 * n_seg + 1

    for i in range(n_seg):
        j = (i + 1) % n_seg

        # 1. Outer wall (Ring 0 -> Ring 1), outward normal
        v00, v01 = i, j
        v10, v11 = n_seg + i, n_seg + j
        faces.append([v00, v01, v11])
        faces.append([v00, v11, v10])

        # 2. Top rim (Ring 1 -> Ring 2), +Z normal
        v20, v21 = 2 * n_seg + i, 2 * n_seg + j
        faces.append([v10, v11, v21])
        faces.append([v10, v21, v20])

        # 3. Inner wall (Ring 2 -> Ring 3), inward normal
        v30, v31 = 3 * n_seg + i, 3 * n_seg + j
        faces.append([v20, v21, v31])
        faces.append([v20, v31, v30])

        # 4. Outer bottom disk (Ring 0 -> c_out), -Z normal
        faces.append([v01, v00, c_out])

        # 5. Inner bottom disk (Ring 3 -> c_in), +Z normal
        faces.append([v30, v31, c_in])

    return vertices, np.asarray(faces, dtype=np.int32)


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write mesh to Wavefront OBJ file."""
    lines = [
        "# Paper cup 3D mesh (units: meters)",
        (
            f"# Height: {HEIGHT * 1000:.1f}mm, Bottom OD: {D_BOT_OUT * 1000:.1f}mm, "
            f"Top OD: {D_TOP_OUT * 1000:.1f}mm, Thickness: {THICKNESS * 1000:.1f}mm, "
            f"Mass: {TOTAL_MASS * 1000:.2f}g"
        ),
        "o paper_cup",
    ]
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for f in faces:
        # OBJ indices are 1-based
        lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write mesh to binary STL file."""
    header = b"Paper cup mesh (units: m)".ljust(80, b"\0")
    tris = vertices[faces]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(norms, 1e-12))

    with path.open("wb") as fp:
        fp.write(header)
        fp.write(struct.pack("<I", len(faces)))
        for n, tri in zip(normals, tris, strict=True):
            fp.write(
                struct.pack(
                    "<12fH",
                    float(n[0]),
                    float(n[1]),
                    float(n[2]),
                    float(tri[0, 0]),
                    float(tri[0, 1]),
                    float(tri[0, 2]),
                    float(tri[1, 0]),
                    float(tri[1, 1]),
                    float(tri[1, 2]),
                    float(tri[2, 0]),
                    float(tri[2, 1]),
                    float(tri[2, 2]),
                    0,
                )
            )


def rot_z_y_to_quat(theta_z: float, alpha_y: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for R = Rz(theta_z) @ Ry(alpha_y)."""
    cz = math.cos(theta_z * 0.5)
    sz = math.sin(theta_z * 0.5)
    cy = math.cos(alpha_y * 0.5)
    sy = math.sin(alpha_y * 0.5)
    # qz = (cz, 0, 0, sz), qy = (cy, 0, sy, 0)
    w = cz * cy
    x = -sz * sy
    y = cz * sy
    z = sz * cy
    return (w, x, y, z)


def build_mjcf(n_col: int = 24) -> str:
    """Generate MuJoCo XML with visual mesh and hollow tapered collision geoms."""
    r_bot_mid = (D_BOT_OUT - THICKNESS) * 0.5
    r_top_mid = (D_TOP_OUT - THICKNESS) * 0.5
    dr = r_top_mid - r_bot_mid
    alpha = math.atan2(dr, HEIGHT)

    # Clear bottom plane at z=0 by starting wall slightly above z=THICKNESS
    z_start = THICKNESS
    z_end = HEIGHT
    h_wall = z_end - z_start
    r_start = r_bot_mid + (dr / HEIGHT) * z_start
    r_end = r_top_mid
    r_mid = 0.5 * (r_start + r_end)
    z_mid = 0.5 * (z_start + z_end)

    slant_len = math.hypot(h_wall, r_end - r_start)
    sx = THICKNESS * 0.5
    # Chord half-width at mid height (with slight overlap to avoid gaps)
    sy = r_mid * math.tan(math.pi / n_col) * 1.02
    sz = slant_len * 0.5 - sx * math.tan(alpha)

    bot_radius = D_BOT_OUT * 0.5 - THICKNESS * 0.5
    bot_hz = THICKNESS * 0.5

    if n_col == 24:
        bot_mass = BOT_MASS
        wall_mass = WALL_MASS
    else:
        bot_mass = TOTAL_MASS * 0.2
        wall_mass = (TOTAL_MASS * 0.8) / n_col

    wall_geoms = []
    for i in range(n_col):
        theta = 2.0 * math.pi * i / n_col
        cx = r_mid * math.cos(theta)
        cy = r_mid * math.sin(theta)
        w, x, y, z = rot_z_y_to_quat(theta, alpha)
        wall_geoms.append(
            f'      <geom name="cup_wall_{i:02d}" class="cup_wall" '
            f'pos="{cx:.5f} {cy:.5f} {z_mid:.5f}" '
            f'quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}" />'
        )

    wall_block = "\n".join(wall_geoms)
    return f"""<mujoco model="paper_cup">
  <compiler angle="radian"/>

  <visual>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6" specular="0.3 0.3 0.3"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <mesh name="paper_cup" file="paper_cup.obj"/>
    <material name="paper_cup_mat" rgba="0.96 0.95 0.92 1.0" specular="0.15" shininess="0.2"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.85 0.87 0.90" rgb2="0.72 0.75 0.78"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8" reflectance="0.08"/>
  </asset>

  <default>
    <default class="collision">
      <geom group="3" friction="0.8 0.02 0.002" solref="0.005 1" solimp="0.95 0.99 0.001" rgba="0.2 0.6 0.9 0.45"/>
      <default class="cup_wall">
        <geom type="box" size="{sx:.5f} {sy:.5f} {sz:.5f}" mass="{wall_mass:.6f}"/>
      </default>
      <default class="cup_bottom">
        <geom type="cylinder" size="{bot_radius:.5f} {bot_hz:.5f}" mass="{bot_mass:.6f}"/>
      </default>
    </default>
  </default>

  <worldbody>
    <light pos="0.3 -0.4 0.6" dir="-0.5 0.6 -1.0" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light pos="-0.3 0.3 0.5" dir="0.4 -0.4 -1.0" diffuse="0.4 0.4 0.4" castshadow="false"/>
    <geom name="floor" type="plane" size="1.0 1.0 0.01" material="floor_mat"/>

    <body name="paper_cup" pos="0 0 0">
      <freejoint name="paper_cup_joint"/>
      <!-- Visual mesh (group 1, visible by default) -->
      <geom name="paper_cup_visual" type="mesh" mesh="paper_cup" material="paper_cup_mat" group="1" contype="0" conaffinity="0" mass="0"/>
      <!-- Hollow collision primitives (group 3, press '4' in viewer to toggle) -->
      <geom name="cup_bottom" class="cup_bottom" pos="0 0 {bot_hz:.5f}"/>
{wall_block}
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    vertices, faces = build_paper_cup_mesh(n_seg=64)

    obj_path = out_dir / "paper_cup.obj"
    stl_path = out_dir / "paper_cup.stl"
    xml_path = out_dir / "paper_cup.xml"

    write_obj(obj_path, vertices, faces)
    write_binary_stl(stl_path, vertices, faces)
    xml_path.write_text(build_mjcf(n_col=24), encoding="utf-8")

    print(f"Generated: {obj_path} ({len(vertices)} vertices, {len(faces)} faces)")
    print(f"Generated: {stl_path}")
    print(f"Generated: {xml_path}")


if __name__ == "__main__":
    main()
