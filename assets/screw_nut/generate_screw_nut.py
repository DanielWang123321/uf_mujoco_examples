#!/usr/bin/env python3
"""Generate 1:1 3D meshes (OBJ/STL) and MuJoCo MJCF models for the 0.0.3 screw and nut.

Specifications (from prd.md 0.0.3 + Orbbec Gemini 305 multi-view RGB-D scan):
1. Screw (螺丝):
   - Total mass: 56.5 g (0.056500 kg)
   - Total length: 45.0 mm (0.045 m)
   - Head length: 14.0 mm (0.014 m), Head outer diameter: 20.0 mm (0.020 m)
   - Thread shaft length: 31.0 mm (0.031 m), Thread outer diameter: 12.0 mm (0.012 m)
   - 12 helical thread ridges (pitch 2.40 mm) with 1.4 mm top lead-in chamfer

2. Nut (六角法兰螺母):
   - Total mass: 29.0 g (0.029000 kg)
   - Bottom flange outer diameter: 29.0 mm (0.029 m)
   - Top body outer diameter: 20.0 mm (0.020 m)
   - Total height (measured via Gemini 305 multi-view point cloud): 14.5 mm (0.0145 m)
     * Bottom flange thickness: 3.8 mm (0.0038 m)
     * Upper hexagonal/chamfered body height: 10.7 mm (0.0107 m)
   - Central threaded bore: nominal 12.0 mm (0.012 m) with coaxial bottom pilot skirt and internal helical thread teeth
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np

# --- Screw physical constants (units: meters, kilograms) ---
SCREW_TOTAL_LEN = 0.045
SCREW_HEAD_LEN = 0.014
SCREW_SHAFT_LEN = 0.031  # 0.045 - 0.014
SCREW_HEAD_DIAM = 0.020
SCREW_THREAD_DIAM = 0.012
SCREW_THREAD_TURNS = 12
SCREW_PITCH = 0.0024  # 12 helical ridges x 2.4 mm = 28.8 mm active thread on 31 mm shaft
SCREW_THREAD_Z_START = 0.0148
SCREW_THREAD_SEG_PER_TURN = 16
SCREW_THREAD_HELIX_R = 0.00540
SCREW_THREAD_CAP_R = 0.00060  # 5.40 + 0.60 = 6.00 mm crest radius (12.0 mm outer diameter)
SCREW_CORE_R = 0.00490
SCREW_THREAD_DEPTH = 0.00095
SCREW_MASS = 0.056500  # 56.5 g exact
SCREW_HEAD_MASS = 0.033850
SCREW_SHAFT_MASS = 0.021350
SCREW_TIP_MASS = 0.001300  # 0.033850 + 0.021350 + 0.001300 == 0.056500 kg

# --- Nut physical constants (units: meters, kilograms) ---
NUT_BOT_DIAM = 0.029
NUT_TOP_DIAM = 0.020
NUT_BORE_DIAM = 0.012
NUT_HEIGHT = 0.0145
NUT_FLANGE_HEIGHT = 0.0038
NUT_BODY_HEIGHT = 0.0107  # 0.0145 - 0.0038
NUT_MASS = 0.029000  # 29.0 g exact
NUT_N_COL = 20
NUT_FLANGE_BOX_MASS = 0.000780  # 20 * 0.000780 = 0.015600 kg
NUT_BODY_BOX_MASS = 0.000670  # 20 * 0.000670 = 0.013400 kg -> sum = 0.029000 kg
NUT_N_TEETH = 8
NUT_TOOTH_HELIX_R = 0.00585
NUT_TOOTH_R = 0.00045  # inner tip at 5.85 - 0.45 = 5.40 mm (engages 2.4 mm screw groove)
NUT_TOOTH_Z_START = 0.0072


def build_screw_mesh(n_seg: int = 96, n_shaft_z: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Build a watertight triangle mesh (vertices, faces) for the M12x31 screw (total 45mm)."""
    r_head = SCREW_HEAD_DIAM * 0.5
    r_shaft = SCREW_THREAD_DIAM * 0.5
    head_chamfer = 0.0006
    tip_chamfer = 0.0015

    angles = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    rings: list[np.ndarray] = []

    # Head rings (z = 0 to z = 0.014 m), ensuring exact max radius = r_head
    head_profile = [
        (r_head - head_chamfer, 0.0),
        (r_head, head_chamfer),
        (r_head, SCREW_HEAD_LEN - head_chamfer),
        (r_head - head_chamfer, SCREW_HEAD_LEN),
    ]
    for r, z in head_profile:
        rings.append(np.column_stack([r * cos_a, r * sin_a, np.full_like(angles, z)]))

    # Shaft rings with helical M12 thread modulation (z = 0.014 to z = 0.045 m)
    z_vals = np.linspace(SCREW_HEAD_LEN, SCREW_TOTAL_LEN, n_shaft_z)
    for idx, z in enumerate(z_vals):
        if idx == 0:
            r_ring = np.full_like(angles, r_shaft)
        elif idx == len(z_vals) - 1:
            r_ring = np.full_like(angles, r_shaft - tip_chamfer)
        else:
            # Envelope fades thread depth smoothly at the root and top chamfer
            z_rel = max(0.0, z - SCREW_THREAD_Z_START)
            env_bot = min(1.0, (z - SCREW_HEAD_LEN) / 0.0012)
            env_top = min(1.0, (SCREW_TOTAL_LEN - tip_chamfer - z) / 0.0012)
            env = max(0.0, min(env_bot, env_top))
            phase = angles - 2.0 * math.pi * (z_rel / SCREW_PITCH)
            # Thread crest sits exactly at r_shaft, root at r_shaft - SCREW_THREAD_DEPTH
            thread_offset = 0.5 * SCREW_THREAD_DEPTH * (1.0 - np.cos(phase)) * env
            base_r = r_shaft
            if z > SCREW_TOTAL_LEN - tip_chamfer:
                t_c = (z - (SCREW_TOTAL_LEN - tip_chamfer)) / tip_chamfer
                base_r = r_shaft - t_c * tip_chamfer
            r_ring = base_r - thread_offset
        rings.append(np.column_stack([r_ring * cos_a, r_ring * sin_a, np.full_like(angles, z)]))

    n_rings = len(rings)
    # Append bottom center (0, 0, 0) and top center (0, 0, SCREW_TOTAL_LEN)
    c_bot_idx = n_rings * n_seg
    c_top_idx = n_rings * n_seg + 1
    vertices = np.vstack(
        rings + [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, SCREW_TOTAL_LEN]], dtype=np.float64)]
    )

    faces: list[list[int]] = []
    # Bottom cap (-Z normal)
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([j, i, c_bot_idx])

    # Side quads between consecutive rings (outward normal)
    for r_idx in range(n_rings - 1):
        base0 = r_idx * n_seg
        base1 = (r_idx + 1) * n_seg
        for i in range(n_seg):
            j = (i + 1) % n_seg
            v00, v01 = base0 + i, base0 + j
            v10, v11 = base1 + i, base1 + j
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    # Top cap (+Z normal)
    top_base = (n_rings - 1) * n_seg
    for i in range(n_seg):
        j = (i + 1) % n_seg
        faces.append([top_base + i, top_base + j, c_top_idx])

    return vertices, np.asarray(faces, dtype=np.int32)


def _hex_outer_radius(angles: np.ndarray, r_circum: float, blend_circle: float = 0.0) -> np.ndarray:
    """Compute radial profile of a regular hexagon with circumradius r_circum."""
    # Sector angle within [-pi/6, +pi/6]
    sector = np.mod(angles + math.pi / 6.0, math.pi / 3.0) - math.pi / 6.0
    r_flat = r_circum * math.cos(math.pi / 6.0)
    r_hex = r_flat / np.cos(sector)
    return (1.0 - blend_circle) * r_hex + blend_circle * r_circum


def build_nut_mesh(n_seg: int = 96, n_bore_z: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Build a watertight hollow triangle mesh for the hex flange nut (29mm base, 20mm top, 14.5mm high)."""
    r_bot = NUT_BOT_DIAM * 0.5  # 14.5 mm
    r_top = NUT_TOP_DIAM * 0.5  # 10.0 mm
    r_bore = NUT_BORE_DIAM * 0.5  # 6.0 mm
    bore_chamfer = 0.0010

    # Ensure angles include exact multiples of pi/6 so hex corners & flats are exact
    angles = np.linspace(0.0, 2.0 * math.pi, n_seg, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    outer_rings: list[np.ndarray] = []
    # Outer profile from z = 0 to z = NUT_HEIGHT (14.5 mm):
    # 1. Circular bottom flange (z = 0 .. 2.8 mm, diameter 29.0 mm)
    # 2. Tapered flange shoulder (z = 2.8 .. 4.2 mm, transitioning to hex upper body)
    # 3. Hexagonal upper body (z = 4.2 .. 13.6 mm, circumdiameter 20.0 mm)
    # 4. Chamfered circular top washer face (z = 13.6 .. 14.5 mm)
    outer_specs = [
        (0.0000, np.full_like(angles, r_bot - 0.0004)),
        (0.0004, np.full_like(angles, r_bot)),
        (0.0028, np.full_like(angles, r_bot)),
        (0.0038, _hex_outer_radius(angles, r_top, blend_circle=0.35)),
        (0.0045, _hex_outer_radius(angles, r_top, blend_circle=0.0)),
        (0.0075, _hex_outer_radius(angles, r_top, blend_circle=0.0)),
        (0.0105, _hex_outer_radius(angles, r_top, blend_circle=0.0)),
        (0.0136, _hex_outer_radius(angles, r_top, blend_circle=0.0)),
        (0.0145, np.full_like(angles, r_top * math.cos(math.pi / 6.0))),
    ]
    for z, r_arr in outer_specs:
        outer_rings.append(np.column_stack([r_arr * cos_a, r_arr * sin_a, np.full_like(angles, z)]))

    # Inner bore profile from z = NUT_HEIGHT down to z = 0 (with internal helical thread grooves)
    inner_rings: list[np.ndarray] = []
    z_bore = np.linspace(NUT_HEIGHT, 0.0, n_bore_z)
    for idx, z in enumerate(z_bore):
        if idx == 0 or idx == len(z_bore) - 1:
            r_in = np.full_like(angles, r_bore + bore_chamfer)
        else:
            env_bot = min(1.0, z / 0.0015)
            env_top = min(1.0, (NUT_HEIGHT - z) / 0.0015)
            env = max(0.0, min(env_bot, env_top))
            phase = angles - 2.0 * math.pi * (z / SCREW_PITCH)
            thread_groove = 0.5 * (SCREW_THREAD_DEPTH * 0.8) * (1.0 - np.cos(phase)) * env
            base_r = r_bore
            if z < bore_chamfer:
                base_r = r_bore + (bore_chamfer - z)
            elif z > NUT_HEIGHT - bore_chamfer:
                base_r = r_bore + (z - (NUT_HEIGHT - bore_chamfer))
            r_in = base_r + thread_groove
        inner_rings.append(np.column_stack([r_in * cos_a, r_in * sin_a, np.full_like(angles, z)]))

    all_rings = outer_rings + inner_rings
    n_total_rings = len(all_rings)
    vertices = np.vstack(all_rings)

    faces: list[list[int]] = []
    for r_idx in range(n_total_rings):
        next_r_idx = (r_idx + 1) % n_total_rings
        base0 = r_idx * n_seg
        base1 = next_r_idx * n_seg
        for i in range(n_seg):
            j = (i + 1) % n_seg
            v00, v01 = base0 + i, base0 + j
            v10, v11 = base1 + i, base1 + j
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    return vertices, np.asarray(faces, dtype=np.int32)


def write_obj(
    path: Path,
    obj_name: str,
    header_comment: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    crease_deg: float = 26.0,
) -> None:
    """Write triangle mesh to Wavefront OBJ with crease-split normals (vn) and reflection UVs (vt)."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(fn_norm, 1e-12)

    cos_thresh = math.cos(math.radians(crease_deg))
    v_faces: list[list[int]] = [[] for _ in range(len(vertices))]
    for f_idx, tri in enumerate(faces):
        v_faces[int(tri[0])].append(f_idx)
        v_faces[int(tri[1])].append(f_idx)
        v_faces[int(tri[2])].append(f_idx)

    vn_list: list[np.ndarray] = []
    vt_list: list[tuple[float, float]] = []
    f_vn_idx = np.zeros_like(faces)
    for f_idx, tri in enumerate(faces):
        n_this = fn[f_idx]
        for c in range(3):
            v_id = int(tri[c])
            n_sum = np.zeros(3, dtype=np.float64)
            for nb_f in v_faces[v_id]:
                if float(np.dot(n_this, fn[nb_f])) >= cos_thresh:
                    n_sum += fn[nb_f]
            norm_val = float(np.linalg.norm(n_sum))
            n_final = n_sum / norm_val if norm_val > 1e-12 else n_this
            vn_list.append(n_final)
            pt = vertices[v_id]
            u_val = 0.5 + 0.45 * n_final[0] - 0.35 * n_final[1] + 12.0 * pt[0]
            v_val = 0.5 + 0.45 * n_final[2] + 0.25 * n_final[1] + 18.0 * pt[2]
            vt_list.append((float(u_val), float(v_val)))
            f_vn_idx[f_idx, c] = len(vn_list)

    lines = [
        f"# {header_comment}",
        f"o {obj_name}",
    ]
    for v in vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for u_val, v_val in vt_list:
        lines.append(f"vt {u_val:.5f} {v_val:.5f}")
    for vn in vn_list:
        lines.append(f"vn {vn[0]:.5f} {vn[1]:.5f} {vn[2]:.5f}")
    for f_idx, tri in enumerate(faces):
        i0, i1, i2 = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
        n0, n1, n2 = int(f_vn_idx[f_idx, 0]), int(f_vn_idx[f_idx, 1]), int(f_vn_idx[f_idx, 2])
        lines.append(f"f {i0}/{n0}/{n0} {i1}/{n1}/{n1} {i2}/{n2}/{n2}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chrome_texture(path: Path, width: int = 512, height: int = 512) -> None:
    """Write a seamless 24-bit RGB PNG texture simulating polished stainless-steel / chrome reflections."""
    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float64)
    u = x_grid / width * (2.0 * math.pi)
    v = y_grid / height * (2.0 * math.pi)
    env_band = (
        0.82
        + 0.14 * np.cos(u - 0.6 * v)
        + 0.07 * np.cos(2.0 * u + 1.1 * v + 0.4)
        + 0.04 * np.sin(3.0 * u - 1.5 * v)
    )
    env_band = np.clip(env_band, 0.58, 1.00)
    r_ch = np.clip(env_band * 246.0, 0, 255).astype(np.uint8)
    g_ch = np.clip(env_band * 250.0, 0, 255).astype(np.uint8)
    b_ch = np.clip(env_band * 255.0, 0, 255).astype(np.uint8)
    rgb = np.dstack([r_ch, g_ch, b_ch])

    raw_rows = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw_rows, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png_bytes)


def write_binary_stl(path: Path, header_text: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write triangle mesh to binary STL file."""
    header = header_text.encode("ascii", errors="ignore")[:80].ljust(80, b"\0")
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


def _z_quat(theta: float) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) for rotation around +Z by theta radians."""
    return (math.cos(0.5 * theta), 0.0, 0.0, math.sin(0.5 * theta))


def _build_screw_collision_geoms(col_class: str = "screw_col") -> str:
    """Generate 3D collision geoms for the screw: head, core shaft, tip, and 12 helical thread turns."""
    lines = [
        f'      <geom name="screw_head" class="{col_class}" type="cylinder" pos="0 0 0.00700" size="0.01000 0.00700" mass="{SCREW_HEAD_MASS:.6f}"/>',
        f'      <geom name="screw_shaft" class="{col_class}" type="cylinder" pos="0 0 0.02850" size="{SCREW_CORE_R:.5f} 0.01450" mass="{SCREW_SHAFT_MASS:.6f}"/>',
        f'      <geom name="screw_tip" class="{col_class}" type="cylinder" pos="0 0 0.04420" size="0.00470 0.00080" mass="{SCREW_TIP_MASS:.6f}"/>',
    ]
    n_total = SCREW_THREAD_TURNS * SCREW_THREAD_SEG_PER_TURN
    for i in range(n_total):
        t0 = 2.0 * math.pi * (i / SCREW_THREAD_SEG_PER_TURN)
        t1 = 2.0 * math.pi * ((i + 1) / SCREW_THREAD_SEG_PER_TURN)
        x0 = SCREW_THREAD_HELIX_R * math.cos(t0)
        y0 = SCREW_THREAD_HELIX_R * math.sin(t0)
        z0 = SCREW_THREAD_Z_START + SCREW_PITCH * (i / SCREW_THREAD_SEG_PER_TURN)
        x1 = SCREW_THREAD_HELIX_R * math.cos(t1)
        y1 = SCREW_THREAD_HELIX_R * math.sin(t1)
        z1 = SCREW_THREAD_Z_START + SCREW_PITCH * ((i + 1) / SCREW_THREAD_SEG_PER_TURN)
        lines.append(
            f'      <geom name="screw_thread_{i:03d}" class="{col_class}" type="capsule" '
            f'size="{SCREW_THREAD_CAP_R:.5f}" '
            f'fromto="{x0:.5f} {y0:.5f} {z0:.5f} {x1:.5f} {y1:.5f} {z1:.5f}" mass="0"/>'
        )
    return "\n".join(lines)


def build_screw_mjcf() -> str:
    """Generate standalone MuJoCo XML for the 56.5g screw with 12-turn 3D helical thread collision."""
    screw_geom_block = _build_screw_collision_geoms()
    return f"""<mujoco model="screw">
  <compiler angle="radian"/>

  <visual>
    <headlight ambient="0.26 0.27 0.29" diffuse="0.42 0.44 0.46" specular="0.95 0.97 1.00"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <mesh name="screw" file="screw.obj"/>
    <texture name="chrome_tex" type="2d" file="chrome_steel.png"/>
    <material name="screw_mat" texture="chrome_tex" rgba="0.80 0.83 0.87 1.0" specular="1.0" shininess="0.82" reflectance="0.35"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.85 0.87 0.90" rgb2="0.72 0.75 0.78"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8" specular="0.08" shininess="0.15" reflectance="0.06"/>
  </asset>

  <default>
    <default class="screw_col">
      <geom group="3" friction="0.35 0.005 0.0005" solref="0.004 1" solimp="0.95 0.99 0.001" rgba="0.2 0.6 0.9 0.45"/>
    </default>
  </default>

  <worldbody>
    <light pos="0.3 -0.4 0.6" dir="-0.5 0.6 -1.0" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light pos="-0.3 0.3 0.5" dir="0.4 -0.4 -1.0" diffuse="0.4 0.4 0.4" castshadow="false"/>
    <geom name="floor" type="plane" size="1.0 1.0 0.01" material="floor_mat"/>

    <body name="screw" pos="0 0 0">
      <freejoint name="screw_joint"/>
      <!-- Visual mesh (group 1) -->
      <geom name="screw_visual" type="mesh" mesh="screw" material="screw_mat" group="1" contype="0" conaffinity="0" mass="0"/>
      <!-- Collision primitives (group 3): total mass = 0.033850 + 0.021350 + 0.001300 = 0.056500 kg -->
{screw_geom_block}
    </body>
  </worldbody>
</mujoco>
"""


def _build_nut_collision_geoms(
    n_col: int = NUT_N_COL,
    flange_class: str = "nut_flange",
    body_class: str = "nut_body",
    col_class: str = "nut_col",
) -> tuple[str, dict[str, tuple[float, float, float]]]:
    """Generate hollow collision ring geoms and internal helical thread teeth for the flange nut."""
    # Bottom flange ring: z in [0.000, 0.0038], r_in = 0.0070 (lead-in funnel), r_out = 0.0145
    r_in_flange = 0.0070
    r_out_flange = NUT_BOT_DIAM * 0.5  # 0.0145
    r_mid_flange = 0.5 * (r_in_flange + r_out_flange)
    sx_flange = 0.5 * (r_out_flange - r_in_flange)
    sy_flange = r_mid_flange * math.tan(math.pi / n_col) * 0.98
    sz_flange = NUT_FLANGE_HEIGHT * 0.5
    z_mid_flange = sz_flange

    # Upper body ring: z in [0.0038, 0.0145], r_in = 0.00635 (coaxial pilot bore), r_out = 0.0100
    r_in_body = 0.00635
    r_out_body = NUT_TOP_DIAM * 0.5  # 0.0100
    r_mid_body = 0.5 * (r_in_body + r_out_body)
    sx_body = 0.5 * (r_out_body - r_in_body)
    sy_body = r_mid_body * math.tan(math.pi / n_col) * 0.98
    sz_body = NUT_BODY_HEIGHT * 0.5
    z_mid_body = NUT_FLANGE_HEIGHT + sz_body

    lines: list[str] = []
    for i in range(n_col):
        theta = 2.0 * math.pi * i / n_col
        w, x, y, z = _z_quat(theta)
        fx = r_mid_flange * math.cos(theta)
        fy = r_mid_flange * math.sin(theta)
        lines.append(
            f'      <geom name="nut_flange_{i:02d}" class="{flange_class}" '
            f'pos="{fx:.5f} {fy:.5f} {z_mid_flange:.5f}" '
            f'quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}" />'
        )
    for i in range(n_col):
        theta = 2.0 * math.pi * i / n_col
        w, x, y, z = _z_quat(theta)
        bx = r_mid_body * math.cos(theta)
        by = r_mid_body * math.sin(theta)
        lines.append(
            f'      <geom name="nut_body_{i:02d}" class="{body_class}" '
            f'pos="{bx:.5f} {by:.5f} {z_mid_body:.5f}" '
            f'quat="{w:.6f} {x:.6f} {y:.6f} {z:.6f}" />'
        )
    for i in range(NUT_N_TEETH):
        t = 2.0 * math.pi * (i / NUT_N_TEETH)
        tx = NUT_TOOTH_HELIX_R * math.cos(t)
        ty = NUT_TOOTH_HELIX_R * math.sin(t)
        tz = NUT_TOOTH_Z_START + SCREW_PITCH * (i / NUT_N_TEETH)
        lines.append(
            f'      <geom name="nut_tooth_{i:02d}" class="{col_class}" type="sphere" '
            f'pos="{tx:.5f} {ty:.5f} {tz:.5f}" size="{NUT_TOOTH_R:.5f}" mass="0" />'
        )

    sizes = {
        "flange": (sx_flange, sy_flange, sz_flange),
        "body": (sx_body, sy_body, sz_body),
    }
    return "\n".join(lines), sizes


def build_nut_mjcf() -> str:
    """Generate standalone MuJoCo XML for the 29.0g hollow hex flange nut."""
    geom_block, sizes = _build_nut_collision_geoms()
    f_sx, f_sy, f_sz = sizes["flange"]
    b_sx, b_sy, b_sz = sizes["body"]

    return f"""<mujoco model="nut">
  <compiler angle="radian"/>

  <visual>
    <headlight ambient="0.26 0.27 0.29" diffuse="0.42 0.44 0.46" specular="0.95 0.97 1.00"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <mesh name="nut" file="nut.obj"/>
    <texture name="chrome_tex" type="2d" file="chrome_steel.png"/>
    <material name="nut_mat" texture="chrome_tex" rgba="0.78 0.81 0.86 1.0" specular="1.0" shininess="0.82" reflectance="0.35"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.85 0.87 0.90" rgb2="0.72 0.75 0.78"/>
    <material name="floor_mat" texture="grid" texrepeat="8 8" specular="0.08" shininess="0.15" reflectance="0.06"/>
  </asset>

  <default>
    <default class="nut_col">
      <geom group="3" friction="0.35 0.005 0.0005" solref="0.004 1" solimp="0.95 0.99 0.001" rgba="0.2 0.8 0.5 0.45"/>
      <default class="nut_flange">
        <geom type="box" size="{f_sx:.5f} {f_sy:.5f} {f_sz:.5f}" mass="{NUT_FLANGE_BOX_MASS:.6f}"/>
      </default>
      <default class="nut_body">
        <geom type="box" size="{b_sx:.5f} {b_sy:.5f} {b_sz:.5f}" mass="{NUT_BODY_BOX_MASS:.6f}"/>
      </default>
    </default>
  </default>

  <worldbody>
    <light pos="0.3 -0.4 0.6" dir="-0.5 0.6 -1.0" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light pos="-0.3 0.3 0.5" dir="0.4 -0.4 -1.0" diffuse="0.4 0.4 0.4" castshadow="false"/>
    <geom name="floor" type="plane" size="1.0 1.0 0.01" material="floor_mat"/>

    <body name="nut" pos="0 0 0">
      <freejoint name="nut_joint"/>
      <!-- Visual mesh (group 1) -->
      <geom name="nut_visual" type="mesh" mesh="nut" material="nut_mat" group="1" contype="0" conaffinity="0" mass="0"/>
      <!-- Hollow collision rings + internal helical thread teeth (group 3): 20 * 0.000780 + 20 * 0.000670 = 0.029000 kg -->
{geom_block}
    </body>
  </worldbody>
</mujoco>
"""


def build_combined_mjcf() -> str:
    """Generate combined MuJoCo preview & assembly scene with fixed upright screw and free nut."""
    screw_geom_block = _build_screw_collision_geoms()
    geom_block, sizes = _build_nut_collision_geoms()
    f_sx, f_sy, f_sz = sizes["flange"]
    b_sx, b_sy, b_sz = sizes["body"]

    return f"""<mujoco model="screw_nut">
  <compiler angle="radian"/>
  <option timestep="0.001" integrator="implicitfast"/>

  <visual>
    <headlight ambient="0.26 0.27 0.29" diffuse="0.42 0.44 0.46" specular="0.95 0.97 1.00"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <mesh name="screw" file="screw.obj"/>
    <mesh name="nut" file="nut.obj"/>
    <texture name="chrome_tex" type="2d" file="chrome_steel.png"/>
    <material name="screw_mat" texture="chrome_tex" rgba="0.80 0.83 0.87 1.0" specular="1.0" shininess="0.82" reflectance="0.35"/>
    <material name="nut_mat" texture="chrome_tex" rgba="0.78 0.81 0.86 1.0" specular="1.0" shininess="0.82" reflectance="0.35"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.18 0.52 0.45" rgb2="0.14 0.44 0.38"/>
    <material name="floor_mat" texture="grid" texrepeat="10 10" specular="0.08" shininess="0.15" reflectance="0.06"/>
  </asset>

  <default>
    <default class="screw_col">
      <geom group="3" condim="4" friction="0.10 0.002 0.0001" solref="0.003 1" solimp="0.99 0.999 0.0001" rgba="0.2 0.6 0.9 0.45"/>
    </default>
    <default class="nut_col">
      <geom group="3" condim="4" friction="0.10 0.002 0.0001" solref="0.003 1" solimp="0.99 0.999 0.0001" rgba="0.2 0.8 0.5 0.45"/>
      <default class="nut_flange">
        <geom type="box" size="{f_sx:.5f} {f_sy:.5f} {f_sz:.5f}" mass="{NUT_FLANGE_BOX_MASS:.6f}"/>
      </default>
      <default class="nut_body">
        <geom type="box" size="{b_sx:.5f} {b_sy:.5f} {b_sz:.5f}" mass="{NUT_BODY_BOX_MASS:.6f}"/>
      </default>
    </default>
  </default>

  <worldbody>
    <light pos="0.3 -0.4 0.6" dir="-0.5 0.6 -1.0" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light pos="-0.3 0.3 0.5" dir="0.4 -0.4 -1.0" diffuse="0.4 0.4 0.4" castshadow="false"/>
    <geom name="floor" type="plane" size="1.0 1.0 0.01" material="floor_mat" friction="0.8 0.01 0.001"/>

    <!-- Upright screw fixed to the tabletop at (0, -0.045, 0) -->
    <body name="screw" pos="0 -0.045 0">
      <geom name="screw_visual" type="mesh" mesh="screw" material="screw_mat" group="1" contype="0" conaffinity="0" mass="0"/>
{screw_geom_block}
    </body>

    <!-- Free hollow hex flange nut resting on the tabletop at (0, +0.045, 0) -->
    <body name="nut" pos="0 0.045 0">
      <freejoint name="nut_joint"/>
      <geom name="nut_visual" type="mesh" mesh="nut" material="nut_mat" group="1" contype="0" conaffinity="0" mass="0"/>
{geom_block}
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    write_chrome_texture(out_dir / "chrome_steel.png")

    screw_v, screw_f = build_screw_mesh()
    write_obj(
        out_dir / "screw.obj",
        "screw",
        "Screw 3D mesh (units: m) - L=45.0mm, Head=20x14mm, Thread=M12x31mm, Mass=56.5g",
        screw_v,
        screw_f,
        crease_deg=28.0,
    )
    write_binary_stl(out_dir / "screw.stl", "Screw mesh (units: m)", screw_v, screw_f)
    (out_dir / "screw.xml").write_text(build_screw_mjcf(), encoding="utf-8")

    nut_v, nut_f = build_nut_mesh()
    write_obj(
        out_dir / "nut.obj",
        "nut",
        "Hex flange nut 3D mesh (units: m) - BotOD=29.0mm, TopOD=20.0mm, Bore=12.0mm, H=14.5mm, Mass=29.0g",
        nut_v,
        nut_f,
        crease_deg=25.0,
    )
    write_binary_stl(out_dir / "nut.stl", "Hex flange nut mesh (units: m)", nut_v, nut_f)
    (out_dir / "nut.xml").write_text(build_nut_mjcf(), encoding="utf-8")
    (out_dir / "screw_nut.xml").write_text(build_combined_mjcf(), encoding="utf-8")

    print(f"Generated screw: {len(screw_v)} vertices, {len(screw_f)} faces -> {out_dir / 'screw.obj'}")
    print(f"Generated nut:   {len(nut_v)} vertices, {len(nut_f)} faces -> {out_dir / 'nut.obj'}")
    print(f"Generated MJCF:  screw.xml, nut.xml, screw_nut.xml in {out_dir}")


if __name__ == "__main__":
    main()
