"""Render a converted object the way the game skins it (for previews/checks).

Each vertex is transformed with ``bone_world @ inverse_bind`` exactly like the
game's skinning, using the reference (Miku) skeleton in its rest pose, plus
optional extra rotations on individual bones.  The result is drawn with a tiny
painter's-algorithm rasterizer, so no GPU or game is needed.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .formats.objset import Object, from_diva_matrix


def game_bone_worlds(obj: Object, ref, chains=(), rotations: Optional[Dict[str, np.ndarray]] = None,
                     bind_frames: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """World matrices of the object's skin bones in the game's rest pose."""
    rotations = rotations or {}
    worlds = {}
    chain_parent = {}
    for c in chains:
        for n in c.node_names:
            chain_parent[n] = c.parent_bone
    for b in obj.skin.bones:
        if b.name in ref.bones:
            w = ref.bones[b.name].world.copy()
        elif b.name in chain_parent:
            p = chain_parent[b.name]
            bind_p = np.linalg.inv(from_diva_matrix(obj.skin.bones[_index(obj, p)].ibm))
            w = ref.bones[p].world @ np.linalg.inv(bind_p) @ np.linalg.inv(from_diva_matrix(b.ibm))
        else:
            w = np.linalg.inv(from_diva_matrix(b.ibm))
        worlds[b.name] = w
    for name, r in rotations.items():
        if name in worlds:
            m = worlds[name].copy()
            m[:3, :3] = m[:3, :3] @ r
            worlds[name] = m
    return worlds


def _index(obj, name):
    for i, b in enumerate(obj.skin.bones):
        if b.name == name:
            return i
    raise KeyError(name)


def skin_vertices(obj: Object, worlds: Dict[str, np.ndarray]):
    """Yield (positions, normals, triangles, material index) per sub mesh."""
    mats = [worlds[b.name] @ from_diva_matrix(b.ibm) for b in obj.skin.bones]
    mats = np.stack(mats)
    for mesh in obj.meshes:
        for sm in mesh.submeshes:
            pal = np.array(sm.bone_indices or [0])
            bi = np.where(mesh.blend_indices >= 0, pal[np.maximum(mesh.blend_indices, 0)], 0)
            w = np.where(mesh.blend_indices >= 0, mesh.blend_weights, 0)
            m = np.einsum("nk,nkij->nij", w, mats[bi])
            p = np.einsum("nij,nj->ni", m[:, :3, :3], mesh.positions) + m[:, :3, 3]
            n = np.einsum("nij,nj->ni", m[:, :3, :3], mesh.normals)
            yield p, n, np.asarray(sm.indices).reshape(-1, 3), sm.material_index


def render(obj: Object, worlds: Dict[str, np.ndarray], path: str, size=(600, 900), view="front",
           center=None, extent=None, title: str = "") -> None:
    from PIL import Image, ImageDraw

    tris_all, shade_all, depth_all = [], [], []
    palette = [(230, 190, 170), (120, 150, 220), (220, 120, 150), (140, 200, 140), (200, 180, 100),
               (160, 120, 200), (100, 200, 200), (200, 140, 90)]
    for p, n, tris, mi in skin_vertices(obj, worlds):
        if view == "side":
            p = p[:, [2, 1, 0]] * np.array([1, 1, -1])
            n = n[:, [2, 1, 0]] * np.array([1, 1, -1])
        v = p[tris]
        fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        light = np.clip(np.abs(fn @ np.array([0.3, 0.4, 0.86])), 0.15, 1.0)
        base = np.array(palette[mi % len(palette)])
        tris_all.append(v[:, :, :2])
        shade_all.append(light[:, None] * base)
        depth_all.append(v[:, :, 2].mean(axis=1))
    tris = np.concatenate(tris_all)
    shade = np.concatenate(shade_all)
    depth = np.concatenate(depth_all)
    order = np.argsort(depth)  # far (small z) first, viewer looks down -Z
    lo = tris.reshape(-1, 2).min(axis=0)
    hi = tris.reshape(-1, 2).max(axis=0)
    if center is None:
        center = (lo + hi) / 2
    if extent is None:
        extent = max(hi - lo) * 1.05
    w, h = size
    s = min(w, h) / extent
    img = Image.new("RGB", size, (250, 250, 250))
    d = ImageDraw.Draw(img)
    for i in order:
        pts = [(w / 2 + (x - center[0]) * s, h / 2 - (y - center[1]) * s) for x, y in tris[i]]
        c = tuple(int(x) for x in shade[i])
        d.polygon(pts, fill=c)
    if title:
        d.text((8, 8), title, fill=(0, 0, 0))
    img.save(path)


def apply_face_fit(worlds: Dict[str, np.ndarray], face_fit, expression: str, amount: float = 1.0):
    """Move face bones the way the fitted VRoid expression moves them."""
    from .convert.retarget import axis_angle
    out = dict(worlds)
    for fit in face_fit:
        if fit["driver"] != expression or fit["bone"] not in out:
            continue
        m = out[fit["bone"]].copy()
        if fit["kind"] == "rot":
            r = axis_angle(np.array(fit["axis"]), fit["amount"] * amount)
            p = m[:3, 3].copy()
            m[:3, :3] = r @ m[:3, :3]
            m[:3, 3] = p
        else:
            m[:3, 3] += np.array(fit["axis"]) * fit["amount"] * amount
        out[fit["bone"]] = m
    return out
