"""Assemble a DIVA object (meshes, materials, skin, physics) from a VRM model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..formats.objset import BoneInfo, Mesh, Object, Skin, SubMesh, to_diva_matrix
from ..formats.txp import Texture
from ..vrm.model import VrmModel
from .face import bake_face, expression_deltas
from .materials import build_textures, make_material, submesh_flags
from .physics import OsageChain, build_osage, skin_param_text
from .reference import Reference
from .retarget import Retarget

MAX_BONES_PER_SUBMESH = 250   # the game's skinning buffer holds 256 matrices
MAX_VERTICES_PER_MESH = 65000  # 16-bit indices


@dataclass
class BuildOptions:
    object_name: str = "VROID"
    texture_prefix: str = "VROID"
    style: str = "toon"            # "toon" (SKIN/CLOTH shaders) or "simple" (BLINN)
    physics: bool = True
    face: bool = True
    compress_textures: bool = True
    max_texture_size: int = 2048


@dataclass
class BuildResult:
    obj: Object
    textures: List[Texture]
    skin_params: Dict[str, object]
    chains: List[OsageChain] = field(default_factory=list)
    face_fit: List[dict] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


def _top4(idx: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Merge duplicate bones per vertex, keep the 4 strongest, renormalise."""
    idx = idx.copy()
    w = np.where(idx >= 0, w, 0.0).astype(np.float64)
    k = idx.shape[1]
    for a in range(k):
        for b in range(a + 1, k):
            same = (idx[:, a] == idx[:, b]) & (idx[:, a] >= 0)
            w[same, a] += w[same, b]
            w[same, b] = 0.0
            idx[same, b] = -1
    w = np.where(idx >= 0, w, 0.0)
    order = np.argsort(-w, axis=1, kind="stable")[:, :4]
    out_w = np.take_along_axis(w, order, axis=1)
    out_i = np.take_along_axis(idx, order, axis=1)
    if out_w.shape[1] < 4:
        pad = 4 - out_w.shape[1]
        out_w = np.pad(out_w, ((0, 0), (0, pad)))
        out_i = np.pad(out_i, ((0, 0), (0, pad)), constant_values=-1)
    out_i = np.where(out_w > 0, out_i, -1)
    total = out_w.sum(axis=1, keepdims=True)
    out_w = np.where(total > 0, out_w / np.maximum(total, 1e-12), 0.0)
    return out_i.astype(np.int64), out_w


def build_object(model: VrmModel, ref: Reference, opts: BuildOptions, log=print) -> BuildResult:
    rt = Retarget(model, ref, log)
    log("Skeleton: %d VRM bones mapped to DIVA bones (scale %.3f)." % (len(rt.mapping), rt.scale))

    # VRM node -> DIVA bone for every humanoid node.
    targets: Dict[int, str] = {model.humanoid[b]: d for b, d in rt.mapping.items()}
    chains: List[OsageChain] = build_osage(model, rt, ref, targets, log) if opts.physics else []

    # -- gather vertices -----------------------------------------------------
    names: List[str] = []
    name_index: Dict[str, int] = {}

    def nid(name: str) -> int:
        if name not in name_index:
            name_index[name] = len(names)
            names.append(name)
        return name_index[name]

    node_cache: Dict[int, int] = {}

    def node_bone(node: int) -> int:
        if node not in node_cache:
            node_cache[node] = nid(rt.ancestor_target(node, targets))
        return node_cache[node]

    prims = [p for p in model.primitives if len(p.indices)]
    pos_list, nrm_list, uv_list, idx_list, w_list, morph_list, ranges = [], [], [], [], [], [], []
    start = 0
    kmax = max((p.joints.shape[1] for p in prims), default=1)
    for p in prims:
        pos, nrm, morphs = rt.deform(p.joints, p.weights, p.positions, p.normals, p.morphs)
        k = p.joints.shape[1]
        bones = np.full((len(pos), kmax), -1, dtype=np.int64)
        wts = np.zeros((len(pos), kmax))
        uniq = {int(j) for j in np.unique(p.joints) if j >= 0}
        lut = {j: node_bone(j) for j in uniq}
        for c in range(k):
            col = p.joints[:, c]
            bones[:, c] = [lut[int(j)] if j >= 0 else -1 for j in col]
            wts[:, c] = np.where(col >= 0, p.weights[:, c], 0)
        pos_list.append(pos)
        nrm_list.append(nrm)
        uv_list.append(p.uv0 if p.uv0 is not None else np.zeros((len(pos), 2)))
        idx_list.append(bones)
        w_list.append(wts)
        morph_list.append(morphs)
        ranges.append((p.mesh_index, start, start + len(pos)))
        start += len(pos)

    positions = np.concatenate(pos_list)
    normals = np.concatenate(nrm_list)
    uvs = np.concatenate(uv_list)
    inf_idx = np.concatenate(idx_list)
    inf_w = np.concatenate(w_list)

    if opts.face:
        deltas = expression_deltas(model, ranges, morph_list, len(positions))
        inf_idx, inf_w, names = bake_face(ref, rt, positions, inf_idx, inf_w, names, deltas, log)
        name_index = {n: i for i, n in enumerate(names)}

    # -- resolve helper bones (twist bones etc.) ------------------------------
    osage_names = {n for c in chains for n in c.node_names}
    used = {names[i] for i in np.unique(inf_idx) if i >= 0}
    ex_used = [n for n in used if n not in osage_names and ref.bones[n].is_ex]
    blocks, missing = ref.resolve_blocks(ex_used)
    if missing and ref.motion_bone_names:
        log("Note: helper bones %s depend on data that is not available; using their parents."
            % ", ".join(sorted(missing)))
        remap = {}
        for n in ex_used:
            _, miss = ref.resolve_blocks([n])
            if miss:
                remap[name_index[n]] = nid(ref.motion_parent(n))
                name_index = {x: i for i, x in enumerate(names)}
        for old, new in remap.items():
            inf_idx = np.where(inf_idx == old, new, inf_idx)
        used = {names[i] for i in np.unique(inf_idx) if i >= 0}
        blocks, _ = ref.resolve_blocks([n for n in used if n not in osage_names and ref.bones[n].is_ex])

    inf_idx, inf_w = _top4(inf_idx, inf_w)

    # -- skin -----------------------------------------------------------------
    skin = Skin()
    skin_index: Dict[str, int] = {}
    used_names = [names[i] for i in sorted({int(i) for i in np.unique(inf_idx) if i >= 0})]
    chain_frames = {n: f for c in chains for n, f in zip(c.node_names, c.frames)}
    chain_parent = {}
    for c in chains:
        prev = c.parent_bone
        for n in c.node_names:
            chain_parent[n] = prev
            prev = n
    # Every osage joint must be in the skin even if no vertex uses it.
    for n in list(osage_names):
        if n not in used_names:
            used_names.append(n)
    for c in chains:
        if c.parent_bone not in used_names:
            used_names.append(c.parent_bone)
    for name in used_names:
        if name in chain_frames:
            frame = chain_frames[name]
            info = BoneInfo(name, is_ex=True, parent=chain_parent.get(name))
        else:
            rb = ref.bones[name]
            frame = rt.bind_frame(name, rt.bound_position(name))
            info = BoneInfo(name, id=rb.id, is_ex=rb.is_ex, parent=rb.parent)
        info.ibm = to_diva_matrix(np.linalg.inv(frame))
        skin_index[name] = len(skin.bones)
        skin.bones.append(info)
    skin.blocks = list(blocks) + [c.block for c in chains]

    # global name index -> skin bone index
    to_skin = np.full(len(names) + 1, -1, dtype=np.int64)
    for n, i in skin_index.items():
        if n in name_index:
            to_skin[name_index[n]] = i
    sk_idx = np.where(inf_idx >= 0, to_skin[np.maximum(inf_idx, 0)], -1)

    # -- materials & textures ---------------------------------------------------
    used_mats = sorted({p.material for p in prims})
    textures, tex_ids = build_textures(model, used_mats, opts.texture_prefix, opts.compress_textures,
                                       opts.max_texture_size, log)
    obj = Object(name=opts.object_name)
    mat_index: Dict[int, int] = {}
    for mi in used_mats:
        vm = model.materials[mi] if 0 <= mi < len(model.materials) else None
        if vm is None:
            from ..vrm.model import VrmMaterial
            vm = VrmMaterial(name="default")
        tid = tex_ids.get(vm.texture) if vm.texture is not None else None
        mat_index[mi] = len(obj.materials)
        obj.materials.append(make_material(vm, len(obj.materials), tid, ref, opts.style))

    # -- meshes (chunked by bone palette / vertex limits) --------------------------
    total_tris = 0
    for (mesh_index, s, e), p in zip(ranges, prims):
        vm = model.materials[p.material] if 0 <= p.material < len(model.materials) else None
        tris = p.indices.reshape(-1, 3).astype(np.int64) + s
        total_tris += len(tris)
        for chunk in _chunk_triangles(tris, sk_idx):
            verts, local_tris = np.unique(chunk, return_inverse=True)
            local_tris = local_tris.reshape(-1, 3)
            palette = sorted({int(b) for b in sk_idx[verts].ravel() if b >= 0})
            pal_index = {b: i for i, b in enumerate(palette)}
            bi = np.vectorize(lambda b: pal_index.get(int(b), -1))(sk_idx[verts]) if len(verts) else sk_idx[verts]
            bw = inf_w[verts].astype(np.float32)
            bw = np.where(bi >= 0, bw, 0)
            mesh = Mesh(name=("%s_%d" % (obj.materials[mat_index[p.material]].name, len(obj.meshes)))[:63],
                        positions=positions[verts].astype(np.float32),
                        normals=normals[verts].astype(np.float32),
                        uv0=uvs[verts].astype(np.float32),
                        blend_weights=bw, blend_indices=bi.astype(np.int32))
            mesh.submeshes.append(SubMesh(material_index=mat_index[p.material],
                                          indices=local_tris.reshape(-1).astype(np.uint32),
                                          bone_indices=palette, flags=submesh_flags(vm)))
            obj.meshes.append(mesh)
    obj.skin = skin
    skp = skin_param_text(chains) if chains else {}
    stats = {"vertices": int(sum(len(m.positions) for m in obj.meshes)), "triangles": int(total_tris),
             "meshes": len(obj.meshes), "bones": len(skin.bones), "osage_chains": len(chains),
             "textures": len(textures)}
    log("Model: %(vertices)d vertices, %(triangles)d triangles, %(meshes)d meshes, %(bones)d bones." % stats)
    return BuildResult(obj, textures, skp, chains, rt.face_fit, stats)


def _chunk_triangles(tris: np.ndarray, sk_idx: np.ndarray):
    """Split a triangle list so each chunk stays within the bone/vertex limits."""
    tri_bones = [set(int(b) for b in sk_idx[t].ravel() if b >= 0) for t in tris]
    all_bones = set().union(*tri_bones) if tri_bones else set()
    if len(all_bones) <= MAX_BONES_PER_SUBMESH and len(np.unique(tris)) <= MAX_VERTICES_PER_MESH:
        yield tris
        return
    cur, cur_bones, cur_verts = [], set(), set()
    for t, tb in zip(tris, tri_bones):
        nb = cur_bones | tb
        nv = cur_verts | set(int(x) for x in t)
        if cur and (len(nb) > MAX_BONES_PER_SUBMESH or len(nv) > MAX_VERTICES_PER_MESH):
            yield np.array(cur)
            cur, cur_bones, cur_verts = [], set(), set()
            nb, nv = set(tb), set(int(x) for x in t)
        cur.append(t)
        cur_bones, cur_verts = nb, nv
    if cur:
        yield np.array(cur)
