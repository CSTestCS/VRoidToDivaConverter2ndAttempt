"""Load a VRM (0.x or 1.0) file into a normalized, engine-neutral model.

Everything is evaluated in the file's rest pose and in world space so later
stages never need to care which VRM version the file was.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .gltf import Gltf, node_local_matrix

# VRM 0.x preset names -> VRM 1.0 names.
_EXPRESSION_ALIASES = {
    "a": "aa", "i": "ih", "u": "ou", "e": "ee", "o": "oh",
    "joy": "happy", "sorrow": "sad", "fun": "relaxed", "angry": "angry",
    "blink": "blink", "blink_l": "blinkLeft", "blink_r": "blinkRight",
    "lookup": "lookUp", "lookdown": "lookDown", "lookleft": "lookLeft", "lookright": "lookRight",
    "neutral": "neutral", "surprised": "surprised",
}

# VRM 0.x thumb names -> VRM 1.0 names.
_BONE_ALIASES = {}
for _side in ("left", "right"):
    _BONE_ALIASES[_side + "ThumbProximal"] = _side + "ThumbMetacarpal"
    _BONE_ALIASES[_side + "ThumbIntermediate"] = _side + "ThumbProximal"


@dataclass
class Node:
    index: int
    name: str
    parent: int
    children: List[int]
    local: np.ndarray
    world: np.ndarray = None

    @property
    def position(self) -> np.ndarray:
        return self.world[:3, 3]


@dataclass
class Primitive:
    mesh_index: int
    node_index: int
    material: int
    positions: np.ndarray            # (n, 3) world rest positions
    normals: np.ndarray              # (n, 3)
    uv0: Optional[np.ndarray]        # (n, 2)
    joints: np.ndarray               # (n, k) node indices (-1 = unused)
    weights: np.ndarray              # (n, k)
    indices: np.ndarray              # (m,) triangle list
    morphs: Dict[int, np.ndarray] = field(default_factory=dict)  # morph index -> world deltas
    colors: Optional[np.ndarray] = None


@dataclass
class VrmMaterial:
    name: str
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    texture: Optional[int] = None    # glTF image index
    alpha_mode: str = "OPAQUE"       # OPAQUE / MASK / BLEND
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    emissive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    shade_color: Optional[Tuple[float, float, float]] = None


@dataclass
class SpringJoint:
    node: int
    stiffness: float
    gravity_power: float
    gravity_dir: Tuple[float, float, float]
    drag: float
    hit_radius: float


@dataclass
class SpringChain:
    name: str
    joints: List[SpringJoint]
    collider_groups: List[int]


@dataclass
class Collider:
    node: int
    offset: np.ndarray
    radius: float
    tail: Optional[np.ndarray] = None  # capsule end (node local)


@dataclass
class ExpressionBind:
    mesh_index: int
    morph_index: int
    weight: float  # 0..1


@dataclass
class VrmModel:
    title: str
    version: int
    nodes: List[Node]
    humanoid: Dict[str, int]
    primitives: List[Primitive]
    materials: List[VrmMaterial]
    springs: List[SpringChain]
    colliders: List[Collider]
    collider_groups: List[List[int]]
    expressions: Dict[str, List[ExpressionBind]]
    gltf: Gltf

    def image(self, index: int):
        from PIL import Image
        return Image.open(io.BytesIO(self.gltf.image_bytes(index)))

    def bone_position(self, bone: str) -> Optional[np.ndarray]:
        idx = self.humanoid.get(bone)
        return None if idx is None else self.nodes[idx].position.copy()

    def node_children(self, index: int) -> List[int]:
        return self.nodes[index].children


def _vec3(v, default=(0.0, 0.0, 0.0)):
    if v is None:
        return np.array(default, dtype=np.float64)
    if isinstance(v, dict):
        return np.array([v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)], dtype=np.float64)
    return np.array(v, dtype=np.float64)


def load_vrm(path: str) -> VrmModel:
    g = Gltf.load(path)
    js = g.json
    ext = js.get("extensions", {})
    if "VRMC_vrm" in ext:
        version = 1
    elif "VRM" in ext:
        version = 0
    else:
        raise ValueError("This file is a plain glTF model, not a VRM. Export your model from VRoid Studio as VRM.")

    # -- nodes --------------------------------------------------------------
    raw_nodes = js.get("nodes", [])
    nodes = [Node(i, n.get("name", "node%d" % i), -1, list(n.get("children", [])), node_local_matrix(n))
             for i, n in enumerate(raw_nodes)]
    for n in nodes:
        for c in n.children:
            nodes[c].parent = n.index

    def compute_world(i):
        n = nodes[i]
        if n.world is None:
            n.world = n.local if n.parent < 0 else compute_world(n.parent) @ n.local
        return n.world

    for n in nodes:
        compute_world(n.index)

    # -- humanoid -------------------------------------------------------------
    humanoid: Dict[str, int] = {}
    if version == 1:
        for bone, info in ext["VRMC_vrm"].get("humanoid", {}).get("humanBones", {}).items():
            humanoid[bone] = info["node"]
    else:
        for hb in ext["VRM"].get("humanoid", {}).get("humanBones", []):
            if hb.get("node", -1) >= 0:
                humanoid[_BONE_ALIASES.get(hb["bone"], hb["bone"])] = hb["node"]
    for required in ("hips", "spine", "head", "leftUpperArm", "rightUpperArm", "leftUpperLeg", "rightUpperLeg"):
        if required not in humanoid:
            raise ValueError("The VRM humanoid is missing the '%s' bone." % required)

    # -- primitives -----------------------------------------------------------
    primitives: List[Primitive] = []
    skins = js.get("skins", [])
    for node in nodes:
        rn = raw_nodes[node.index]
        if "mesh" not in rn:
            continue
        mesh_index = rn["mesh"]
        mesh = js["meshes"][mesh_index]
        skin = skins[rn["skin"]] if "skin" in rn else None
        if skin is not None:
            joint_nodes = np.array(skin["joints"], dtype=np.int64)
            if "inverseBindMatrices" in skin:
                ibms = g.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
            else:
                ibms = np.tile(np.eye(4), (len(joint_nodes), 1, 1))
            joint_mats = np.stack([nodes[j].world for j in joint_nodes]) @ ibms
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            attrs = prim["attributes"]
            pos = g.accessor(attrs["POSITION"]).astype(np.float64)
            n = len(pos)
            nrm = g.accessor(attrs["NORMAL"]).astype(np.float64) if "NORMAL" in attrs else np.zeros((n, 3))
            uv = g.accessor(attrs["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in attrs else None
            col = g.accessor(attrs["COLOR_0"]).astype(np.float64) if "COLOR_0" in attrs else None
            idx = g.accessor(prim["indices"]).reshape(-1).astype(np.uint32) if "indices" in prim \
                else np.arange(n, dtype=np.uint32)
            morph_raw = {}
            for mi, target in enumerate(prim.get("targets", [])):
                if "POSITION" in target:
                    morph_raw[mi] = g.accessor(target["POSITION"]).astype(np.float64)

            if skin is not None and "JOINTS_0" in attrs:
                js_list, ws_list = [], []
                k = 0
                while "JOINTS_%d" % k in attrs:
                    js_list.append(g.accessor(attrs["JOINTS_%d" % k]).astype(np.int64))
                    ws_list.append(g.accessor(attrs["WEIGHTS_%d" % k]).astype(np.float64))
                    k += 1
                jidx = np.concatenate(js_list, axis=1)
                wts = np.concatenate(ws_list, axis=1)
                wsum = wts.sum(axis=1, keepdims=True)
                wts = np.where(wsum > 0, wts / np.maximum(wsum, 1e-12), 0)
                # Blend the skinning matrices to get each vertex's rest transform.
                vm = np.einsum("nk,nkij->nij", wts, joint_mats[jidx])
                world_pos = np.einsum("nij,nj->ni", vm[:, :3, :3], pos) + vm[:, :3, 3]
                world_nrm = np.einsum("nij,nj->ni", vm[:, :3, :3], nrm)
                morphs = {mi: np.einsum("nij,nj->ni", vm[:, :3, :3], d) for mi, d in morph_raw.items()}
                joints = np.where(wts > 0, joint_nodes[jidx], -1)
            else:
                m = node.world
                world_pos = pos @ m[:3, :3].T + m[:3, 3]
                world_nrm = nrm @ m[:3, :3].T
                morphs = {mi: d @ m[:3, :3].T for mi, d in morph_raw.items()}
                joints = np.full((n, 1), node.index, dtype=np.int64)
                wts = np.ones((n, 1))
            lens = np.linalg.norm(world_nrm, axis=1, keepdims=True)
            world_nrm = np.where(lens > 1e-12, world_nrm / np.maximum(lens, 1e-12), np.array([0, 1.0, 0]))
            primitives.append(Primitive(mesh_index, node.index, prim.get("material", -1), world_pos, world_nrm,
                                        uv, joints, wts, idx, morphs, col))

    # -- materials --------------------------------------------------------------
    textures = js.get("textures", [])

    def tex_image(tex_index):
        if tex_index is None or tex_index < 0 or tex_index >= len(textures):
            return None
        return textures[tex_index].get("source")

    materials: List[VrmMaterial] = []
    for m in js.get("materials", []):
        pbr = m.get("pbrMetallicRoughness", {})
        vm = VrmMaterial(name=m.get("name", "material%d" % len(materials)))
        vm.color = tuple(pbr.get("baseColorFactor", [1, 1, 1, 1]))
        if "baseColorTexture" in pbr:
            vm.texture = tex_image(pbr["baseColorTexture"]["index"])
        vm.alpha_mode = m.get("alphaMode", "OPAQUE")
        vm.alpha_cutoff = m.get("alphaCutoff", 0.5)
        vm.double_sided = bool(m.get("doubleSided", False))
        vm.emissive = tuple(m.get("emissiveFactor", [0, 0, 0]))
        mtoon = m.get("extensions", {}).get("VRMC_materials_mtoon")
        if mtoon and "shadeColorFactor" in mtoon:
            vm.shade_color = tuple(mtoon["shadeColorFactor"][:3])
        materials.append(vm)
    if version == 0:
        # VRM 0.x keeps the authoritative MToon settings in its own extension.
        for props in ext["VRM"].get("materialProperties", []):
            target = next((vm for vm in materials if vm.name == props.get("name")), None)
            if target is None:
                continue
            tp = props.get("textureProperties", {})
            fp = props.get("floatProperties", {})
            vp = props.get("vectorProperties", {})
            if target.texture is None and "_MainTex" in tp:
                target.texture = tex_image(tp["_MainTex"])
            if "_Color" in vp:
                target.color = tuple(vp["_Color"])
            if "_ShadeColor" in vp:
                target.shade_color = tuple(vp["_ShadeColor"][:3])
            blend = int(fp.get("_BlendMode", 0))
            target.alpha_mode = {0: "OPAQUE", 1: "MASK", 2: "BLEND", 3: "BLEND"}.get(blend, target.alpha_mode)
            target.alpha_cutoff = fp.get("_Cutoff", target.alpha_cutoff)
            if "_CullMode" in fp:
                target.double_sided = int(fp["_CullMode"]) == 0

    # -- spring bones -------------------------------------------------------------
    springs: List[SpringChain] = []
    colliders: List[Collider] = []
    collider_groups: List[List[int]] = []
    if version == 1:
        sb = ext.get("VRMC_springBone", {})
        for c in sb.get("colliders", []):
            shape = c.get("shape", {})
            if "sphere" in shape:
                s = shape["sphere"]
                colliders.append(Collider(c["node"], _vec3(s.get("offset")), float(s.get("radius", 0))))
            elif "capsule" in shape:
                s = shape["capsule"]
                colliders.append(Collider(c["node"], _vec3(s.get("offset")), float(s.get("radius", 0)),
                                          _vec3(s.get("tail"))))
        for grp in sb.get("colliderGroups", []):
            collider_groups.append(list(grp.get("colliders", [])))
        for i, spring in enumerate(sb.get("springs", [])):
            joints = []
            for jt in spring.get("joints", []):
                joints.append(SpringJoint(jt["node"], float(jt.get("stiffness", 1.0)),
                                          float(jt.get("gravityPower", 0.0)),
                                          tuple(jt.get("gravityDir", [0, -1, 0])),
                                          float(jt.get("dragForce", 0.5)), float(jt.get("hitRadius", 0.0))))
            if len(joints) >= 1:
                springs.append(SpringChain(spring.get("name") or "spring%d" % i, joints,
                                           list(spring.get("colliderGroups", []))))
    else:
        sa = ext["VRM"].get("secondaryAnimation", {})
        for grp in sa.get("colliderGroups", []):
            members = []
            for c in grp.get("colliders", []):
                members.append(len(colliders))
                colliders.append(Collider(grp["node"], _vec3(c.get("offset")), float(c.get("radius", 0))))
            collider_groups.append(members)
        for gi, bg in enumerate(sa.get("boneGroups", [])):
            stiff = float(bg.get("stiffiness", bg.get("stiffness", 1.0)))
            params = dict(stiffness=stiff, gravity_power=float(bg.get("gravityPower", 0.0)),
                          gravity_dir=tuple(_vec3(bg.get("gravityDir"), (0, -1, 0))),
                          drag=float(bg.get("dragForce", 0.5)), hit_radius=float(bg.get("hitRadius", 0.02)))
            for root in bg.get("bones", []):
                # Each root's whole subtree is simulated: split it into linear chains.
                for chain in _linear_chains(nodes, root):
                    joints = [SpringJoint(n, **params) for n in chain]
                    springs.append(SpringChain("%s_%d" % (bg.get("comment") or "group%d" % gi, len(springs)),
                                               joints, list(bg.get("colliderGroups", []))))

    # -- expressions --------------------------------------------------------------
    expressions: Dict[str, List[ExpressionBind]] = {}
    if version == 1:
        ex = ext["VRMC_vrm"].get("expressions", {})
        groups = dict(ex.get("preset", {}))
        groups.update(ex.get("custom", {}))
        for name, e in groups.items():
            binds = []
            for b in e.get("morphTargetBinds", []):
                rn = raw_nodes[b["node"]]
                if "mesh" in rn:
                    binds.append(ExpressionBind(rn["mesh"], b["index"], float(b.get("weight", 1.0))))
            expressions[name] = binds
    else:
        for grp in ext["VRM"].get("blendShapeMaster", {}).get("blendShapeGroups", []):
            preset = (grp.get("presetName") or "unknown").lower()
            name = _EXPRESSION_ALIASES.get(preset) if preset != "unknown" else grp.get("name", "")
            if not name:
                continue
            binds = [ExpressionBind(b["mesh"], b["index"], float(b.get("weight", 100)) / 100.0)
                     for b in grp.get("binds", [])]
            expressions[name] = binds

    title = ""
    if version == 1:
        title = ext["VRMC_vrm"].get("meta", {}).get("name", "")
    else:
        title = ext["VRM"].get("meta", {}).get("title", "")

    return VrmModel(title or "VRoid", version, nodes, humanoid, primitives, materials, springs, colliders,
                    collider_groups, expressions, g)


def _linear_chains(nodes: List[Node], root: int) -> List[List[int]]:
    """Split the subtree under ``root`` into linear chains (first child continues)."""
    chains = []
    stack = [root]
    while stack:
        start = stack.pop()
        chain = [start]
        cur = start
        while nodes[cur].children:
            kids = nodes[cur].children
            for other in kids[1:]:
                stack.append(other)
            cur = kids[0]
            chain.append(cur)
        chains.append(chain)
    return chains
