"""VRM spring bones -> DIVA osage chains.

DIVA simulates "osage" chains defined in the object's skin ex-data:

* a root node attached to a motion bone (``OSG`` block transform),
* joint nodes; joint *i*'s matrix sits at the previous joint's position and
  its X axis points at joint *i* (``length`` = distance to the previous one),
* an effector at the tip.

A VRM spring bone ``b_k`` rotates the mesh between ``b_k`` and ``b_k+1``, which
is exactly DIVA joint ``k+1``, so vertex weights map one to one.

Physics parameters live in ``rom/skin_param/ext_skp_<object>.txt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..formats.objset import OsageBlock, OsageNode
from ..vrm.model import VrmModel
from .reference import Reference
from .retarget import Retarget

MAX_COLLISIONS = 12


@dataclass
class OsageChain:
    index: int
    root_name: str
    effector_name: str
    node_names: List[str]
    vrm_nodes: List[int]
    parent_bone: str
    points: np.ndarray        # (n+2, 3): p0..p_{n+1}
    frames: List[np.ndarray]  # bind frames of each joint (column 4x4)
    block: OsageBlock = None
    params: Dict[str, float] = field(default_factory=dict)
    collisions: List[dict] = field(default_factory=list)


def euler_zyx(r: np.ndarray) -> List[float]:
    """Angles (x, y, z) such that r = Rz(z) @ Ry(y) @ Rx(x) (DIVA's rotate_zyx)."""
    sy = -r[2, 0]
    sy = max(-1.0, min(1.0, sy))
    y = np.arcsin(sy)
    if abs(sy) < 0.99999:
        x = np.arctan2(r[2, 1], r[2, 2])
        z = np.arctan2(r[1, 0], r[0, 0])
    else:
        x = np.arctan2(-r[1, 2], r[1, 1])
        z = 0.0
    return [float(x), float(y), float(z)]


def _frame(origin: np.ndarray, x_dir: np.ndarray, prev_y: Optional[np.ndarray]) -> np.ndarray:
    x = x_dir / max(np.linalg.norm(x_dir), 1e-12)
    hint = prev_y if prev_y is not None else np.array([0.0, 0.0, 1.0])
    y = hint - x * np.dot(hint, x)
    if np.linalg.norm(y) < 1e-4:
        for alt in ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]):
            alt = np.array(alt)
            y = alt - x * np.dot(alt, x)
            if np.linalg.norm(y) > 1e-4:
                break
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, origin
    return m


def build_osage(model: VrmModel, rt: Retarget, ref: Reference, bone_targets: Dict[int, str],
                log=print) -> List[OsageChain]:
    """Create osage chains.  ``bone_targets`` (VRM node -> DIVA bone) is updated
    with the osage joint names for every simulated node."""
    chains: List[OsageChain] = []
    claimed = set()
    for spring in model.springs:
        nodes = [j.node for j in spring.joints if j.node not in claimed]
        # VRM 1.0 lists the tail joint (which itself does not deform) last.
        if model.version == 1 and len(nodes) > 1:
            tail_node = nodes[-1]
            nodes = nodes[:-1]
        else:
            tail_node = None
        if not nodes:
            continue
        idx = len(chains)
        pts = [rt.node_position(n) for n in nodes]
        if tail_node is not None:
            tail = rt.node_position(tail_node)
        else:
            last = nodes[-1]
            kids = model.nodes[last].children
            if kids:
                tail = rt.node_position(kids[0])
            else:
                prev = pts[-2] if len(pts) > 1 else rt.node_position(model.nodes[nodes[0]].parent)
                d = pts[-1] - prev
                if np.linalg.norm(d) < 1e-6:
                    d = np.array([0.0, -1.0, 0.0])
                tail = pts[-1] + d / np.linalg.norm(d) * 0.07 * rt.scale
        pts.append(tail)
        # Zero-length segments break the solver: merge them into the previous joint.
        seg_nodes, seg_pts, merged = [], [], {}
        for k, n in enumerate(nodes):
            if np.linalg.norm(pts[k + 1] - pts[k]) < 1e-4:
                merged[n] = len(seg_nodes) - 1
                continue
            seg_nodes.append(n)
            seg_pts.append(pts[k])
        if not seg_nodes:
            continue
        seg_pts.append(pts[-1])
        nodes, pts = seg_nodes, np.array(seg_pts)

        parent_node = model.nodes[nodes[0]].parent
        parent_bone = ref.motion_parent(rt.ancestor_target(parent_node, bone_targets) if parent_node >= 0
                                        else rt.mapping["hips"])
        frames = []
        prev_y = None
        for k in range(len(nodes)):
            f = _frame(pts[k], pts[k + 1] - pts[k], prev_y)
            prev_y = f[:3, 1]
            frames.append(f)

        prefix = "v%02d" % idx
        node_names = ["j_%s_%03d_wj" % (prefix, k) for k in range(len(nodes))]
        chain = OsageChain(idx, "c_%s_osg" % prefix, "e_%s_osg" % prefix, node_names, nodes, parent_bone,
                           pts, frames)
        for n, name in zip(nodes, node_names):
            bone_targets[n] = name
            claimed.add(n)
        for n, k in merged.items():
            bone_targets[n] = node_names[max(k, 0)]
            claimed.add(n)
        if tail_node is not None and tail_node not in bone_targets:
            bone_targets[tail_node] = node_names[-1]

        parent_frame = rt.bind_frame(parent_bone, rt.bound_position(parent_bone))
        local = np.linalg.inv(parent_frame) @ frames[0]
        block = OsageBlock()
        block.parent_name = parent_bone
        block.position = [float(v) for v in local[:3, 3]]
        block.rotation = euler_zyx(local[:3, :3])
        block.scale = [1.0, 1.0, 1.0]
        block.name = chain.effector_name
        block.external_name = chain.root_name
        block.nodes = [OsageNode(name, float(np.linalg.norm(pts[k + 1] - pts[k])))
                       for k, name in enumerate(node_names)]
        chain.block = block

        joints = [j for j in spring.joints if j.node in nodes] or spring.joints
        stiffness = float(np.mean([j.stiffness for j in joints]))
        drag = float(np.mean([j.drag for j in joints]))
        gravity = float(np.mean([j.gravity_power for j in joints]))
        radius = float(np.mean([j.hit_radius for j in joints])) * rt.scale
        chain.params = {
            "force": float(np.clip(0.008 * stiffness + 0.002, 0.002, 0.06)),
            "force_gain": 0.3,
            "air_res": float(np.clip(0.15 + 0.6 * drag, 0.1, 0.9)),
            "friction": 0.5,
            "wind_afc": 0.5,
            "hinge": 90.0 if gravity < 0.5 else 70.0,
            "coli_r": max(radius, 0.005),
        }
        chain.collisions = _collisions(model, rt, ref, spring.collider_groups, bone_targets)
        chains.append(chain)
    if chains:
        log("Physics: %d spring chains converted to osage (%d joints)."
            % (len(chains), sum(len(c.node_names) for c in chains)))
    return chains


def _collisions(model: VrmModel, rt: Retarget, ref: Reference, groups: List[int],
                bone_targets: Dict[int, str]) -> List[dict]:
    if not ref.collision_bones:
        return []
    out = []
    for g in groups:
        if g >= len(model.collider_groups):
            continue
        for ci in model.collider_groups[g]:
            c = model.colliders[ci]
            bone = rt.ancestor_target(c.node, {})
            bone = ref.motion_parent(bone)
            if bone not in ref.collision_bones:
                continue
            frame = rt.bind_frame(bone, rt.bound_position(bone))
            inv = np.linalg.inv(frame)
            node_world = rt.world[c.node]

            def to_bone(local_off):
                p = node_world @ np.append(local_off, 1.0)
                return (inv @ p)[:3]

            entry = {"type": 1, "radius": c.radius * rt.scale, "bone0": bone, "pos0": to_bone(c.offset)}
            if c.tail is not None:
                entry["type"] = 2
                entry["bone1"] = bone
                entry["pos1"] = to_bone(c.tail)
            out.append(entry)
            if len(out) >= MAX_COLLISIONS:
                return out
    return out


def skin_param_text(chains: List[OsageChain]) -> Dict[str, object]:
    kv: Dict[str, object] = {}
    for c in chains:
        r = c.root_name
        p = c.params
        kv[r + ".node.length"] = len(c.node_names)
        for k in range(len(c.node_names)):
            kv["%s.node.%d.coli_r" % (r, k)] = p["coli_r"]
            kv["%s.node.%d.weight" % (r, k)] = 1.0
        kv[r + ".root.force"] = p["force"]
        kv[r + ".root.force_gain"] = p["force_gain"]
        kv[r + ".root.air_res"] = p["air_res"]
        kv[r + ".root.rot_y"] = 0.0
        kv[r + ".root.rot_z"] = 0.0
        kv[r + ".root.init_rot_y"] = 0.0
        kv[r + ".root.init_rot_z"] = 0.0
        kv[r + ".root.friction"] = p["friction"]
        kv[r + ".root.wind_afc"] = p["wind_afc"]
        kv[r + ".root.coli_type"] = 0
        kv[r + ".root.hinge_y"] = p["hinge"]
        kv[r + ".root.hinge_z"] = p["hinge"]
        kv[r + ".root.coli_r"] = p["coli_r"]
        if c.collisions:
            kv[r + ".root.coli.length"] = len(c.collisions)
            for i, col in enumerate(c.collisions):
                base = "%s.root.coli.%d." % (r, i)
                kv[base + "type"] = col["type"]
                kv[base + "radius"] = float(col["radius"])
                kv[base + "bone.0.name"] = col["bone0"]
                for axis, v in zip("xyz", col["pos0"]):
                    kv[base + "bone.0.pos" + axis] = float(v)
                if "bone1" in col:
                    kv[base + "bone.1.name"] = col["bone1"]
                    for axis, v in zip("xyz", col["pos1"]):
                        kv[base + "bone.1.pos" + axis] = float(v)
    return kv
