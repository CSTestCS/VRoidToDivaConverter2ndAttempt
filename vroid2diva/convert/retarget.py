"""Fit a VRM model onto DIVA's skeleton.

1. A global similarity transform turns the model to face the same way as the
   reference, scales it to the reference height and centres it.
2. Arms, hands, fingers and legs are re-posed so every limb points the same
   way as in the reference bind pose (VRoid models are in a T-pose, DIVA
   modules are bound in the game's own rest pose).
3. Every DIVA bone gets a bind frame: the reference orientation, placed at the
   matching joint of the re-posed VRM model.

Because the bind frames share the reference orientations, the game's motions
rotate the VRoid mesh exactly like they rotate Miku.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..vrm.model import VrmModel
from .bonemap import HAND_ROLL, HUMANOID_TO_DIVA, REPOSE_CHAINS
from .reference import Reference


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return axis_angle(axis, np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def _about(pivot: np.ndarray, r: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = pivot - r @ pivot
    return m


class Retarget:
    def __init__(self, model: VrmModel, ref: Reference, log=print):
        self.model = model
        self.ref = ref
        self.log = log
        self.mapping: Dict[str, str] = {}  # VRM humanoid bone -> DIVA bone
        for bone, cands in HUMANOID_TO_DIVA.items():
            if bone in model.humanoid:
                chosen = ref.first(cands)
                if chosen:
                    self.mapping[bone] = chosen
        if "head" not in self.mapping or "hips" not in self.mapping:
            raise ValueError("The reference module does not contain the head/hip bones needed for binding.")

        n = len(model.nodes)
        self.orig_world = np.stack([nd.world for nd in model.nodes])
        self.global_xf = self._global_transform()
        world = np.einsum("ij,njk->nik", self.global_xf, self.orig_world)
        self.world = world
        self._repose()
        # Per-node deformation from the original file pose to the bound pose.
        self.delta = np.einsum("nij,njk->nik", self.world, np.linalg.inv(self.orig_world))
        self.scale = float(np.cbrt(abs(np.linalg.det(self.global_xf[:3, :3]))))
        self.humanoid_node_bone = {idx: bone for bone, idx in model.humanoid.items()}
        self.overrides: Dict[str, np.ndarray] = {}  # explicit bind positions (face bones)
        self.face_fit: List[dict] = []  # filled by the face baker (used for previews)

    # ------------------------------------------------------------------
    def _ref_pos(self, vrm_bone: str) -> Optional[np.ndarray]:
        cands = HUMANOID_TO_DIVA.get(vrm_bone, [])
        for c in cands:
            if c in self.ref.bones:
                return self.ref.position(c)
        return None

    def _global_transform(self) -> np.ndarray:
        model, ref = self.model, self.ref
        vrm_l = model.bone_position("leftUpperArm")
        vrm_r = model.bone_position("rightUpperArm")
        ref_l = self._ref_pos("leftUpperArm")
        ref_r = self._ref_pos("rightUpperArm")
        rot = np.eye(3)
        if ref_l is not None and ref_r is not None:
            vrm_side = vrm_l - vrm_r
            ref_side = ref_l - ref_r
            vrm_side[1] = ref_side[1] = 0
            angle = np.arctan2(np.cross(vrm_side, ref_side)[1], np.dot(vrm_side, ref_side))
            rot = axis_angle([0, 1, 0], angle)
        head_vrm = model.bone_position("head")
        head_ref = self.ref.position(self.mapping["head"])
        foot = min((p[1] for p in (model.bone_position("leftFoot"), model.bone_position("rightFoot"))
                    if p is not None), default=0.0)
        foot_ref = min((self._ref_pos(b)[1] for b in ("leftFoot", "rightFoot") if self._ref_pos(b) is not None),
                       default=0.0)
        scale = (head_ref[1] - foot_ref) / max(head_vrm[1] - foot, 1e-6)
        if not 0.2 < scale < 5.0:
            self.log("Warning: unusual scale factor %.3f, check the model's units." % scale)
        m = np.eye(4)
        m[:3, :3] = rot * scale
        hips = m[:3, :3] @ model.bone_position("hips")
        hips_ref = self.ref.position(self.mapping["hips"])
        m[:3, 3] = [hips_ref[0] - hips[0], (foot_ref - foot * scale), hips_ref[2] - hips[2]]
        return m

    def _subtree(self, root: int) -> List[int]:
        out, stack = [], [root]
        while stack:
            i = stack.pop()
            out.append(i)
            stack.extend(self.model.nodes[i].children)
        return out

    def _rotate_subtree(self, node: int, r: np.ndarray) -> None:
        pivot = self.world[node][:3, 3].copy()
        m = _about(pivot, r)
        idx = self._subtree(node)
        self.world[idx] = np.einsum("ij,njk->nik", m, self.world[idx])

    def _repose(self) -> None:
        hum = self.model.humanoid
        for parent, child in REPOSE_CHAINS:
            if parent not in hum or child not in hum:
                continue
            ra, rb = self._ref_pos(parent), self._ref_pos(child)
            if ra is None or rb is None:
                continue
            a = self.world[hum[parent]][:3, 3]
            b = self.world[hum[child]][:3, 3]
            if np.linalg.norm(b - a) < 1e-6 or np.linalg.norm(rb - ra) < 1e-6:
                continue
            self._rotate_subtree(hum[parent], rotation_between(b - a, rb - ra))
            if parent in HAND_ROLL:
                self._fix_roll(parent, child, *HAND_ROLL[parent])

    def _fix_roll(self, bone: str, dir_child: str, side_a: str, side_b: str) -> None:
        hum = self.model.humanoid
        if side_a not in hum or side_b not in hum:
            return
        ra, rb = self._ref_pos(side_a), self._ref_pos(side_b)
        if ra is None or rb is None:
            return
        axis = self.world[hum[dir_child]][:3, 3] - self.world[hum[bone]][:3, 3]
        axis /= max(np.linalg.norm(axis), 1e-12)
        cur = self.world[hum[side_b]][:3, 3] - self.world[hum[side_a]][:3, 3]
        tgt = rb - ra
        cur -= axis * np.dot(cur, axis)
        tgt -= axis * np.dot(tgt, axis)
        if np.linalg.norm(cur) < 1e-6 or np.linalg.norm(tgt) < 1e-6:
            return
        angle = np.arctan2(np.dot(np.cross(cur, tgt), axis), np.dot(cur, tgt))
        self._rotate_subtree(hum[bone], axis_angle(axis, angle))

    # ------------------------------------------------------------------
    def node_position(self, node: int) -> np.ndarray:
        return self.world[node][:3, 3].copy()

    def bound_position(self, diva_bone: str) -> np.ndarray:
        """Where ``diva_bone`` sits on the re-posed VRM model."""
        if diva_bone in self.overrides:
            return self.overrides[diva_bone].copy()
        for bone, target in self.mapping.items():
            if target == diva_bone:
                return self.node_position(self.model.humanoid[bone])
        # Not a humanoid bone: carry the reference position along with the
        # closest mapped bone.
        ref_p = self.ref.position(diva_bone)
        best, best_d = None, None
        for bone, target in self.mapping.items():
            d = np.linalg.norm(self.ref.position(target) - ref_p)
            if best_d is None or d < best_d:
                best, best_d = bone, d
        anchor = self.mapping[best]
        return ref_p + (self.node_position(self.model.humanoid[best]) - self.ref.position(anchor))

    def bind_frame(self, diva_bone: str, position: np.ndarray) -> np.ndarray:
        m = self.ref.bones[diva_bone].world.copy()
        m[:3, 3] = position
        return m

    def deform(self, joints: np.ndarray, weights: np.ndarray, positions: np.ndarray,
               normals: np.ndarray, morphs: Dict[int, np.ndarray]):
        """Linear-blend the per-node deltas onto vertex data."""
        safe = np.where(joints >= 0, joints, 0)
        mats = np.einsum("nk,nkij->nij", weights, self.delta[safe])
        pos = np.einsum("nij,nj->ni", mats[:, :3, :3], positions) + mats[:, :3, 3]
        nrm = np.einsum("nij,nj->ni", mats[:, :3, :3], normals)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        out_morphs = {k: np.einsum("nij,nj->ni", mats[:, :3, :3], d) for k, d in morphs.items()}
        return pos, nrm, out_morphs

    def ancestor_target(self, node: int, targets: Dict[int, str]) -> str:
        """Nearest DIVA bone for an arbitrary VRM node."""
        cur = node
        while cur >= 0:
            if cur in targets:
                return targets[cur]
            bone = self.humanoid_node_bone.get(cur)
            if bone and bone in self.mapping:
                return self.mapping[bone]
            cur = self.model.nodes[cur].parent
        return self.mapping["hips"]
