"""Bake VRM blend shapes onto DIVA's facial bones.

Mega Mix+ animates faces (expressions, blinking, lip sync) by moving face
bones, not by morph targets.  To make the game's facial animation drive a
VRoid face, each face vertex is skinned to the face bone that moves that part
of Miku's face, with a weight derived from how far the vertex moves in the
matching VRoid blend shape:

* rotation bones (jaw, eyelids): weight = vertex rotation angle about the bone
  pivot / the largest angle in that bone's region;
* translation bones (mouth corners, lips, brows): weight = vertex displacement
  / the largest displacement in that bone's region.

So the part of the face that moves the most in VRoid's "A" shape follows the
jaw completely, the parts that move less follow it partially, and so on.  The
bone list below is matched against the reference head skin; bones that do not
exist there are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FaceBone:
    name: str
    kind: str                  # "rot" or "move"
    drivers: Sequence[str]     # VRM expressions, in order of preference
    side: Optional[str] = None  # "up" / "down" relative to the eye centre


def _eyelids(d: str, drv: str) -> List[FaceBone]:
    return [FaceBone("kl_mabu_u_%s_wj" % d, "rot", (drv, "blink"), "up"),
            FaceBone("kl_mabu_d_%s_wj" % d, "rot", (drv, "blink"), "down")]


FACE_RIG: List[FaceBone] = (
    [FaceBone("kl_ago_wj", "rot", ("aa", "oh"))]
    + _eyelids("l", "blinkLeft") + _eyelids("r", "blinkRight")
    + [FaceBone(n, "move", ("ih", "ee", "happy")) for n in ("tl_kuti_l_wj", "tl_kuti_r_wj")]
    + [FaceBone(n, "move", ("ou", "oh")) for n in ("tl_kuti_u_wj", "tl_kuti_d_wj", "tl_kuti_u_l_wj",
                                                     "tl_kuti_u_r_wj", "tl_kuti_d_l_wj", "tl_kuti_d_r_wj")]
    + [FaceBone(n, "move", ("angry", "sad", "surprised", "happy"))
       for n in ("tl_mayu_l_wj", "tl_mayu_r_wj", "tl_mayu_b_l_wj", "tl_mayu_b_r_wj",
                 "tl_mayu_c_l_wj", "tl_mayu_c_r_wj")]
)

MIN_MOTION = 1e-4  # metres (after scaling) below which a vertex is considered static


def face_similarity(ref, rt) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (scale, ref_anchor, vrm_anchor) mapping reference face space to the VRM face."""
    model = rt.model
    eyes = []
    for vrm_eye, diva_eye in (("leftEye", "kl_eye_l_wj"), ("rightEye", "kl_eye_r_wj")):
        if vrm_eye in model.humanoid and diva_eye in ref.bones:
            eyes.append((rt.node_position(model.humanoid[vrm_eye]), ref.position(diva_eye)))
    head_vrm = rt.node_position(model.humanoid["head"])
    head_ref = ref.position(rt.mapping["head"])
    if len(eyes) == 2:
        vrm_mid = (eyes[0][0] + eyes[1][0]) / 2
        ref_mid = (eyes[0][1] + eyes[1][1]) / 2
        d_vrm = np.linalg.norm(eyes[0][0] - eyes[1][0])
        d_ref = np.linalg.norm(eyes[0][1] - eyes[1][1])
        scale = d_vrm / d_ref if d_ref > 1e-6 else 1.0
        return float(np.clip(scale, 0.5, 2.0)), ref_mid, vrm_mid
    return 1.0, head_ref, head_vrm


def expression_deltas(model, prim_ranges, prim_morphs, total: int) -> Dict[str, np.ndarray]:
    """Combine morph target binds into per-vertex deltas for each expression."""
    out = {}
    by_mesh: Dict[int, List[int]] = {}
    for pi, (mesh_index, _start, _end) in enumerate(prim_ranges):
        by_mesh.setdefault(mesh_index, []).append(pi)
    for name, binds in model.expressions.items():
        if not binds:
            continue
        delta = np.zeros((total, 3))
        for b in binds:
            for pi in by_mesh.get(b.mesh_index, []):
                d = prim_morphs[pi].get(b.morph_index)
                if d is None:
                    continue
                _, start, end = prim_ranges[pi]
                delta[start:end] += d * b.weight
        if np.abs(delta).max() > 0:
            out[name] = delta
    return out


def bake_face(ref, rt, positions: np.ndarray, inf_idx: np.ndarray, inf_w: np.ndarray, names: List[str],
              deltas: Dict[str, np.ndarray], log=print):
    """Return updated (inf_idx, inf_w, names) with face bone influences added."""
    head = rt.mapping["head"]
    available = [fb for fb in FACE_RIG if fb.name in ref.bones]
    if not available or not deltas:
        log("Face: no face bones or blend shapes found; facial animation will not be baked.")
        return inf_idx, inf_w, names

    scale, ref_anchor, vrm_anchor = face_similarity(ref, rt)

    def to_vrm(p):
        return vrm_anchor + (p - ref_anchor) * scale

    bone_pos = {fb.name: to_vrm(ref.position(fb.name)) for fb in available}
    for fb in available:
        rt.overrides[fb.name] = bone_pos[fb.name]

    # Face vertices: mostly bound to the head (or eyes).
    face_targets = {head, "kl_eye_l_wj", "kl_eye_r_wj"}
    target_ids = [i for i, n in enumerate(names) if n in face_targets]
    head_w = np.zeros(len(positions))
    for t in target_ids:
        head_w += np.where(inf_idx == t, inf_w, 0).sum(axis=1)
    is_face = head_w > 0.5

    eye_center = {}
    for d, vrm_eye in (("l", "leftEye"), ("r", "rightEye")):
        if vrm_eye in rt.model.humanoid:
            eye_center[d] = rt.node_position(rt.model.humanoid[vrm_eye])
        elif "kl_eye_%s_wj" % d in ref.bones:
            eye_center[d] = to_vrm(ref.position("kl_eye_%s_wj" % d))

    lateral = np.array([1.0, 0.0, 0.0])
    if "kl_eye_l_wj" in ref.bones and "kl_eye_r_wj" in ref.bones:
        lateral = ref.position("kl_eye_l_wj") - ref.position("kl_eye_r_wj")
        lateral /= max(np.linalg.norm(lateral), 1e-12)

    face_idx = np.nonzero(is_face)[0]
    if len(face_idx) == 0:
        return inf_idx, inf_w, names
    pts = positions[face_idx]

    best_w = np.zeros(len(face_idx))
    best_bone = np.full(len(face_idx), -1)
    fitted = []
    fit_info = rt.face_fit = []
    for bi, fb in enumerate(available):
        driver = next((d for d in fb.drivers if d in deltas), None)
        if driver is None:
            continue
        delta = deltas[driver][face_idx]
        q = bone_pos[fb.name]
        if fb.kind == "rot":
            r = pts - q
            r_perp = r - np.outer(r @ lateral, lateral)
            radius = np.linalg.norm(r_perp, axis=1)
            tangent = np.cross(lateral, r_perp)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
            motion = np.einsum("ij,ij->i", delta, tangent) / np.maximum(radius, 1e-3)
            mag = np.linalg.norm(delta, axis=1)
            movers = mag > MIN_MOTION
            if movers.sum() < 3:
                continue
            # Positive direction = direction of the biggest movers.
            top = np.argsort(-mag)[: max(3, int(movers.sum() * 0.1))]
            sign = 1.0 if np.median(motion[top]) >= 0 else -1.0
            motion = np.where(movers, motion * sign, 0.0)
            direction = lateral * sign
        else:
            motion = np.linalg.norm(delta, axis=1)
            motion = np.where(motion > MIN_MOTION, motion, 0.0)
            direction = None

        # Region: vertices closer to this bone than to any other bone of the same
        # kind and driver family; eyelids additionally split up/down.
        dist = np.linalg.norm(pts - q, axis=1)
        region = motion > 0
        for other in available:
            if other is fb or other.kind != fb.kind or other.drivers != fb.drivers:
                continue
            if other.side and fb.side and other.side != fb.side:
                continue
            region &= dist <= np.linalg.norm(pts - bone_pos[other.name], axis=1) + 1e-9
        if fb.side:
            d = "l" if "_l_" in fb.name else "r"
            if d in eye_center:
                above = pts[:, 1] >= eye_center[d][1]
                region &= above if fb.side == "up" else ~above
        if region.sum() < 3:
            continue
        peak = np.percentile(motion[region], 95)
        if peak <= 0:
            continue
        w = np.where(region, np.clip(motion / peak, 0.0, 1.0), 0.0)
        better = w > best_w
        best_w = np.where(better, w, best_w)
        best_bone = np.where(better, bi, best_bone)
        if direction is None:
            sel_r = region & (motion >= peak * 0.5)
            mean = delta[sel_r].mean(axis=0) if sel_r.any() else np.zeros(3)
            direction = mean / max(np.linalg.norm(mean), 1e-12)
        fitted.append("%s<-%s" % (fb.name, driver))
        fit_info.append({"bone": fb.name, "kind": fb.kind, "driver": driver, "amount": float(peak),
                         "axis": [float(x) for x in direction]})

    if not fitted:
        log("Face: blend shapes did not move any face vertices; skipping.")
        return inf_idx, inf_w, names

    # Add one extra influence slot holding the face bone.
    names = list(names)
    name_index = {n: i for i, n in enumerate(names)}
    for fb in available:
        if fb.name not in name_index:
            name_index[fb.name] = len(names)
            names.append(fb.name)
    new_idx = np.concatenate([inf_idx, np.full((len(inf_idx), 1), -1)], axis=1)
    new_w = np.concatenate([inf_w, np.zeros((len(inf_w), 1))], axis=1)
    sel = best_bone >= 0
    rows = face_idx[sel]
    w = best_w[sel]
    new_w[rows, :-1] *= (1.0 - w)[:, None]
    new_idx[rows, -1] = [name_index[available[b].name] for b in best_bone[sel]]
    new_w[rows, -1] = w
    log("Face: baked %d blend shapes onto face bones (%s); %d face vertices animated."
        % (len(fitted), ", ".join(fitted), int(sel.sum())))
    return new_idx, new_w, names
