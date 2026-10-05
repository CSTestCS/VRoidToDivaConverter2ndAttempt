"""The reference skeleton: how the game's own Miku module is bound.

Everything the converter needs to know about DIVA's skeleton comes from here:
object bone IDs, bind-pose matrices, and the expression/constraint blocks that
define helper bones such as the arm twist bones.  It is extracted from the
module being replaced, so it always matches the user's game version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

from ..formats.objset import (ConstraintBlock, ExpressionBlock, Material, MotionBlock, ObjectSet,
                              from_diva_matrix)

_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+")


@dataclass
class RefBone:
    name: str
    id: int
    is_ex: bool
    world: np.ndarray  # column-vector bind matrix (orthonormal rotation + translation)
    parent: Optional[str] = None


def orthonormalize(m: np.ndarray) -> np.ndarray:
    out = np.array(m, dtype=np.float64)
    u, _, vt = np.linalg.svd(out[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    out[:3, :3] = r
    out[3, :] = [0, 0, 0, 1]
    return out


@dataclass
class Reference:
    bones: Dict[str, RefBone] = field(default_factory=dict)
    blocks: List[object] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    collision_bones: Optional[Set[str]] = None
    motion_bone_names: Set[str] = field(default_factory=set)
    source_objects: List[str] = field(default_factory=list)

    def has(self, name: str) -> bool:
        return name in self.bones

    def position(self, name: str) -> np.ndarray:
        return self.bones[name].world[:3, 3].copy()

    def first(self, candidates: Iterable[str]) -> Optional[str]:
        for c in candidates:
            if c in self.bones and (not self.bones[c].is_ex or self.defining_block(c) is not None):
                return c
        return None

    # -- ex-data block handling -------------------------------------------------
    def _defined_by(self, block) -> List[str]:
        if isinstance(block, MotionBlock):
            return [n for n, _ in block.nodes]
        return [block.name]

    def defining_block(self, name: str):
        for b in self.blocks:
            if name in self._defined_by(b):
                return b
        return None

    def _references(self, block) -> Set[str]:
        refs = set()
        if block.parent_name:
            refs.add(block.parent_name)
        if isinstance(block, ExpressionBlock):
            for e in block.expressions:
                refs.update(_IDENT.findall(e or ""))
        elif isinstance(block, ConstraintBlock):
            if block.source_name:
                refs.add(block.source_name)
            refs.update(v for k, v in block.data if k == "s" and v)
        return refs

    def resolve_blocks(self, ex_names: Iterable[str]):
        """Return the blocks needed for ``ex_names`` (with dependencies) and the
        set of names that could not be resolved."""
        needed: List[object] = []
        missing: Set[str] = set()
        known_motion = {n for n, b in self.bones.items() if not b.is_ex} | self.motion_bone_names
        stack = list(ex_names)
        seen = set()
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            block = self.defining_block(name)
            if block is None:
                if name not in known_motion:
                    missing.add(name)
                continue
            if block not in needed:
                needed.append(block)
                for ref in self._references(block):
                    if ref not in seen and (self.defining_block(ref) is not None or ref not in known_motion):
                        stack.append(ref)
        # Keep original order (definition order matters for evaluation).
        ordered = [b for b in self.blocks if b in needed]
        return ordered, missing

    def motion_parent(self, name: str) -> str:
        """Walk up ex-data parents until a real motion bone is reached."""
        seen = set()
        while name and name not in seen:
            seen.add(name)
            bone = self.bones.get(name)
            if bone is not None and not bone.is_ex:
                return name
            block = self.defining_block(name)
            if block is None or not block.parent_name:
                break
            name = block.parent_name
        return "kl_mune_b_wj" if "kl_mune_b_wj" in self.bones else next(
            n for n, b in self.bones.items() if not b.is_ex)


def reference_from_objsets(objsets: List[ObjectSet], collision_bones: Optional[Set[str]] = None,
                           motion_bone_names: Optional[Set[str]] = None) -> Reference:
    ref = Reference(collision_bones=collision_bones, motion_bone_names=set(motion_bone_names or ()))
    seen_blocks = set()
    for objset in objsets:
        for obj in objset.objects:
            ref.materials.extend(obj.materials)
            if obj.skin is None:
                continue
            ref.source_objects.append(obj.name)
            for bone in obj.skin.bones:
                if bone.name in ref.bones:
                    continue
                world = orthonormalize(np.linalg.inv(from_diva_matrix(bone.ibm)))
                ref.bones[bone.name] = RefBone(bone.name, bone.id, bone.is_ex, world, bone.parent)
            for block in obj.skin.blocks:
                if isinstance(block, (ExpressionBlock, ConstraintBlock, MotionBlock)):
                    key = (block.signature, block.name, tuple(n for n, _ in getattr(block, "nodes", [])))
                    if key not in seen_blocks:
                        seen_blocks.add(key)
                        ref.blocks.append(block)
    if not ref.bones:
        raise ValueError("The selected module has no skinned objects to use as a reference skeleton.")
    return ref
