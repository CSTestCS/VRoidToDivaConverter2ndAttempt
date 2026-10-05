"""Synthetic stand-ins for game data, used by the tests.

The real converter reads Miku's skeleton binding from the user's game.  For
tests we fabricate a plausible DIVA-like reference: bones named like the CMN
skeleton, an A-pose bind, X axes along the bones, a couple of expression
(helper) bones and face bones.
"""

from __future__ import annotations

import numpy as np

from vroid2diva.formats.objset import (BoneInfo, ExpressionBlock, Material, Object, ObjectSet, Skin,
                                       to_diva_matrix)
from vroid2diva.convert.retarget import axis_angle


def _frame(pos, x_dir=(1, 0, 0)):
    x = np.asarray(x_dir, dtype=float)
    x /= np.linalg.norm(x)
    up = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y = np.cross(up, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, pos
    return m


def make_reference_objset() -> ObjectSet:
    """A body+head object whose skin looks like a DIVA module's."""
    bones = {}
    h = 1.45  # head height
    bones["kl_kosi_etc_wj"] = _frame([0, 0.92, 0], [0, 1, 0])
    bones["kl_mune_b_wj"] = _frame([0, 1.12, 0], [0, 1, 0])
    bones["kl_kubi"] = _frame([0, 1.32, 0], [0, 1, 0])
    bones["j_kao_wj"] = _frame([0, h, 0], [0, 1, 0])
    bones["kl_eye_l_wj"] = _frame([0.032, h + 0.07, 0.07])
    bones["kl_eye_r_wj"] = _frame([-0.032, h + 0.07, 0.07])
    bones["kl_ago_wj"] = _frame([0, h + 0.0, 0.02], [0, -0.3, 1])
    for d, s in (("l", 1), ("r", -1)):
        down = np.array([s * np.cos(np.radians(40)), -np.sin(np.radians(40)), 0])
        sh = np.array([s * 0.07, 1.28, 0])
        up_arm = np.array([s * 0.17, 1.27, 0])
        elbow = up_arm + down * 0.24
        wrist = elbow + down * 0.22
        bones["kl_waki_%s_wj" % d] = _frame(sh, up_arm - sh)
        bones["j_kata_%s_wj_cu" % d] = _frame(up_arm, down)
        bones["n_skata_%s_wj_cd_ex" % d] = _frame(up_arm, down)
        bones["j_ude_%s_wj" % d] = _frame(elbow, down)
        bones["kl_te_%s_wj" % d] = _frame(wrist, down)
        side = np.cross(down, [0, 0, 1])
        for i, f in enumerate(("hito", "naka", "kusu", "ko")):
            base = wrist + down * 0.08 + side * (0.03 - 0.02 * i) * s
            bones["nl_%s_%s_wj" % (f, d)] = _frame(base, down)
            bones["nl_%s_b_%s_wj" % (f, d)] = _frame(base + down * 0.03, down)
            bones["nl_%s_c_%s_wj" % (f, d)] = _frame(base + down * 0.055, down)
        thumb = wrist + np.array([0, 0, 0.03])
        bones["nl_oya_%s_wj" % d] = _frame(thumb, down)
        bones["nl_oya_b_%s_wj" % d] = _frame(thumb + down * 0.03, down)
        bones["nl_oya_c_%s_wj" % d] = _frame(thumb + down * 0.055, down)
        hip = np.array([s * 0.08, 0.85, 0])
        bones["j_momo_%s_wj" % d] = _frame(hip, [0, -1, 0])
        bones["j_sune_%s_wj" % d] = _frame(hip + [0, -0.4, 0], [0, -1, 0])
        bones["kl_asi_%s_wj_co" % d] = _frame(hip + [0, -0.78, 0], [0, -0.3, 1])
        bones["kl_toe_%s_wj" % d] = _frame(hip + [0, -0.83, 0.1], [0, 0, 1])
        bones["kl_mabu_u_%s_wj" % d] = _frame(bones["kl_eye_%s_wj" % d][:3, 3], [0, 0, 1])
        bones["kl_mabu_d_%s_wj" % d] = _frame(bones["kl_eye_%s_wj" % d][:3, 3], [0, 0, 1])
        bones["tl_kuti_%s_wj" % d] = _frame([s * 0.022, h + 0.0, 0.085])
        bones["tl_mayu_%s_wj" % d] = _frame([s * 0.035, h + 0.11, 0.08])
    bones["tl_kuti_u_wj"] = _frame([0, h + 0.008, 0.09])
    bones["tl_kuti_d_wj"] = _frame([0, h - 0.008, 0.09])

    skin = Skin()
    for i, (name, m) in enumerate(bones.items()):
        is_ex = name.endswith("_ex")
        skin.bones.append(BoneInfo(name, id=0 if is_ex else i, ibm=to_diva_matrix(np.linalg.inv(m)), is_ex=is_ex))
    for d in ("l", "r"):
        skin.blocks.append(ExpressionBlock(parent_name="j_kata_%s_wj_cu" % d, name="n_skata_%s_wj_cd_ex" % d,
                                           expressions=["= 0 n_skata_%s_wj_cd_ex j_kata_%s_wj_cu" % (d, d)]))
    mats = [Material(shader_name="SKIN", name="skin"), Material(shader_name="CLOTH", name="cloth")]
    obj = Object("MIKITM001_BODY", 1, [], mats, skin)
    return ObjectSet([obj])


ITEM_TABLE = """cos.0.id=0
cos.0.item.0=1
cos.0.item.1=500
cos.0.item.length=2
cos.length=1
item.0.attr=1
item.0.des_id=0
item.0.exclusion=0
item.0.flag=0
item.0.name=BODY
item.0.no=1
item.0.objset.0=MIKITM001
item.0.objset.length=1
item.0.data.obj.0.rpk=1
item.0.data.obj.0.uid=MIKITM001_BODY
item.0.data.obj.length=1
item.0.org_itm=0
item.0.point=0
item.0.sub_id=10
item.0.type=0
item.1.attr=2049
item.1.des_id=0
item.1.name=HEAD
item.1.no=500
item.1.objset.0=MIKITM500
item.1.objset.length=1
item.1.org_itm=0
item.1.sub_id=24
item.1.type=0
item.length=2
"""
