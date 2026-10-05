"""Static mapping tables between VRM humanoid bones and DIVA's CMN skeleton.

Each VRM bone lists DIVA candidates in order of preference; the first one that
exists in the reference (Miku's own module skin) is used.
"""

HUMANOID_TO_DIVA = {
    "hips": ["kl_kosi_etc_wj", "kl_kosi_y"],
    "spine": ["n_hara_b_wj_ex", "kl_kosi_etc_wj"],
    "chest": ["n_hara_c_wj_ex", "kl_mune_b_wj"],
    "upperChest": ["kl_mune_b_wj"],
    "neck": ["n_kubi_wj_ex", "kl_kubi"],
    "head": ["j_kao_wj", "kl_kao"],
    "leftEye": ["kl_eye_l_wj"],
    "rightEye": ["kl_eye_r_wj"],
    "jaw": ["kl_ago_wj"],
}

for _s, _d in (("left", "l"), ("right", "r")):
    HUMANOID_TO_DIVA.update({
        _s + "Shoulder": ["kl_waki_%s_wj" % _d],
        _s + "UpperArm": ["n_skata_%s_wj_cd_ex" % _d, "j_kata_%s_wj_cu" % _d],
        _s + "LowerArm": ["n_sude_%s_wj_ex" % _d, "j_ude_%s_wj" % _d],
        _s + "Hand": ["kl_te_%s_wj" % _d],
        _s + "UpperLeg": ["n_momo_a_%s_wj_cd_ex" % _d, "j_momo_%s_wj" % _d],
        _s + "LowerLeg": ["j_sune_%s_wj" % _d],
        _s + "Foot": ["kl_asi_%s_wj_co" % _d],
        _s + "Toes": ["kl_toe_%s_wj" % _d],
        _s + "ThumbMetacarpal": ["nl_oya_%s_wj" % _d],
        _s + "ThumbProximal": ["nl_oya_b_%s_wj" % _d],
        _s + "ThumbDistal": ["nl_oya_c_%s_wj" % _d],
    })
    for _vrm, _diva in (("Index", "hito"), ("Middle", "naka"), ("Ring", "kusu"), ("Little", "ko")):
        HUMANOID_TO_DIVA[_s + _vrm + "Proximal"] = ["nl_%s_%s_wj" % (_diva, _d)]
        HUMANOID_TO_DIVA[_s + _vrm + "Intermediate"] = ["nl_%s_b_%s_wj" % (_diva, _d)]
        HUMANOID_TO_DIVA[_s + _vrm + "Distal"] = ["nl_%s_c_%s_wj" % (_diva, _d)]

# Bone pairs whose direction is matched to the reference before binding
# (parent first).  Only limbs are re-posed; the torso is left as modelled.
REPOSE_CHAINS = []
for _s in ("left", "right"):
    REPOSE_CHAINS += [
        (_s + "UpperArm", _s + "LowerArm"),
        (_s + "LowerArm", _s + "Hand"),
        (_s + "Hand", _s + "MiddleProximal"),
        (_s + "UpperLeg", _s + "LowerLeg"),
        (_s + "LowerLeg", _s + "Foot"),
        (_s + "ThumbMetacarpal", _s + "ThumbProximal"),
        (_s + "ThumbProximal", _s + "ThumbDistal"),
    ]
    for _f in ("Index", "Middle", "Ring", "Little"):
        REPOSE_CHAINS += [
            (_s + _f + "Proximal", _s + _f + "Intermediate"),
            (_s + _f + "Intermediate", _s + _f + "Distal"),
        ]

# Secondary axis used to fix the roll of the hands (palm orientation).
HAND_ROLL = {
    "leftHand": ("leftIndexProximal", "leftLittleProximal"),
    "rightHand": ("rightIndexProximal", "rightLittleProximal"),
}
