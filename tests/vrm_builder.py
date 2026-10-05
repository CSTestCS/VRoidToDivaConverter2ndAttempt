"""Generate a small but complete VRM 0.x file for tests.

T-pose humanoid facing -Z (like UniVRM 0.x exports), one skinned mesh made of
small boxes around the joints, a hair spring chain, a face patch with "A" and
blink blend shapes, and a texture.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np

BONES = [  # name, parent, position (left side at -X like VRM 0.x)
    ("hips", None, (0, 0.9, 0)), ("spine", "hips", (0, 1.0, 0)), ("chest", "spine", (0, 1.1, 0)),
    ("neck", "chest", (0, 1.3, 0)), ("head", "neck", (0, 1.4, 0)),
    ("leftEye", "head", (-0.03, 1.47, -0.06)), ("rightEye", "head", (0.03, 1.47, -0.06)),
]
for side, sx in (("left", -1), ("right", 1)):
    BONES += [
        (side + "Shoulder", "chest", (sx * 0.05, 1.27, 0)), (side + "UpperArm", side + "Shoulder", (sx * 0.15, 1.27, 0)),
        (side + "LowerArm", side + "UpperArm", (sx * 0.4, 1.27, 0)), (side + "Hand", side + "LowerArm", (sx * 0.62, 1.27, 0)),
        (side + "MiddleProximal", side + "Hand", (sx * 0.7, 1.27, 0)),
        (side + "IndexProximal", side + "Hand", (sx * 0.7, 1.27, -0.02)),
        (side + "LittleProximal", side + "Hand", (sx * 0.7, 1.27, 0.02)),
        (side + "UpperLeg", "hips", (sx * 0.08, 0.86, 0)), (side + "LowerLeg", side + "UpperLeg", (sx * 0.08, 0.47, 0)),
        (side + "Foot", side + "LowerLeg", (sx * 0.08, 0.08, 0)),
    ]
BONES += [("hair1", "head", (0, 1.5, 0.08)), ("hair2", "hair1", (0, 1.4, 0.1)), ("hair3", "hair2", (0, 1.3, 0.1))]


def _box(center, size=0.02):
    c = np.array(center, dtype=np.float32)
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32) * size
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return corners + c, np.array(faces, dtype=np.uint32)


def make_test_vrm(path: str) -> str:
    from PIL import Image

    names = [b[0] for b in BONES]
    index = {n: i for i, n in enumerate(names)}
    world = {n: np.array(p, dtype=np.float64) for n, _, p in BONES}

    pos, idx, joints, weights = [], [], [], []
    for name, parent, p in BONES:
        if name in ("leftEye", "rightEye"):
            continue
        v, f = _box(p)
        idx.append(f + sum(len(x) for x in pos))
        pos.append(v)
        joints.append(np.tile([index[name], 0, 0, 0], (len(v), 1)))
        weights.append(np.tile([1.0, 0, 0, 0], (len(v), 1)))
    # Face patch in front of the head (faces -Z).
    gx, gy = np.meshgrid(np.linspace(-0.06, 0.06, 7), np.linspace(1.36, 1.52, 9))
    face = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -0.08)], axis=1).astype(np.float32)
    base = sum(len(x) for x in pos)
    tris = []
    for r in range(8):
        for c in range(6):
            a = r * 7 + c
            tris += [(a, a + 1, a + 8), (a, a + 8, a + 7)]
    idx.append(np.array(tris, dtype=np.uint32) + base)
    pos.append(face)
    joints.append(np.tile([index["head"], 0, 0, 0], (len(face), 1)))
    weights.append(np.tile([1.0, 0, 0, 0], (len(face), 1)))
    positions = np.concatenate(pos).astype(np.float32)
    indices = np.concatenate(idx).astype(np.uint32).ravel()
    joints = np.concatenate(joints).astype(np.uint16)
    weights = np.concatenate(weights).astype(np.float32)
    normals = np.tile([0, 0, -1], (len(positions), 1)).astype(np.float32)
    uvs = np.zeros((len(positions), 2), dtype=np.float32)
    morph_a = np.zeros_like(positions)
    morph_blink = np.zeros_like(positions)
    fsl = slice(base, base + len(face))
    below = face[:, 1] < 1.42
    morph_a[fsl][below, 1] = -(1.42 - face[below, 1]) * 0.5
    near_eye = (np.abs(face[:, 1] - 1.48) < 0.02)
    morph_blink[fsl][near_eye, 1] = -0.012

    ibms = []
    for n in names:
        m = np.eye(4)
        m[:3, 3] = -world[n]
        ibms.append(m.T.ravel())
    ibm = np.array(ibms, dtype=np.float32)

    png = io.BytesIO()
    Image.new("RGBA", (16, 16), (200, 150, 150, 255)).save(png, "PNG")

    blobs, views, accessors = [], [], []

    def add(arr, typ, comp, target=None, minmax=False):
        data = arr.tobytes()
        off = sum(len(b) for b in blobs)
        blobs.append(data + b"\0" * (-len(data) % 4))
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data)})
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": len(arr), "type": typ}
        if minmax:
            acc["min"] = arr.min(axis=0).tolist()
            acc["max"] = arr.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    a_pos = add(positions, "VEC3", 5126, minmax=True)
    a_nrm = add(normals, "VEC3", 5126)
    a_uv = add(uvs, "VEC2", 5126)
    a_j = add(joints, "VEC4", 5123)
    a_w = add(weights, "VEC4", 5126)
    a_i = add(indices, "SCALAR", 5125)
    a_ma = add(morph_a, "VEC3", 5126, minmax=True)
    a_mb = add(morph_blink, "VEC3", 5126, minmax=True)
    a_ibm = add(ibm, "MAT4", 5126)
    off = sum(len(b) for b in blobs)
    img = png.getvalue()
    blobs.append(img + b"\0" * (-len(img) % 4))
    views.append({"buffer": 0, "byteOffset": off, "byteLength": len(img)})

    nodes = []
    for n, parent, p in BONES:
        pp = world[parent] if parent else np.zeros(3)
        nodes.append({"name": n, "translation": (world[n] - pp).tolist(),
                      "children": [index[c] for c, par, _ in BONES if par == n]})
    nodes.append({"name": "Body", "mesh": 0, "skin": 0})
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0, "scenes": [{"nodes": [0, len(nodes) - 1]}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "TEXCOORD_0": a_uv,
                                                   "JOINTS_0": a_j, "WEIGHTS_0": a_w},
                                    "indices": a_i, "material": 0,
                                    "targets": [{"POSITION": a_ma}, {"POSITION": a_mb}]}]}],
        "skins": [{"joints": list(range(len(BONES))), "inverseBindMatrices": a_ibm}],
        "materials": [{"name": "Body_skin", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}],
        "images": [{"bufferView": len(views) - 1, "mimeType": "image/png"}],
        "accessors": accessors, "bufferViews": views,
        "buffers": [{"byteLength": sum(len(b) for b in blobs)}],
        "extensionsUsed": ["VRM"],
        "extensions": {"VRM": {
            "meta": {"title": "TestChan"},
            "humanoid": {"humanBones": [{"bone": n, "node": index[n]} for n, _, _ in BONES if not n.startswith("hair")]},
            "blendShapeMaster": {"blendShapeGroups": [
                {"name": "A", "presetName": "a", "binds": [{"mesh": 0, "index": 0, "weight": 100}]},
                {"name": "Blink", "presetName": "blink", "binds": [{"mesh": 0, "index": 1, "weight": 100}]}]},
            "secondaryAnimation": {
                "boneGroups": [{"stiffiness": 1.0, "gravityPower": 0, "gravityDir": {"x": 0, "y": -1, "z": 0},
                                "dragForce": 0.4, "center": -1, "hitRadius": 0.02, "bones": [index["hair1"]],
                                "colliderGroups": [0]}],
                "colliderGroups": [{"node": index["head"], "colliders": [{"offset": {"x": 0, "y": 0.1, "z": 0},
                                                                          "radius": 0.1}]}]},
            "materialProperties": [],
        }},
    }
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    binary = b"".join(blobs)
    out = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(js) + 8 + len(binary))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binary), 0x004E4942) + binary
    with open(path, "wb") as f:
        f.write(out)
    return path
