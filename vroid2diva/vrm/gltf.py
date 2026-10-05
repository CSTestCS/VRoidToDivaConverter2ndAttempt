"""Minimal glTF 2.0 / GLB reader built on numpy."""

from __future__ import annotations

import base64
import json
import struct
from typing import Optional

import numpy as np

_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class Gltf:
    def __init__(self, json_data: dict, bin_chunk: Optional[bytes]):
        self.json = json_data
        self.bin = bin_chunk
        self._buffers = {}

    @staticmethod
    def load(path: str) -> "Gltf":
        with open(path, "rb") as f:
            data = f.read()
        return Gltf.from_bytes(data)

    @staticmethod
    def from_bytes(data: bytes) -> "Gltf":
        if data[:4] != b"glTF":
            raise ValueError("This is not a .vrm/.glb file (missing glTF header).")
        length = struct.unpack("<I", data[8:12])[0]
        pos = 12
        js = None
        bin_chunk = None
        while pos < length:
            clen, ctype = struct.unpack("<II", data[pos:pos + 8])
            chunk = data[pos + 8:pos + 8 + clen]
            if ctype == 0x4E4F534A:
                js = json.loads(chunk.decode("utf-8"))
            elif ctype == 0x004E4942:
                bin_chunk = chunk
            pos += 8 + clen
        if js is None:
            raise ValueError("GLB file has no JSON chunk.")
        return Gltf(js, bin_chunk)

    def buffer(self, index: int) -> bytes:
        if index in self._buffers:
            return self._buffers[index]
        buf = self.json["buffers"][index]
        uri = buf.get("uri")
        if uri is None:
            data = self.bin
        elif uri.startswith("data:"):
            data = base64.b64decode(uri.split(",", 1)[1])
        else:
            raise ValueError("External buffers are not supported (use a .vrm/.glb file).")
        self._buffers[index] = data
        return data

    def buffer_view(self, index: int) -> bytes:
        bv = self.json["bufferViews"][index]
        data = self.buffer(bv["buffer"])
        start = bv.get("byteOffset", 0)
        return data[start:start + bv["byteLength"]]

    def accessor(self, index: int) -> np.ndarray:
        acc = self.json["accessors"][index]
        dtype = np.dtype(_COMPONENT[acc["componentType"]]).newbyteorder("<")
        comps = _COUNT[acc["type"]]
        count = acc["count"]
        out = np.zeros((count, comps), dtype=dtype)
        if "bufferView" in acc:
            bv = self.json["bufferViews"][acc["bufferView"]]
            raw = self.buffer_view(acc["bufferView"])
            offset = acc.get("byteOffset", 0)
            item = dtype.itemsize * comps
            stride = bv.get("byteStride") or item
            if stride == item:
                out = np.frombuffer(raw, dtype=dtype, count=count * comps, offset=offset).reshape(count, comps).copy()
            else:
                arr = np.lib.stride_tricks.as_strided(
                    np.frombuffer(raw, dtype=np.uint8, offset=offset),
                    shape=(count, item), strides=(stride, 1))
                out = np.frombuffer(arr.copy().tobytes(), dtype=dtype).reshape(count, comps).copy()
        sparse = acc.get("sparse")
        if sparse:
            idx_info = sparse["indices"]
            idx_dtype = np.dtype(_COMPONENT[idx_info["componentType"]]).newbyteorder("<")
            idx = np.frombuffer(self.buffer_view(idx_info["bufferView"]), dtype=idx_dtype,
                                count=sparse["count"], offset=idx_info.get("byteOffset", 0))
            val_info = sparse["values"]
            vals = np.frombuffer(self.buffer_view(val_info["bufferView"]), dtype=dtype,
                                 count=sparse["count"] * comps, offset=val_info.get("byteOffset", 0))
            out = out.copy()
            out[idx.astype(np.int64)] = vals.reshape(-1, comps)
        if acc.get("normalized") and dtype.kind in "iu":
            maxv = float(np.iinfo(dtype).max)
            out = np.maximum(out.astype(np.float32) / maxv, -1.0)
        return out

    def image_bytes(self, index: int) -> bytes:
        img = self.json["images"][index]
        if "bufferView" in img:
            return self.buffer_view(img["bufferView"])
        uri = img.get("uri", "")
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        raise ValueError("External images are not supported.")


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    t = node.get("translation", [0, 0, 0])
    r = node.get("rotation", [0, 0, 0, 1])
    s = node.get("scale", [1, 1, 1])
    m[:3, :3] = quat_to_matrix(r) @ np.diag(s)
    m[:3, 3] = t
    return m
