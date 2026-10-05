"""Read-only access to Mega Mix+ CPK archives (diva_main.cpk etc.).

Mega Mix+ CPKs store every file AES-256-CBC encrypted with a fixed key.  Only
the table of contents is parsed; files are decrypted on demand.  Based on the
MM+ specific reader in MikuMikuLibrary.
"""

from __future__ import annotations

import struct
from typing import Dict, List, Tuple

from ..util.binary import BinaryReader, align_up

_KEY = bytes([0xCF, 0x53, 0xBF, 0x9C, 0x37, 0x67, 0xAF, 0xB0, 0x35, 0x54, 0x4E, 0xB9, 0x96, 0xAA, 0x24, 0x39,
              0x26, 0x5D, 0x40, 0x89, 0x7E, 0xD0, 0x1C, 0x3A, 0x6B, 0xA6, 0x5D, 0xD5, 0xFD, 0x6C, 0x19, 0xA3])
_IV = bytes([0xC2, 0x55, 0xFD, 0x73, 0xD8, 0x30, 0xFA, 0xEF, 0xD5, 0x32, 0x08, 0x54, 0xA2, 0x26, 0x44, 0x14])

_TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "Q", 7: "q", 8: "f", 9: "d"}


def _unmask(data: bytearray) -> bytearray:
    current = 25951
    for i in range(len(data)):
        data[i] ^= current & 0xFF
        current = (current * 16661) & 0xFFFFFFFF
    return data


def read_utf_table(data: bytes) -> List[dict]:
    data = bytearray(data)
    if data[:4] != b"@UTF":
        _unmask(data)
    r = BinaryReader(bytes(data), ">")
    if r.read(4) != b"@UTF":
        raise ValueError("Invalid @UTF table")
    r.u32()  # length
    r.push_base()
    r.u16()  # encoding
    rows_offset = r.u16()
    string_pool = r.u32()
    data_pool = r.u32()
    r.u32()  # table name
    field_count = r.u16()
    row_stride = r.u16()
    row_count = r.u32()
    base = r.base

    def read_string() -> str:
        off = r.u32()
        return r.cstring_at(base + string_pool + off) if off else ""

    def read_value(kind: int):
        if kind in _TYPES:
            fmt = _TYPES[kind]
            return struct.unpack(">" + fmt, r.read(struct.calcsize(fmt)))[0]
        if kind == 0xA:
            return read_string()
        if kind == 0xB:
            off = r.u32()
            size = r.u32()
            start = base + data_pool + off
            return bytes(r.data[start:start + size])
        raise ValueError("Unknown @UTF field type %d" % kind)

    fields: List[Tuple[str, int, bool, object, bool]] = []
    for _ in range(field_count):
        flags = r.u8()
        kind = flags & 0xF
        name = read_string() if flags & 0x10 else ""
        has_default = bool(flags & 0x20)
        default = read_value(kind) if has_default else None
        fields.append((name, kind, has_default, default, bool(flags & 0x40)))

    rows = []
    for i in range(row_count):
        r.pos = base + rows_offset + i * row_stride
        row = {}
        for name, kind, has_default, default, valid in fields:
            if has_default:
                row[name] = default
            elif valid:
                row[name] = read_value(kind)
            else:
                row[name] = None
        rows.append(row)
    return rows


def _read_chunk(f, pos: int, expected: bytes) -> List[dict]:
    f.seek(pos)
    head = f.read(16)
    if head[:4] != expected:
        raise ValueError("Expected %r chunk" % expected)
    length = struct.unpack("<I", head[8:12])[0]
    return read_utf_table(f.read(length))


class CpkArchive:
    def __init__(self, path: str):
        self.path = path
        self.entries: Dict[str, Tuple[int, int]] = {}
        with open(path, "rb") as f:
            cpk = _read_chunk(f, 0, b"CPK ")[0]
            toc_offset = int(cpk["TocOffset"])
            content_offset = int(cpk["ContentOffset"])
            align = int(cpk.get("Align") or 1)
            for row in _read_chunk(f, toc_offset, b"TOC "):
                dir_name = row.get("DirName") or ""
                name = ((dir_name + "/") if dir_name else "") + (row.get("FileName") or "")
                pos = int(row["FileOffset"])
                pos += content_offset if content_offset < toc_offset else toc_offset
                pos = align_up(pos, align)
                self.entries[name.replace("\\", "/").lower()] = (pos, int(row["FileSize"]))

    def names(self):
        return self.entries.keys()

    def __contains__(self, name: str) -> bool:
        return name.lower() in self.entries

    def read(self, name: str) -> bytes:
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        pos, size = self.entries[name.lower()]
        with open(self.path, "rb") as f:
            f.seek(pos)
            raw = f.read(size)
        dec = Cipher(algorithms.AES(_KEY), modes.CBC(_IV)).decryptor()
        plain = dec.update(raw[: len(raw) - len(raw) % 16]) + dec.finalize()
        try:
            unpadder = padding.PKCS7(128).unpadder()
            return unpadder.update(plain) + unpadder.finalize()
        except ValueError:
            return plain
