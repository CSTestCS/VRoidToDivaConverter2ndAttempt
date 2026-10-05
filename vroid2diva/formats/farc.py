"""FARC archives.

Reading supports the three variants found in Mega Mix+ data:

* ``FArc`` - plain
* ``FArC`` - gzip compressed entries
* ``FARC`` - Future Tone style (AES-CBC encrypted header and entries)

Writing always produces ``FArC`` (what MikuMikuModel writes and what mods use).
"""

from __future__ import annotations

import gzip
import io
import struct
import zlib
from typing import Dict

from ..util.binary import align_up

_FT_KEY = bytes([0x13, 0x72, 0xD5, 0x7B, 0x6E, 0x9E, 0x31, 0xEB,
                 0xA2, 0x39, 0xB8, 0x3C, 0x15, 0x57, 0xC6, 0xBB])
_DT_KEY = b"project_diva.bin"


def _aes_cbc_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    data = data[: len(data) - len(data) % 16]
    dec = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    return dec.update(data) + dec.finalize()


def _aes_ecb_decrypt(key: bytes, data: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    data = data[: len(data) - len(data) % 16]
    dec = Cipher(algorithms.AES(key), modes.ECB()).decryptor()
    return dec.update(data) + dec.finalize()


def _gunzip(data: bytes) -> bytes:
    try:
        return gzip.decompress(data)
    except (OSError, EOFError, zlib.error):
        # Some archives have trailing garbage after the gzip stream.
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        return d.decompress(data)


def read_farc(data: bytes) -> Dict[str, bytes]:
    """Return a ``{name: contents}`` mapping."""
    sig = data[:4]
    header_size = struct.unpack(">I", data[4:8])[0] + 8
    entries: Dict[str, bytes] = {}

    def cstr(buf, pos):
        end = buf.index(b"\0", pos)
        return buf[pos:end].decode("utf-8"), end + 1

    if sig == b"FArc" or sig == b"FArC":
        pos = 12
        while pos < header_size:
            name, pos = cstr(data, pos)
            if sig == b"FArc":
                off, size = struct.unpack(">II", data[pos:pos + 8])
                pos += 8
                entries[name] = bytes(data[off:off + size])
            else:
                off, csize, usize = struct.unpack(">III", data[pos:pos + 12])
                pos += 12
                raw = bytes(data[off:off + csize])
                entries[name] = _gunzip(raw)[:usize] if usize else raw
        return entries

    if sig != b"FARC":
        raise ValueError("Not a FARC archive")

    flags, _pad, alignment = struct.unpack(">III", data[8:20])
    encrypted = bool(flags & 4)
    ft = encrypted and (alignment & (alignment - 1)) != 0
    if ft:
        iv = data[0x10:0x20]
        header = _aes_cbc_decrypt(_FT_KEY, iv, data[0x20:header_size])
        inner_size, alignment, count, entry_size = struct.unpack(">IIII", header[:16])
        pos = 16 + (inner_size - 0x10)
        entry_pad = entry_size - 0x10
        for _ in range(count):
            name, pos = cstr(header, pos)
            off, csize, usize, eflags = struct.unpack(">IIII", header[pos:pos + 16])
            pos += 16 + entry_pad
            comp = bool(eflags & 2)
            enc = bool(eflags & 4)
            raw = bytes(data[off:off + align_up(csize if comp else usize, 16) + (16 if enc else 0)])
            if enc:
                raw = _aes_cbc_decrypt(_FT_KEY, raw[:16], raw[16:])
            if comp and csize != usize:
                raw = _gunzip(raw[:csize])
            entries[name] = raw[:usize]
        return entries

    # DT style FARC (ECB with project_diva.bin key)
    compressed = bool(flags & 2)
    entry_pad, header_pad = struct.unpack(">II", data[20:28]) if len(data) >= 28 else (0, 0)
    pos = 28 + header_pad
    while pos < header_size:
        name, pos = cstr(data, pos)
        off, csize, usize = struct.unpack(">III", data[pos:pos + 12])
        pos += 12 + entry_pad
        raw = bytes(data[off:off + align_up(csize if compressed else usize, 16)])
        if encrypted:
            raw = _aes_ecb_decrypt(_DT_KEY, raw)
        if compressed and csize != usize:
            raw = _gunzip(raw[:csize])
        entries[name] = raw[:usize]
    return entries


def write_farc(entries: Dict[str, bytes], alignment: int = 16, compress: bool = True) -> bytes:
    """Write a ``FArC`` (compressed) or ``FArc`` archive."""
    names = sorted(entries)
    payloads = {}
    for name in names:
        raw = entries[name]
        payloads[name] = gzip.compress(raw, mtime=0) if compress else raw

    header = io.BytesIO()
    header.write(struct.pack(">I", alignment))
    for name in names:
        header.write(name.encode("utf-8") + b"\0")
        header.write(b"\0" * (12 if compress else 8))
    header_size = 8 + len(header.getvalue())

    out = bytearray()
    out += b"FArC" if compress else b"FArc"
    out += struct.pack(">I", header_size - 8)
    out += header.getvalue()

    table_pos = 12
    for name in names:
        table_pos += len(name.encode("utf-8")) + 1
        out += b"\x78" * (align_up(len(out), alignment) - len(out))
        off = len(out)
        out += payloads[name]
        if compress:
            struct.pack_into(">III", out, table_pos, off, len(payloads[name]), len(entries[name]))
            table_pos += 12
        else:
            struct.pack_into(">II", out, table_pos, off, len(payloads[name]))
            table_pos += 8
    out += b"\x78" * (align_up(len(out), alignment) - len(out))
    return bytes(out)
