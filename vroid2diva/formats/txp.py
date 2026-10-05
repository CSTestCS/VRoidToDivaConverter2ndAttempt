"""Texture sets (``*_tex.bin``, TXP format)."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List

from ..util.binary import BinaryReader, BinaryWriter

FMT_RGBA8 = 2
FMT_DXT1 = 6
FMT_DXT5 = 9


@dataclass
class SubTexture:
    width: int
    height: int
    format: int
    data: bytes


@dataclass
class Texture:
    name: str = ""
    id: int = 0
    mips: List[SubTexture] = field(default_factory=list)


def _encode_level(img, fmt: int) -> bytes:
    if fmt == FMT_RGBA8:
        return img.convert("RGBA").tobytes()
    buf = io.BytesIO()
    img.convert("RGBA").save(buf, format="DDS", pixel_format="DXT1" if fmt == FMT_DXT1 else "DXT5")
    raw = buf.getvalue()
    header = 128 + (20 if raw[84:88] == b"DX10" else 0)
    return raw[header:]


def encode_texture(img, name: str, tex_id: int, has_alpha: bool, compress: bool = True,
                   max_size: int = 2048) -> Texture:
    """Encode a Pillow image (with a full mip chain) into a DIVA texture."""
    from PIL import Image

    img = img.convert("RGBA")
    w, h = img.size
    scale = min(1.0, max_size / float(max(w, h)))
    # Block compression and mip maps need power-of-two sizes.
    def pot(v):
        p = 1
        while p * 2 <= v:
            p *= 2
        return max(4, p)

    nw, nh = pot(int(w * scale)), pot(int(h * scale))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)

    fmt = (FMT_DXT5 if has_alpha else FMT_DXT1) if compress else FMT_RGBA8
    tex = Texture(name=name, id=tex_id)
    level = img
    while True:
        lw, lh = level.size
        try:
            data = _encode_level(level, fmt)
        except Exception:
            if fmt == FMT_RGBA8:
                raise
            # Older Pillow without BCn encoding: fall back to uncompressed.
            return encode_texture(img, name, tex_id, has_alpha, compress=False, max_size=max_size)
        tex.mips.append(SubTexture(lw, lh, fmt, data))
        if lw <= 4 or lh <= 4:
            break
        level = level.resize((max(1, lw // 2), max(1, lh // 2)), Image.BOX)
    return tex


def write_texture_set(textures: List[Texture]) -> bytes:
    w = BinaryWriter()
    w.push_base()
    w.u32(0x03505854)
    w.u32(len(textures))
    w.u32(len(textures) | 0x01010100)

    for tex in textures:
        def body(tex=tex):
            w.push_base()
            w.u32(0x04505854)
            w.u32(len(tex.mips))
            w.u32(len(tex.mips) | (1 << 8) | 0x01010000)
            for i, mip in enumerate(tex.mips):
                def sub(mip=mip, i=i):
                    w.u32(0x02505854)
                    w.i32(mip.width)
                    w.i32(mip.height)
                    w.i32(mip.format)
                    w.i32(i)
                    w.i32(len(mip.data))
                    w.write(mip.data)
                w.offset(sub, 4)
            w.pop_base()
        w.offset(body, 4)
    w.pop_base()
    return w.finish()


def read_texture_set(data: bytes) -> List[Texture]:
    r = BinaryReader(data)
    if r.u32() != 0x03505854:
        raise ValueError("Not a TXP texture set")
    count = r.u32()
    r.u32()
    offs = [r.u32() for _ in range(count)]
    out = []
    for off in offs:
        with r.at(off):
            r.u32()
            sub_count = r.u32()
            r.u32()
            sub_offs = [r.u32() + off for _ in range(sub_count)]
            tex = Texture()
            for so in sub_offs:
                with r.at(so):
                    r.u32()
                    wdt, hgt, fmt = r.i32(), r.i32(), r.i32()
                    r.i32()
                    size = r.i32()
                    tex.mips.append(SubTexture(wdt, hgt, fmt, r.read(size)))
            out.append(tex)
    return out
