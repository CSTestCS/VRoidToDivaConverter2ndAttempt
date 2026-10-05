"""VRM (MToon / glTF) materials -> DIVA materials and textures."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..formats.objset import (MATERIAL_FLAG_COLOR, Material, MaterialTexture, SUBMESH_CAST_SHADOW,
                              SUBMESH_RECEIVE_SHADOW, SUBMESH_TRANSPARENT)
from ..formats.txp import Texture, encode_texture
from ..util.murmur import murmur_hash
from ..vrm.model import VrmModel
from .reference import Reference

SKIN_WORDS = re.compile(r"(body|face|skin|mouth|_hada|hada_)", re.I)
UNLIT_WORDS = re.compile(r"(eye|iris|highlight|brow|lash|line)", re.I)

# Shader flag bits (MikuMikuLibrary Material.ShaderFlags layout).
_LAMBERT, _PHONG, _PER_PIXEL = 1 << 4, 1 << 5, 1 << 6
_RECEIVE_SHADOW, _CAST_SHADOW = 1 << 18, 1 << 19


def _ascii(name: str, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return (s or fallback)[:60]


def _template(ref: Optional[Reference], shader: str) -> Optional[Material]:
    if ref is None:
        return None
    for m in ref.materials:
        if m.shader_name.upper() == shader:
            return m
    return None


def make_material(vm, index: int, tex_id: Optional[int], ref: Optional[Reference], style: str) -> Material:
    if style == "toon":
        shader = "SKIN" if SKIN_WORDS.search(vm.name) and not UNLIT_WORDS.search(vm.name) else "CLOTH"
    else:
        shader = "BLINN"
    tpl = _template(ref, shader)
    if tpl is not None:
        mat = tpl.clone()
        color_tpl = next((t for t in tpl.textures if t.type == 1), None)
        mat.textures = [MaterialTexture() for _ in range(8)]
        mat.shader_flags &= ~(0x3 << 8)  # no bump mapping without normal maps
    else:
        mat = Material(shader_name=shader)
        mat.shader_flags = _LAMBERT | _PHONG | _PER_PIXEL | _RECEIVE_SHADOW | _CAST_SHADOW
        color_tpl = None
    mat.flags = 0
    if tex_id is not None:
        t = MaterialTexture.color(tex_id)
        if color_tpl is not None:
            t.sampler_flags = color_tpl.sampler_flags
            t.texture_flags = color_tpl.texture_flags
            t.uv_matrix = color_tpl.uv_matrix
        t.texture_id = tex_id
        mat.textures[0] = t
        mat.flags |= MATERIAL_FLAG_COLOR
    r, g, b, a = [float(c) for c in vm.color]
    mat.diffuse = [r, g, b, a]
    mat.name = _ascii(vm.name, "mat%d" % index)
    mat.set_double_sided(vm.double_sided)
    mat.set_alpha(alpha_texture=vm.alpha_mode != "OPAQUE" and tex_id is not None,
                  punch_through=vm.alpha_mode == "MASK")
    return mat


def submesh_flags(vm) -> int:
    flags = SUBMESH_RECEIVE_SHADOW | SUBMESH_CAST_SHADOW
    if vm is not None and vm.alpha_mode == "BLEND":
        flags = SUBMESH_RECEIVE_SHADOW | SUBMESH_TRANSPARENT
    return flags


def build_textures(model: VrmModel, used_materials: List[int], prefix: str, compress: bool, max_size: int,
                   log=print) -> Tuple[List[Texture], Dict[int, int]]:
    """Encode every image used by ``used_materials``.  Returns textures and a
    map image index -> texture id."""
    textures: List[Texture] = []
    ids: Dict[int, int] = {}
    alpha_users = {}
    for mi in used_materials:
        if 0 <= mi < len(model.materials):
            vm = model.materials[mi]
            if vm.texture is not None:
                alpha_users[vm.texture] = alpha_users.get(vm.texture, False) or vm.alpha_mode != "OPAQUE"
    for n, (image_index, wants_alpha) in enumerate(sorted(alpha_users.items())):
        name = "%s_TEX%02d" % (prefix, n)
        tid = murmur_hash(name)
        try:
            img = model.image(image_index)
            img.load()
        except Exception as e:
            log("Warning: could not decode image %d (%s); using white." % (image_index, e))
            from PIL import Image
            img = Image.new("RGBA", (8, 8), (255, 255, 255, 255))
        rgba = img.convert("RGBA")
        has_alpha = wants_alpha and bool(np.asarray(rgba)[:, :, 3].min() < 250)
        textures.append(encode_texture(rgba, name, tid, has_alpha, compress, max_size))
        ids[image_index] = tid
    log("Textures: %d encoded (%s)." % (len(textures), "DXT compressed" if compress else "uncompressed"))
    return textures, ids
