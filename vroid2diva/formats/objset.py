"""Classic (Future Tone / Mega Mix+) object set format: ``*_obj.bin``.

Layout and semantics follow MikuMikuLibrary's ObjectSet/Object/Mesh/SubMesh/
Material/Skin implementation for the FT format (little endian, 32-bit offsets).

Matrices are stored the way the game stores them: row-major, row-vector
convention (rows are the basis vectors, translation in the last row).  Code
outside this module works with numpy column-vector matrices and converts via
``to_diva_matrix`` / ``from_diva_matrix``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..util.binary import BinaryReader, BinaryWriter

OBJSET_SIGNATURE = 0x5062500

# Vertex attribute bits (classic storage).
VA_POSITION = 1 << 0
VA_NORMAL = 1 << 1
VA_TANGENT = 1 << 2
VA_TEXCOORD0 = 1 << 4
VA_TEXCOORD1 = 1 << 5
VA_TEXCOORD2 = 1 << 6
VA_TEXCOORD3 = 1 << 7
VA_COLOR0 = 1 << 8
VA_COLOR1 = 1 << 9
VA_BLEND_WEIGHT = 1 << 10
VA_BLEND_INDICES = 1 << 11

PRIMITIVE_TRIANGLES = 4
INDEX_U16 = 1

SUBMESH_RECEIVE_SHADOW = 1 << 0
SUBMESH_CAST_SHADOW = 1 << 1
SUBMESH_TRANSPARENT = 1 << 2


def to_diva_matrix(m: np.ndarray) -> List[List[float]]:
    """numpy column-vector 4x4 -> DIVA row-vector rows."""
    return np.asarray(m, dtype=np.float64).T.tolist()


def from_diva_matrix(rows) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).T.copy()


def bounding_sphere(points: np.ndarray):
    if points is None or len(points) == 0:
        return (0.0, 0.0, 0.0), 0.0
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.sqrt(((points - center) ** 2).sum(axis=1)).max())
    return tuple(float(x) for x in center), radius


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

@dataclass
class MaterialTexture:
    sampler_flags: int = 0
    texture_id: int = 0xFFFFFFFF
    texture_flags: int = 0xF0
    extra_shader_name: str = ""
    weight: float = 1.0
    uv_matrix: List[List[float]] = field(default_factory=lambda: np.eye(4).tolist())

    @property
    def type(self) -> int:
        return self.texture_flags & 0xF

    @staticmethod
    def color(texture_id: int, repeat: bool = True) -> "MaterialTexture":
        t = MaterialTexture()
        t.texture_id = texture_id
        flags = 0
        if repeat:
            flags |= 1 | 2  # repeat U / V
        flags |= 7 << 5  # blend
        flags |= 2 << 17  # filter
        flags |= 2 << 20  # mipmap
        t.sampler_flags = flags
        t.texture_flags = 0xF0 | TEXTURE_TYPE_COLOR  # same as MikuMikuModel's importer
        return t


TEXTURE_TYPE_NONE = 0
TEXTURE_TYPE_COLOR = 1
TEXTURE_TYPE_NORMAL = 2
TEXTURE_TYPE_SPECULAR = 3

MATERIAL_FLAG_COLOR = 1 << 0
MATERIAL_FLAG_NORMAL = 1 << 8
MATERIAL_FLAG_SPECULAR = 1 << 7


@dataclass
class Material:
    flags: int = 0
    shader_name: str = "BLINN"
    shader_flags: int = 0x70
    textures: List[MaterialTexture] = field(default_factory=lambda: [MaterialTexture() for _ in range(8)])
    blend_flags: int = 0xA80
    diffuse: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    ambient: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    specular: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    emission: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    shininess: float = 20.0
    intensity: float = 0.0
    reserved_sphere: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    name: str = ""
    bump_depth: float = 1.0

    # Blend flag helpers (bit layout from MikuMikuLibrary).
    def _set_blend_bits(self, value: int, lo: int, hi: int) -> None:
        mask = ((1 << (hi - lo)) - 1) << lo
        self.blend_flags = (self.blend_flags & ~mask) | ((value << lo) & mask)

    def set_alpha(self, alpha_texture: bool, punch_through: bool) -> None:
        self._set_blend_bits(1 if alpha_texture else 0, 0, 1)
        self._set_blend_bits(1 if punch_through else 0, 2, 3)

    def set_double_sided(self, value: bool) -> None:
        self._set_blend_bits(1 if value else 0, 3, 4)

    def clone(self) -> "Material":
        return copy.deepcopy(self)


def _read_material(r: BinaryReader) -> Material:
    m = Material()
    r.u32()  # texture count
    m.flags = r.u32()
    m.shader_name = r.fixed_string(8)
    m.shader_flags = r.u32()
    m.textures = []
    for _ in range(8):
        t = MaterialTexture()
        t.sampler_flags = r.u32()
        t.texture_id = r.u32()
        t.texture_flags = r.u32()
        t.extra_shader_name = r.fixed_string(8)
        t.weight = r.f32()
        t.uv_matrix = r.matrix()
        r.read(32)
        m.textures.append(t)
    m.blend_flags = r.u32()
    m.diffuse = r.floats(4)
    m.ambient = r.floats(4)
    m.specular = r.floats(4)
    m.emission = r.floats(4)
    m.shininess = r.f32()
    m.intensity = r.f32()
    m.reserved_sphere = r.floats(4)
    m.name = r.fixed_string(64)
    m.bump_depth = r.f32()
    r.read(60)
    return m


def _write_material(w: BinaryWriter, m: Material) -> None:
    w.u32(sum(1 for t in m.textures if t.type != TEXTURE_TYPE_NONE))
    w.u32(m.flags)
    w.fixed_string(m.shader_name, 8)
    w.u32(m.shader_flags)
    for t in m.textures:
        w.u32(t.sampler_flags)
        w.u32(t.texture_id)
        w.u32(t.texture_flags)
        w.fixed_string(t.extra_shader_name, 8)
        w.f32(t.weight)
        w.matrix(t.uv_matrix)
        w.nulls(32)
    w.u32(m.blend_flags)
    w.floats(m.diffuse)
    w.floats(m.ambient)
    w.floats(m.specular)
    w.floats(m.emission)
    w.f32(m.shininess)
    w.f32(m.intensity)
    w.floats(m.reserved_sphere)
    w.fixed_string(m.name, 64)
    w.f32(m.bump_depth)
    w.nulls(60)


MATERIAL_SIZE = 0x4B0

# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------


@dataclass
class SubMesh:
    material_index: int = 0
    indices: np.ndarray = None  # uint32 triangle list
    bone_indices: Optional[List[int]] = None  # palette -> skin bone index
    flags: int = SUBMESH_RECEIVE_SHADOW | SUBMESH_CAST_SHADOW
    texcoord_indices: bytes = b"\0" * 8
    sphere: tuple = ((0.0, 0.0, 0.0), 0.0)


@dataclass
class Mesh:
    name: str = ""
    positions: np.ndarray = None
    normals: Optional[np.ndarray] = None
    tangents: Optional[np.ndarray] = None
    uv0: Optional[np.ndarray] = None
    uv1: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    blend_weights: Optional[np.ndarray] = None  # (n, 4) float
    blend_indices: Optional[np.ndarray] = None  # (n, 4) int, palette indices, -1 = none
    submeshes: List[SubMesh] = field(default_factory=list)
    flags: int = 0


# ---------------------------------------------------------------------------
# Skin and ex-data blocks
# ---------------------------------------------------------------------------


@dataclass
class BoneInfo:
    name: str
    id: int = 0xFFFFFFFF  # object bone index for motion bones; set automatically for ex bones
    ibm: List[List[float]] = field(default_factory=lambda: np.eye(4).tolist())  # DIVA layout
    is_ex: bool = False
    parent: Optional[str] = None


@dataclass
class NodeBlockBase:
    parent_name: Optional[str] = None
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    name: str = ""


@dataclass
class ExpressionBlock(NodeBlockBase):
    expressions: List[str] = field(default_factory=list)
    signature: str = "EXP"


@dataclass
class MotionBlock(NodeBlockBase):
    nodes: List[tuple] = field(default_factory=list)  # (name, DIVA matrix rows)
    signature: str = "MOT"


@dataclass
class ConstraintBlock(NodeBlockBase):
    constraint_type: str = "Orientation"
    coupling: int = 1
    source_name: Optional[str] = None
    data: List[tuple] = field(default_factory=list)  # ("s"|"i"|"f", value) tokens
    signature: str = "CNS"


@dataclass
class OsageNode:
    name: str
    length: float
    rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class OsageBlock(NodeBlockBase):
    external_name: str = ""  # root node name (also the skin-param key)
    nodes: List[OsageNode] = field(default_factory=list)
    signature: str = "OSG"


@dataclass
class Skin:
    bones: List[BoneInfo] = field(default_factory=list)
    blocks: list = field(default_factory=list)

    def bone_index(self, name: str) -> int:
        for i, b in enumerate(self.bones):
            if b.name == name:
                return i
        return -1

    def string_set(self) -> List[str]:
        strings: List[str] = []

        def add(s):
            if s and s not in strings:
                strings.append(s)

        for block in self.blocks:
            if isinstance(block, ConstraintBlock):
                add(block.name)
                add(block.source_name)
                for kind, value in block.data:
                    if kind == "s":
                        add(value)
            elif isinstance(block, ExpressionBlock):
                add(block.name)
            elif isinstance(block, MotionBlock):
                for name, _ in block.nodes:
                    add(name)
            elif isinstance(block, OsageBlock):
                add(block.external_name)
                for node in block.nodes:
                    add(node.name)
                add(block.name)
        for bone in self.bones:
            if bone.is_ex:
                add(bone.name)
        return strings


_CNS_LAYOUT = {
    # Token layouts for the constraint data that follows the common header.
    "Orientation": "fff",
    "Direction": "iffffffs" + "ffffff",
    "Position": "iffffffs" + "iifff" + "iifff",
    "Distance": "iffffffs" + "f" + "iifff" + "iifff",
}


def _read_node_block_header(r: BinaryReader, block: NodeBlockBase) -> None:
    block.parent_name = r.string_offset()
    block.position = r.floats(3)
    block.rotation = r.floats(3)
    block.scale = r.floats(3)


def _read_skin(r: BinaryReader) -> Skin:
    skin = Skin()
    bone_ids_off = r.offset()
    matrices_off = r.offset()
    names_off = r.offset()
    ex_off = r.offset()
    bone_count = r.i32()
    parents_off = r.offset()

    ids = []
    with r.at(bone_ids_off):
        for _ in range(bone_count):
            ids.append(r.u32())
    with r.at(matrices_off):
        mats = [r.matrix() for _ in range(bone_count)]
    with r.at(names_off):
        names = [r.string_offset() for _ in range(bone_count)]
    for i in range(bone_count):
        skin.bones.append(BoneInfo(name=names[i], id=ids[i], ibm=mats[i], is_ex=bool(ids[i] & 0x8000)))
    if parents_off:
        with r.at(parents_off):
            by_id = {b.id: b for b in skin.bones}
            for b in skin.bones:
                pid = r.u32()
                if pid != 0xFFFFFFFF and pid in by_id:
                    b.parent = by_id[pid].name

    if not ex_off:
        return skin

    with r.at(ex_off):
        r.i32()  # osage count
        osage_node_count = r.i32()
        r.u32()
        osage_nodes_off = r.offset()
        r.offset()  # osage names
        blocks_off = r.offset()
        string_count = r.i32()
        strings_off = r.offset()

    strings = []
    with r.at(strings_off):
        for _ in range(string_count):
            strings.append(r.string_offset())

    def sid(value: int) -> Optional[str]:
        return strings[value & 0x7FFF] if value & 0x8000 and (value & 0x7FFF) < len(strings) else None

    osage_nodes = []
    with r.at(osage_nodes_off):
        for _ in range(osage_node_count):
            name = sid(r.u32())
            length = r.f32()
            r.u32()
            osage_nodes.append(OsageNode(name, length))

    with r.at(blocks_off):
        while True:
            sig = r.string_offset()
            boff = r.offset()
            if not boff:
                break
            with r.at(boff):
                block = _read_block(r, sig, sid, osage_nodes)
            if block is not None:
                skin.blocks.append(block)
    return skin


def _read_block(r: BinaryReader, sig: str, sid, osage_nodes):
    if sig == "EXP":
        b = ExpressionBlock()
        _read_node_block_header(r, b)
        b.name = r.string_offset()
        count = r.i32()
        b.expressions = [r.string_offset() for _ in range(count)]
        return b
    if sig == "MOT":
        b = MotionBlock()
        _read_node_block_header(r, b)
        name_off = r.offset()
        count = r.i32()
        names_off = r.offset()
        mats_off = r.offset()
        b.name = r.cstring_at(name_off) if name_off else ""
        with r.at(names_off):
            names = [sid(r.u32()) for _ in range(count)]
        with r.at(mats_off):
            mats = [r.matrix() for _ in range(count)]
        b.nodes = list(zip(names, mats))
        return b
    if sig == "CNS":
        b = ConstraintBlock()
        _read_node_block_header(r, b)
        b.constraint_type = r.string_offset()
        b.name = r.string_offset()
        b.coupling = r.i32()
        b.source_name = r.string_offset()
        layout = _CNS_LAYOUT.get(b.constraint_type)
        if layout is None:
            return None
        for kind in layout:
            if kind == "s":
                b.data.append(("s", r.string_offset()))
            elif kind == "i":
                b.data.append(("i", r.i32()))
            else:
                b.data.append(("f", r.f32()))
        return b
    if sig == "OSG":
        b = OsageBlock()
        _read_node_block_header(r, b)
        start = r.i32()
        count = r.i32()
        b.external_name = sid(r.u32())
        b.name = sid(r.u32())
        rot_off = r.offset()
        b.nodes = [copy.copy(n) for n in osage_nodes[start:start + count]]
        if rot_off:
            with r.at(rot_off):
                for node in b.nodes:
                    r.u32()
                    node.length = r.f32()
                    node.rotation = r.floats(3)
        return b
    return None  # cloth and unknown blocks are not needed


def _write_skin(w: BinaryWriter, skin: Skin) -> None:
    strings = skin.string_set()
    index = {s: i for i, s in enumerate(strings)}

    def sid(s: Optional[str]) -> int:
        return 0x8000 | index[s] if s in index else 0

    for b in skin.bones:
        if b.is_ex:
            b.id = sid(b.name)
    by_name = {b.name: b for b in skin.bones}

    def bone_ids():
        for b in skin.bones:
            w.u32(b.id)

    def bone_mats():
        for b in skin.bones:
            w.matrix(b.ibm)

    def bone_names():
        for b in skin.bones:
            w.string_offset(b.name)
        w.u32(0)

    w.offset(bone_ids, 16, "center")
    w.offset(bone_mats, 16, "center")
    w.offset(bone_names, 16, "center")

    osage_blocks = [b for b in skin.blocks if isinstance(b, OsageBlock)]
    osage_nodes: List[OsageNode] = []
    starts = {}
    for b in osage_blocks:
        starts[id(b)] = len(osage_nodes)
        osage_nodes.extend(b.nodes)

    def write_block(block):
        w.string_offset(block.parent_name)
        w.floats(block.position)
        w.floats(block.rotation)
        w.floats(block.scale)
        if isinstance(block, ExpressionBlock):
            w.string_offset(block.name)
            w.i32(len(block.expressions))
            for e in block.expressions:
                w.string_offset(e)
            w.nulls(4 * (9 - len(block.expressions)))
        elif isinstance(block, MotionBlock):
            w.string_offset(block.name)
            w.i32(len(block.nodes))

            def names():
                for name, _ in block.nodes:
                    w.u32(sid(name))

            def mats():
                for _, m in block.nodes:
                    w.matrix(m)

            w.offset(names, 16)
            w.offset(mats, 16)
        elif isinstance(block, ConstraintBlock):
            w.string_offset(block.constraint_type)
            w.string_offset(block.name)
            w.i32(block.coupling)
            w.string_offset(block.source_name)
            for kind, value in block.data:
                if kind == "s":
                    w.string_offset(value)
                elif kind == "i":
                    w.i32(value)
                else:
                    w.f32(value)
        elif isinstance(block, OsageBlock):
            w.i32(starts[id(block)])
            w.i32(len(block.nodes))
            w.u32(sid(block.external_name))
            w.u32(sid(block.name))
            has_rot = any(abs(v) > 0 for n in block.nodes for v in n.rotation)

            def rotations():
                for n in block.nodes:
                    w.u32(sid(n.name))
                    w.f32(n.length)
                    w.floats(n.rotation)

            w.offset(rotations, 4, condition=has_rot)
            w.nulls(20)

    def ex_data():
        w.i32(len(osage_blocks))
        w.i32(len(osage_nodes))
        w.u32(0)

        def nodes():
            for n in osage_nodes:
                i = sid(n.name)
                w.u32(i)
                w.f32(n.length)
                w.u32(i & 0x7FFF)
            w.nulls(12)

        def osage_names():
            for b in osage_blocks:
                w.string_offset(b.external_name)
            w.u32(0)

        def blocks():
            for b in skin.blocks:
                w.string_offset(b.signature)
                w.offset(lambda b=b: write_block(b), 8)
            w.nulls(8)

        def string_table():
            for s in strings:
                w.string_offset(s)
            w.u32(0)

        def sibling_infos():
            w.nulls(12)

        w.offset(nodes, 4)
        w.offset(osage_names, 16)
        w.offset(blocks, 16)
        w.i32(len(strings))
        w.offset(string_table, 16)
        w.offset(sibling_infos, 16)
        w.i32(0)  # cloth count
        w.nulls(28)

    w.offset(ex_data, 16, "center", condition=bool(strings or skin.blocks))
    w.i32(len(skin.bones))

    def parents():
        for b in skin.bones:
            p = by_name.get(b.parent) if b.parent else None
            w.u32(p.id if p is not None else 0xFFFFFFFF)

    w.offset(parents, 16, "center", condition=any(b.parent in by_name for b in skin.bones if b.parent))
    w.nulls(12)


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


@dataclass
class Object:
    name: str
    id: int = 0
    meshes: List[Mesh] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    skin: Optional[Skin] = None


@dataclass
class ObjectSet:
    objects: List[Object] = field(default_factory=list)
    texture_ids: List[int] = field(default_factory=list)


def _read_vec(r: BinaryReader, pos: int, count: int, comps: int) -> np.ndarray:
    data = np.frombuffer(r.data, dtype="<f4", count=count * comps, offset=pos)
    return data.reshape(count, comps).astype(np.float32)


def _read_mesh(r: BinaryReader) -> Mesh:
    mesh = Mesh()
    r.u32()
    r.floats(4)
    submesh_count = r.i32()
    submeshes_off = r.offset()
    vertex_format = r.u32()
    r.i32()  # stride
    vertex_count = r.i32()
    attr_offsets = [r.offset() for _ in range(20)]
    mesh.flags = r.u32()
    r.u32()
    r.read(24)
    mesh.name = r.fixed_string(64)

    def attr(bit, comps):
        i = bit.bit_length() - 1
        if vertex_format & bit and attr_offsets[i]:
            return _read_vec(r, attr_offsets[i], vertex_count, comps)
        return None

    mesh.positions = attr(VA_POSITION, 3)
    mesh.normals = attr(VA_NORMAL, 3)
    mesh.tangents = attr(VA_TANGENT, 4)
    mesh.uv0 = attr(VA_TEXCOORD0, 2)
    mesh.uv1 = attr(VA_TEXCOORD1, 2)
    mesh.colors = attr(VA_COLOR0, 4)
    weights = attr(VA_BLEND_WEIGHT, 4)
    indices = attr(VA_BLEND_INDICES, 4)
    if weights is not None and indices is not None:
        mesh.blend_weights = weights
        idx = np.where((weights > 0) & (indices >= 0), np.floor(indices / 3.0 + 0.5), -1)
        mesh.blend_indices = idx.astype(np.int32)

    with r.at(submeshes_off):
        for _ in range(submesh_count):
            sm = SubMesh()
            r.u32()
            c = r.floats(4)
            sm.sphere = (tuple(c[:3]), c[3])
            sm.material_index = r.u32()
            sm.texcoord_indices = r.read(8)
            bone_count = r.i32()
            bones_off = r.offset()
            bones_per_vertex = r.u32()
            r.u32()  # primitive type
            index_format = r.i32()
            index_count = r.i32()
            indices_off = r.offset()
            sm.flags = r.u32()
            r.read(24)
            r.u32()
            if bones_per_vertex == 4 and bones_off:
                with r.at(bones_off):
                    sm.bone_indices = [r.u16() for _ in range(bone_count)]
            dtype = {0: "<u1", 1: "<u2", 2: "<u4"}[index_format]
            sm.indices = np.frombuffer(r.data, dtype=dtype, count=index_count, offset=indices_off).astype(np.uint32)
            mesh.submeshes.append(sm)
    return mesh


def _read_object(r: BinaryReader) -> Object:
    obj = Object(name="")
    r.u32()
    r.u32()
    r.floats(4)
    mesh_count = r.i32()
    meshes_off = r.offset()
    material_count = r.i32()
    materials_off = r.offset()
    for i in range(mesh_count):
        with r.at(meshes_off + i * 0xD8):
            obj.meshes.append(_read_mesh(r))
    for i in range(material_count):
        with r.at(materials_off + i * MATERIAL_SIZE):
            obj.materials.append(_read_material(r))
    return obj


def read_objset(data: bytes) -> ObjectSet:
    r = BinaryReader(data)
    sig = r.u32()
    if sig not in (0x5062500, 0x5062501):
        raise ValueError("Not a classic object set (signature 0x%X)" % sig)
    count = r.i32()
    r.i32()
    objects_off = r.offset()
    skins_off = r.offset()
    names_off = r.offset()
    ids_off = r.offset()
    tex_ids_off = r.offset()
    tex_count = r.i32()
    s = ObjectSet()
    with r.at(objects_off):
        offs = [r.offset() for _ in range(count)]
    for off in offs:
        with r.at(off):
            r.push_base(off)
            s.objects.append(_read_object(r))
            r.pop_base()
    if skins_off:
        with r.at(skins_off):
            skin_offs = [r.offset() for _ in range(count)]
        for obj, off in zip(s.objects, skin_offs):
            if off:
                with r.at(off):
                    obj.skin = _read_skin(r)
    with r.at(names_off):
        for obj in s.objects:
            obj.name = r.string_offset()
    with r.at(ids_off):
        for obj in s.objects:
            obj.id = r.u32()
    if tex_ids_off:
        with r.at(tex_ids_off):
            s.texture_ids = [r.u32() for _ in range(tex_count)]
    return s


def _write_mesh_body(w: BinaryWriter, mesh: Mesh) -> None:
    n = len(mesh.positions)
    w.u32(0)
    c, rad = bounding_sphere(mesh.positions)
    w.floats(c)
    w.f32(rad)
    w.i32(len(mesh.submeshes))

    def submeshes():
        for sm in mesh.submeshes:
            _write_submesh(w, mesh, sm)

    w.offset(submeshes, 8)

    attrs = []  # (bit, array, comps)
    if mesh.positions is not None:
        attrs.append((VA_POSITION, mesh.positions, 3))
    if mesh.normals is not None:
        attrs.append((VA_NORMAL, mesh.normals, 3))
    if mesh.tangents is not None:
        attrs.append((VA_TANGENT, mesh.tangents, 4))
    if mesh.uv0 is not None:
        attrs.append((VA_TEXCOORD0, mesh.uv0, 2))
    if mesh.uv1 is not None:
        attrs.append((VA_TEXCOORD1, mesh.uv1, 2))
    if mesh.colors is not None:
        attrs.append((VA_COLOR0, mesh.colors, 4))
    if mesh.blend_weights is not None:
        idx = np.where(mesh.blend_indices < 0, -1.0, mesh.blend_indices * 3.0)
        attrs.append((VA_BLEND_WEIGHT, mesh.blend_weights, 4))
        attrs.append((VA_BLEND_INDICES, idx, 4))

    fmt = 0
    stride = 0
    for bit, _, comps in attrs:
        fmt |= bit
        stride += comps * 4
    w.u32(fmt)
    w.i32(stride)
    w.i32(n)
    by_bit = {bit: (arr, comps) for bit, arr, comps in attrs}
    for i in range(20):
        bit = 1 << i
        if bit in by_bit:
            arr, comps = by_bit[bit]
            data = np.ascontiguousarray(np.asarray(arr, dtype="<f4").reshape(n, comps)).tobytes()
            w.offset(lambda data=data: w.write(data), 4)
        else:
            w.u32(0)
    w.u32(mesh.flags)
    w.u32(0)
    w.nulls(24)
    w.fixed_string(mesh.name, 64)


def _write_submesh(w: BinaryWriter, mesh: Mesh, sm: SubMesh) -> None:
    idx = np.asarray(sm.indices, dtype=np.uint32)
    w.u32(0)
    used = mesh.positions[np.unique(idx)] if len(idx) else mesh.positions[:0]
    c, rad = bounding_sphere(used)
    w.floats(c)
    w.f32(rad)
    w.u32(sm.material_index)
    w.write(bytes(sm.texcoord_indices)[:8].ljust(8, b"\0"))
    bones = sm.bone_indices or []
    w.i32(len(bones))

    def bone_palette():
        for b in bones:
            w.u16(b)

    w.offset(bone_palette, 4, condition=bool(bones))
    w.u32(4 if bones else 0)
    w.u32(PRIMITIVE_TRIANGLES)
    if len(idx) and int(idx.max()) >= 0xFFFF:
        raise ValueError("Mesh '%s' has too many vertices for 16-bit indices" % mesh.name)
    w.i32(INDEX_U16)
    w.i32(len(idx))
    data = idx.astype("<u2").tobytes()
    w.offset(lambda: w.write(data), 8)
    w.u32(sm.flags)
    w.nulls(24)
    w.u32(0)  # index offset


def _write_object(w: BinaryWriter, obj: Object) -> None:
    w.u32(0x10000)
    w.u32(0)
    allpos = [m.positions for m in obj.meshes if m.positions is not None and len(m.positions)]
    c, rad = bounding_sphere(np.concatenate(allpos) if allpos else None)
    w.floats(c)
    w.f32(rad)
    w.i32(len(obj.meshes))

    def meshes():
        # Mesh structs must be contiguous; their data is scheduled after them.
        for m in obj.meshes:
            _write_mesh_body(w, m)

    def materials():
        for m in obj.materials:
            _write_material(w, m)

    w.offset(meshes, 8)
    w.i32(len(obj.materials))
    w.offset(materials, 8)
    w.nulls(40)


def write_objset(s: ObjectSet) -> bytes:
    w = BinaryWriter()
    w.u32(OBJSET_SIGNATURE)
    w.i32(len(s.objects))
    w.u32(0x39393939)

    def objects():
        for obj in s.objects:
            def body(obj=obj):
                w.push_base()
                _write_object(w, obj)
                w.pop_base()
            w.offset(body, 4, "center")

    def skins():
        for obj in s.objects:
            w.offset(lambda obj=obj: _write_skin(w, obj.skin), 16, condition=obj.skin is not None)

    def names():
        for obj in s.objects:
            w.offset(lambda obj=obj: w.cstring(obj.name), 16)

    def ids():
        for obj in s.objects:
            w.u32(obj.id)

    def tex_ids():
        for t in s.texture_ids:
            w.u32(t)

    w.offset(objects, 16, "center")
    w.offset(skins, 16, "center")
    w.offset(names, 16, "center")
    w.offset(ids, 16, "center")
    w.offset(tex_ids, 16, "center")
    w.i32(len(s.texture_ids))
    return w.finish()
