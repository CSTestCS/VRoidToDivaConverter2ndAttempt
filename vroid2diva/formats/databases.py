"""Classic object (obj_db.bin) and texture (tex_db.bin) databases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..util.binary import BinaryReader, BinaryWriter


@dataclass
class ObjectSetInfo:
    name: str
    id: int
    file_name: str
    texture_file_name: str
    archive_file_name: str
    objects: List[tuple] = field(default_factory=list)  # (id, name)


@dataclass
class ObjectDatabase:
    sets: List[ObjectSetInfo] = field(default_factory=list)

    def find_set(self, name: str) -> Optional[ObjectSetInfo]:
        name = name.upper()
        for s in self.sets:
            if s.name.upper() == name:
                return s
        return None

    def find_object(self, name: str):
        name = name.upper()
        for s in self.sets:
            for oid, oname in s.objects:
                if oname.upper() == name:
                    return s, oid
        return None, None


def read_obj_db(data: bytes) -> ObjectDatabase:
    r = BinaryReader(data)
    set_count = r.i32()
    r.u32()
    sets_off = r.offset()
    obj_count = r.i32()
    objs_off = r.offset()
    db = ObjectDatabase()
    by_id: Dict[int, ObjectSetInfo] = {}
    with r.at(sets_off):
        for _ in range(set_count):
            info = ObjectSetInfo(r.string_offset(), r.u32(), r.string_offset(), r.string_offset(), r.string_offset())
            r.read(16)
            db.sets.append(info)
            by_id[info.id] = info
    with r.at(objs_off):
        for _ in range(obj_count):
            oid = r.u16()
            sid = r.u16()
            name = r.string_offset()
            if sid in by_id:
                by_id[sid].objects.append((oid, name))
    return db


def write_obj_db(db: ObjectDatabase) -> bytes:
    w = BinaryWriter()
    w.i32(len(db.sets))
    w.u32(max((s.id for s in db.sets), default=0))

    def sets():
        for s in db.sets:
            w.string_offset(s.name)
            w.u32(s.id)
            w.string_offset(s.file_name)
            w.string_offset(s.texture_file_name)
            w.string_offset(s.archive_file_name)
            w.nulls(16)

    def objects():
        for s in db.sets:
            for oid, name in s.objects:
                w.u16(oid)
                w.u16(s.id)
                w.string_offset(name)

    w.offset(sets, 16)
    w.i32(sum(len(s.objects) for s in db.sets))
    w.offset(objects, 16)
    return w.finish()


def read_tex_db(data: bytes) -> Dict[int, str]:
    r = BinaryReader(data)
    count = r.i32()
    off = r.offset()
    out = {}
    with r.at(off):
        for _ in range(count):
            tid = r.u32()
            out[tid] = r.string_offset()
    return out


def write_tex_db(textures: Dict[int, str]) -> bytes:
    w = BinaryWriter()
    w.i32(len(textures))

    def entries():
        for tid, name in textures.items():
            w.u32(tid)
            w.string_offset(name)

    w.offset(entries, 16)
    return w.finish()
