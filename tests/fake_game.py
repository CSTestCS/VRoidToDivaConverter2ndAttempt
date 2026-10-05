"""Build a tiny fake Mega Mix+ install (CPK + data files) for end-to-end tests."""

from __future__ import annotations

import os
import struct

from vroid2diva.formats.cpk import _IV, _KEY
from vroid2diva.formats.databases import ObjectDatabase, ObjectSetInfo, write_obj_db, write_tex_db
from vroid2diva.formats.farc import write_farc
from vroid2diva.formats.objset import write_objset
from vroid2diva.formats.txp import write_texture_set

from .synthetic import ITEM_TABLE, make_reference_objset

MODULE_TABLE = """module.data_list.length=2
module.0.chara=MIKU
module.0.cos=COS_001
module.0.id=0
module.0.name=Hatsune Miku
module.0.sort_index=0
module.1.chara=RIN
module.1.cos=COS_001
module.1.id=1
module.1.name=Kagamine Rin
module.1.sort_index=1
"""


def _utf_table(name: str, columns, rows) -> bytes:
    """Write a CRI @UTF table. ``columns`` = [(name, type_code)] with 4=u32, 7=u64, 0xA=str."""
    strings = bytearray(b"<NULL>\0")
    offsets = {}

    def s(v):
        if v not in offsets:
            offsets[v] = len(strings)
            strings.extend(v.encode() + b"\0")
        return offsets[v]

    s(name)
    fields = b"".join(struct.pack(">BI", 0x50 | t, s(c)) for c, t in columns)
    sizes = {4: 4, 7: 8, 0xA: 4}
    stride = sum(sizes[t] for _, t in columns)
    row_bytes = bytearray()
    for row in rows:
        for (c, t) in columns:
            v = row[c]
            row_bytes += struct.pack(">I", v) if t == 4 else struct.pack(">Q", v) if t == 7 else struct.pack(">I", s(v))
    header_len = 24
    rows_off = header_len + len(fields)
    string_off = rows_off + len(row_bytes)
    data_off = string_off + len(strings)
    body = struct.pack(">HHIIIHHI", 1, rows_off, string_off, data_off, offsets[name], len(columns), stride, len(rows))
    body += fields + row_bytes + strings
    return b"@UTF" + struct.pack(">I", len(body)) + body


def _encrypt(data: bytes) -> bytes:
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    p = padding.PKCS7(128).padder()
    data = p.update(data) + p.finalize()
    enc = Cipher(algorithms.AES(_KEY), modes.CBC(_IV)).encryptor()
    return enc.update(data) + enc.finalize()


def write_cpk(path: str, files: dict) -> None:
    content = bytearray()
    entries = []
    base = 0x800
    for name, data in files.items():
        enc = _encrypt(data)
        entries.append((name, base + len(content), len(enc)))
        content += enc
        content += b"\0" * (-len(content) % 16)
    toc_offset = base + len(content)
    rows = []
    for name, off, size in entries:
        d, _, f = name.rpartition("/")
        rows.append({"DirName": d, "FileName": f, "FileOffset": off, "FileSize": size})
    toc = _utf_table("CpkTocInfo", [("DirName", 0xA), ("FileName", 0xA), ("FileOffset", 7), ("FileSize", 4)], rows)
    cpk = _utf_table("CpkHeader", [("TocOffset", 7), ("ContentOffset", 7), ("Align", 4)],
                     [{"TocOffset": toc_offset, "ContentOffset": 0, "Align": 1}])
    out = bytearray(b"CPK " + struct.pack("<III", 0xFF, len(cpk), 0) + cpk)
    out += b"\0" * (base - len(out))
    out += content
    out += b"TOC " + struct.pack("<III", 0xFF, len(toc), 0) + toc
    with open(path, "wb") as f:
        f.write(out)


def make_fake_game(folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    body = make_reference_objset()
    db = ObjectDatabase([
        ObjectSetInfo("MIKITM001", 1, "mikitm001_obj.bin", "mikitm001_tex.bin", "mikitm001.farc",
                      [(1, "MIKITM001_BODY")]),
        ObjectSetInfo("MIKITM500", 2, "mikitm500_obj.bin", "mikitm500_tex.bin", "mikitm500.farc", []),
    ])
    files = {
        "rom/objset/obj_db.bin": write_obj_db(db),
        "rom/objset/tex_db.bin": write_tex_db({}),
        "rom/objset/mikitm001.farc": write_farc({"mikitm001_obj.bin": write_objset(body),
                                                 "mikitm001_tex.bin": write_texture_set([])}),
        "rom/gm_module_tbl.farc": write_farc({"gm_module_id.bin": MODULE_TABLE.encode()}),
        "rom/chritm_prop.farc": write_farc({"mikitm_tbl.txt": ITEM_TABLE.encode(),
                                            "rinitm_tbl.txt": b"cos.length=0\nitem.length=0\n"}),
    }
    write_cpk(os.path.join(folder, "diva_main.cpk"), files)
    open(os.path.join(folder, "DivaMegaMix.exe"), "wb").close()
    return folder
