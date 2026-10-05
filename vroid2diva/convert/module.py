"""Write the DivaModLoader mod folder for a converted model."""

from __future__ import annotations

import datetime
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..formats.databases import ObjectDatabase, ObjectSetInfo, write_obj_db, write_tex_db
from ..formats.farc import write_farc
from ..formats.keyval import dump_keyval, parse_keyval
from ..formats.objset import ObjectSet, write_objset
from ..formats.txp import write_texture_set
from .builder import BuildResult
from .game import ItemInfo, parse_item_table

ATTR_TEX = 0x4
ATTR_COL = 0x8


@dataclass
class ModulePlan:
    chara_prefix: str          # "mik", "rin", ...
    set_name: str              # e.g. MIKITM987
    object_name: str           # e.g. MIKITM987_VROID
    set_id: int
    object_id: int
    item_no: int
    cos_id: int
    module_id: Optional[int]
    module_name: str


def toml_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def safe_folder_name(s: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip(" .")
    return s or "VRoid Module"


def pick_free_ids(obj_db: Optional[ObjectDatabase], seed: int):
    """Choose an object-set ID and object ID that are not used by the game.

    The classic obj_db stores IDs as 16-bit values, so a high, sparsely used
    range is searched starting from a value derived from the model name.
    """
    used_sets = {s.id for s in obj_db.sets} if obj_db else set()
    used_objs = {oid for s in obj_db.sets for oid, _ in s.objects} if obj_db else set()
    base = 0xC000 + (seed % 0x3000)
    set_id = base
    while set_id in used_sets:
        set_id = 0xC000 + ((set_id - 0xC000 + 1) % 0x3FF0)
    obj_id = base
    while obj_id in used_objs:
        obj_id = 0xC000 + ((obj_id - 0xC000 + 1) % 0x3FF0)
    return set_id, obj_id


def edit_item_table(text: str, body_item: Optional[ItemInfo], plan: ModulePlan, display: str) -> str:
    """Add the VRoid item and make the costume use only that item."""
    items, costumes, kv = parse_item_table(text)
    if plan.item_no in items:
        raise ValueError("Item number %d is already used." % plan.item_no)
    new_index = int(kv.get("item.length", "0"))
    p = "item.%d." % new_index
    raw: Dict[str, str] = {}
    if body_item is not None:
        for k, v in body_item.raw.items():
            if k.startswith(("data.", "objset.")):
                continue
            raw[k] = v
    raw.setdefault("flag", "0")
    raw.setdefault("type", "0")
    raw.setdefault("des_id", "0")
    raw.setdefault("sub_id", "10")
    raw.setdefault("exclusion", "0")
    raw.setdefault("point", "0")
    raw["attr"] = str((int(raw.get("attr", "1")) | 1) & ~(ATTR_TEX | ATTR_COL))
    raw["org_itm"] = "0"
    raw["no"] = str(plan.item_no)
    raw["name"] = display
    raw["objset.length"] = "1"
    raw["objset.0"] = plan.set_name
    rpk = body_item.objects[0][1] if body_item is not None and body_item.objects else -1
    raw["data.obj.length"] = "1"
    raw["data.obj.0.uid"] = plan.object_name
    raw["data.obj.0.rpk"] = str(rpk)
    for k, v in raw.items():
        kv[p + k] = v
    kv["item.length"] = str(new_index + 1)

    # Point the costume at the new item only.
    cos_index = None
    for i in range(int(kv.get("cos.length", "0"))):
        if kv.get("cos.%d.id" % i) == str(plan.cos_id):
            cos_index = i
            break
    if cos_index is None:
        raise ValueError("Costume %d was not found in the item table." % plan.cos_id)
    prefix = "cos.%d.item." % cos_index
    for k in [k for k in kv if k.startswith(prefix)]:
        del kv[k]
    kv[prefix + "0"] = str(plan.item_no)
    kv[prefix + "length"] = "1"
    return dump_keyval(kv)


def write_mod(out_dir: str, result: BuildResult, plan: ModulePlan, chritm_files: Optional[Dict[str, bytes]],
              body_item: Optional[ItemInfo], mod_title: str, author: str, log=print) -> str:
    rom = os.path.join(out_dir, "rom")
    objset_dir = os.path.join(rom, "objset")
    os.makedirs(objset_dir, exist_ok=True)

    result.obj.name = plan.object_name
    result.obj.id = plan.object_id
    objset = ObjectSet([result.obj], [t.id for t in result.textures])
    lower = plan.set_name.lower()
    farc = write_farc({lower + "_obj.bin": write_objset(objset),
                       lower + "_tex.bin": write_texture_set(result.textures)})
    with open(os.path.join(objset_dir, lower + ".farc"), "wb") as f:
        f.write(farc)

    db = ObjectDatabase([ObjectSetInfo(plan.set_name, plan.set_id, lower + "_obj.bin", lower + "_tex.bin",
                                       lower + ".farc", [(plan.object_id, plan.object_name)])])
    with open(os.path.join(objset_dir, "mod_obj_db.bin"), "wb") as f:
        f.write(write_obj_db(db))
    with open(os.path.join(objset_dir, "mod_tex_db.bin"), "wb") as f:
        f.write(write_tex_db({t.id: t.name for t in result.textures}))

    if result.skin_params:
        skp_dir = os.path.join(rom, "skin_param")
        os.makedirs(skp_dir, exist_ok=True)
        with open(os.path.join(skp_dir, "ext_skp_%s.txt" % plan.object_name.lower()), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write(dump_keyval(result.skin_params))

    if chritm_files is not None:
        table = "%sitm_tbl.txt" % plan.chara_prefix
        lower_map = {k.lower(): k for k in chritm_files}
        if table not in lower_map:
            raise ValueError("%s is missing from chritm_prop.farc" % table)
        key = lower_map[table]
        text = chritm_files[key].decode("utf-8", errors="surrogateescape")
        new_text = edit_item_table(text, body_item, plan, plan.module_name)
        files = dict(chritm_files)
        files[key] = new_text.encode("utf-8", errors="surrogateescape")
        with open(os.path.join(rom, "chritm_prop.farc"), "wb") as f:
            f.write(write_farc(files))

    if plan.module_id is not None:
        lang = os.path.join(rom, "lang2")
        os.makedirs(lang, exist_ok=True)
        with open(os.path.join(lang, "mod_str_array.toml"), "w", encoding="utf-8", newline="\n") as f:
            f.write("module.%d = %s\n" % (plan.module_id, toml_string(plan.module_name)))

    with open(os.path.join(out_dir, "config.toml"), "w", encoding="utf-8", newline="\n") as f:
        f.write('enabled = true\ninclude = ["."]\n\n')
        f.write("name = %s\n" % toml_string(mod_title))
        f.write("description = %s\n" % toml_string(
            "VRoid model converted with vroid2diva. Replaces module %s." % (
                plan.module_id if plan.module_id is not None else "costume %d" % plan.cos_id)))
        f.write('version = "1.0"\n')
        f.write("date = %s\n" % toml_string(datetime.date.today().strftime("%d.%m.%Y")))
        f.write("author = %s\n" % toml_string(author or "vroid2diva"))
    log("Mod written to %s" % out_dir)
    return out_dir
