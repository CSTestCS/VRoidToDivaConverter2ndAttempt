"""High level conversion entry point shared by the GUI and the CLI."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ..util.murmur import murmur_hash
from ..vrm.model import load_vrm
from .builder import BuildOptions, build_object
from .game import GameData, ItemInfo, ModuleInfo, parse_item_table
from .module import ModulePlan, pick_free_ids, safe_folder_name, write_mod
from .reference import reference_from_objsets

# Bones that tell the body item apart from head/hair/accessory items.
_BODY_BONES = ("kl_mune_b_wj", "kl_kosi_etc_wj", "kl_te_l_wj", "kl_te_r_wj", "j_sune_l_wj", "j_sune_r_wj",
               "n_skata_l_wj_cd_ex", "n_momo_a_l_wj_cd_ex")


@dataclass
class ConvertRequest:
    vrm_path: str
    game_dir: str
    module_id: int
    module_name: str = ""         # new display name ("" = keep VRM title)
    output_dir: str = ""          # "" = <game>/mods/<name>
    author: str = ""
    options: BuildOptions = field(default_factory=BuildOptions)


def list_modules(game_dir: str, log=print) -> List[ModuleInfo]:
    return GameData(game_dir, log).modules()


def convert(req: ConvertRequest, log: Callable[[str], None] = print, progress: Callable[[float], None] = None) -> str:
    progress = progress or (lambda f: None)
    log("Reading game data from %s ..." % req.game_dir)
    game = GameData(req.game_dir, log)
    modules = {m.id: m for m in game.modules()}
    if req.module_id not in modules:
        raise ValueError("Module %d does not exist in this game version." % req.module_id)
    module = modules[req.module_id]
    progress(0.05)

    chritm = game.chritm_prop()
    table_name = "%sitm_tbl.txt" % module.prefix
    table_key = next((k for k in chritm if k.lower() == table_name), None)
    if table_key is None:
        raise ValueError("%s not found in chritm_prop.farc" % table_name)
    items, costumes, _ = parse_item_table(chritm[table_key].decode("utf-8", errors="surrogateescape"))
    if module.cos not in costumes:
        raise ValueError("Costume %d of %s was not found." % (module.cos, module.chara))

    # Load the module's own objects: they are the reference skeleton.
    objsets, set_items = [], []
    for no in costumes[module.cos]:
        item = items.get(no)
        if item is None:
            continue
        for set_name in item.objsets:
            try:
                objsets.append(game.load_objset(set_name))
                set_items.append(item)
            except (KeyError, FileNotFoundError) as e:
                log("Skipping %s (%s)" % (set_name, e))
    if not objsets:
        raise ValueError("Could not load any objects of module '%s'." % module.name)

    def body_score(objset):
        names = {b.name for o in objset.objects if o.skin for b in o.skin.bones}
        return sum(1 for b in _BODY_BONES if b in names)

    scored = sorted(zip(objsets, set_items), key=lambda x: -body_score(x[0]))
    body_item: Optional[ItemInfo] = scored[0][1]
    collision_bones, all_bone_names = game.bone_names()
    ref = reference_from_objsets([s for s, _ in scored], collision_bones, all_bone_names)
    log("Reference skeleton: %d bones from %s." % (len(ref.bones), ", ".join(ref.source_objects)))
    progress(0.15)

    log("Loading %s ..." % os.path.basename(req.vrm_path))
    model = load_vrm(req.vrm_path)
    display = req.module_name or model.title
    progress(0.25)

    item_no = max(items) + 1 if items else 900
    set_name = "%sITM%03d" % (module.prefix.upper(), item_no)
    set_id, obj_id = pick_free_ids(game.obj_db, murmur_hash(set_name + display))
    plan = ModulePlan(module.prefix, set_name, set_name + "_VROID", set_id, obj_id, item_no, module.cos,
                      module.id, display)

    opts = req.options
    opts.object_name = plan.object_name
    opts.texture_prefix = set_name
    result = build_object(model, ref, opts, log)
    progress(0.85)

    out = req.output_dir or os.path.join(req.game_dir, "mods", safe_folder_name("VRoid - " + display))
    write_mod(out, result, plan, chritm, body_item, "VRoid: " + display, req.author, log)
    try:
        from ..preview import game_bone_worlds, render
        render(result.obj, game_bone_worlds(result.obj, ref, result.chains), os.path.join(out, "preview.png"),
               title="%s (bound to %s's skeleton)" % (display, module.chara.title()))
    except Exception as e:  # a preview must never break the conversion
        log("Preview image skipped (%s)." % e)
    progress(1.0)
    return out
