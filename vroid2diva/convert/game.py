"""Access to the user's Mega Mix+ installation.

The converter never ships any game data.  Instead it reads what it needs
(Miku's skeleton binding, the module/item tables, databases) straight out of
the CPK archives of the user's own copy of the game.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..formats.cpk import CpkArchive
from ..formats.databases import ObjectDatabase, read_obj_db, read_tex_db
from ..formats.farc import read_farc
from ..formats.keyval import parse_keyval
from ..formats.objset import ObjectSet, read_objset
from ..util.binary import BinaryReader

STEAM_APP_ID = "1761390"
GAME_FOLDER_NAMES = ["Hatsune Miku Project DIVA Mega Mix Plus"]
EXE_NAME = "DivaMegaMix.exe"

CHARA_PREFIX = {
    "MIKU": "mik", "RIN": "rin", "LEN": "len", "LUKA": "luk", "NERU": "ner",
    "HAKU": "hak", "KAITO": "kai", "MEIKO": "mei", "SAKINE": "sak", "TETO": "tet",
}


@dataclass
class ModuleInfo:
    id: int
    name: str
    chara: str
    cos: int  # costume index in the item table (COS_001 -> 0)

    @property
    def prefix(self) -> str:
        return CHARA_PREFIX.get(self.chara.upper(), "mik")

    def label(self) -> str:
        return "%s  [%s, module %d]" % (self.name, self.chara.title(), self.id)


@dataclass
class ItemInfo:
    no: int
    name: str
    objsets: List[str]
    objects: List[tuple]  # (uid, rpk)
    sub_id: int
    attr: int
    raw: Dict[str, str] = field(default_factory=dict)  # keys relative to "item.N."


def find_game_dir() -> Optional[str]:
    """Best-effort search of Steam libraries for the game folder."""
    candidates: List[str] = []
    steam_roots: List[str] = []
    if sys.platform.startswith("win"):
        try:
            import winreg
            for hive, key in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                              (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        for value in ("SteamPath", "InstallPath"):
                            try:
                                steam_roots.append(winreg.QueryValueEx(k, value)[0])
                            except OSError:
                                pass
                except OSError:
                    pass
        except ImportError:
            pass
        steam_roots += [r"C:\Program Files (x86)\Steam", r"C:\Program Files\Steam"]
    else:
        home = os.path.expanduser("~")
        steam_roots += [os.path.join(home, ".steam", "steam"), os.path.join(home, ".local", "share", "Steam"),
                        os.path.join(home, "Library", "Application Support", "Steam")]

    libraries = []
    for root in steam_roots:
        vdf = os.path.join(root, "steamapps", "libraryfolders.vdf")
        libraries.append(root)
        if os.path.isfile(vdf):
            try:
                with open(vdf, encoding="utf-8", errors="ignore") as f:
                    libraries += [p.replace("\\\\", "\\") for p in re.findall(r'"path"\s+"([^"]+)"', f.read())]
            except OSError:
                pass
    for lib in libraries:
        for name in GAME_FOLDER_NAMES:
            candidates.append(os.path.join(lib, "steamapps", "common", name))
    for c in candidates:
        if is_game_dir(c):
            return os.path.normpath(c)
    return None


def is_game_dir(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and bool(glob.glob(os.path.join(path, "*.cpk")))


class GameData:
    """Reads files from the game's CPK archives (and loose ``rom`` folders)."""

    def __init__(self, game_dir: str, log=print):
        if not is_game_dir(game_dir):
            raise FileNotFoundError("No .cpk files found in '%s'. Please select the folder that contains %s."
                                    % (game_dir, EXE_NAME))
        self.game_dir = game_dir
        self.log = log
        self.archives: List[CpkArchive] = []
        # Load the main archive first so it has the lowest priority, DLC after it.
        paths = sorted(glob.glob(os.path.join(game_dir, "*.cpk")), key=lambda p: (0 if "main" in p.lower() else 1, p))
        for p in paths:
            try:
                self.archives.append(CpkArchive(p))
            except Exception as e:  # pragma: no cover - depends on user files
                log("Skipping %s (%s)" % (os.path.basename(p), e))
        self._index: Dict[str, tuple] = {}
        for arc in self.archives:
            for name in arc.names():
                self._index[name] = (arc, name)
        self._cache: Dict[str, bytes] = {}
        self._obj_db: Optional[ObjectDatabase] = None

    # -- raw files ----------------------------------------------------------
    def find(self, path: str) -> Optional[str]:
        path = path.lower().replace("\\", "/")
        if path in self._index:
            return path
        # Fall back to any archive path ending with the requested path.
        matches = [n for n in self._index if n.endswith("/" + path) or n.endswith(path)]
        if not matches:
            return None
        return min(matches, key=len)

    def exists(self, path: str) -> bool:
        return self.find(path) is not None

    def read(self, path: str) -> bytes:
        key = self.find(path)
        if key is None:
            raise FileNotFoundError("'%s' was not found in the game's archives." % path)
        if key not in self._cache:
            arc, name = self._index[key]
            self._cache[key] = arc.read(name)
        return self._cache[key]

    # -- databases ------------------------------------------------------------
    @property
    def obj_db(self) -> ObjectDatabase:
        if self._obj_db is None:
            self._obj_db = read_obj_db(self.read("rom/objset/obj_db.bin"))
        return self._obj_db

    def tex_db(self) -> Dict[int, str]:
        return read_tex_db(self.read("rom/objset/tex_db.bin"))

    def modules(self) -> List[ModuleInfo]:
        files = read_farc(self.read("rom/gm_module_tbl.farc"))
        data = files.get("gm_module_id.bin") or next(iter(files.values()))
        kv = parse_keyval(data.decode("utf-8", errors="replace"))
        out = []
        count = int(kv.get("module.data_list.length", "0"))
        for i in range(count):
            p = "module.%d." % i
            if p + "id" not in kv:
                continue
            cos = kv.get(p + "cos", "COS_001")
            m = re.match(r"COS_(\d+)", cos)
            out.append(ModuleInfo(int(kv[p + "id"]), kv.get(p + "name", "module %s" % kv[p + "id"]),
                                  kv.get(p + "chara", "MIKU"), int(m.group(1)) - 1 if m else 0))
        out.sort(key=lambda m: m.id)
        return out

    def chritm_prop(self) -> Dict[str, bytes]:
        return read_farc(self.read("rom/chritm_prop.farc"))

    def bone_names(self):
        """(bone names usable in skin-param collisions, all skeleton names) or (None, set())."""
        try:
            data = self.read("rom/bone_data.bin")
        except FileNotFoundError:
            return None, set()
        try:
            return read_bone_data_names(data)
        except Exception as e:  # pragma: no cover - depends on game data
            self.log("Could not read bone_data.bin (%s); physics colliders will be skipped." % e)
            return None, set()

    def load_objset(self, set_name: str) -> ObjectSet:
        info = self.obj_db.find_set(set_name)
        if info is None:
            raise KeyError("Object set %s is not in obj_db.bin" % set_name)
        farc = read_farc(self.read("rom/objset/" + info.archive_file_name))
        lower = {k.lower(): v for k, v in farc.items()}
        objset = read_objset(lower[info.file_name.lower()])
        db_names = {oid: name for oid, name in info.objects}
        for obj in objset.objects:
            if not obj.name and obj.id in db_names:
                obj.name = db_names[obj.id]
        return objset


def read_bone_data_names(data: bytes):
    r = BinaryReader(data)
    r.u32()
    count = r.i32()
    skel_off = r.offset()
    names_off = r.offset()
    with r.at(names_off):
        skel_names = [r.string_offset() for _ in range(count)]
    with r.at(skel_off):
        offs = [r.offset() for _ in range(count)]
    idx = next((i for i, n in enumerate(skel_names) if n and n.upper() == "CMN"), 0)
    lists = []
    with r.at(offs[idx]):
        r.offset()
        r.i32()
        r.offset()
        r.offset()
        n1 = r.i32()
        o1 = r.offset()
        n2 = r.i32()
        o2 = r.offset()
    for n, o in ((n1, o1), (n2, o2)):
        names = set()
        with r.at(o):
            for _ in range(n):
                names.add(r.string_offset())
        lists.append(names)
    # The first table is what skin-param collisions look names up in.
    return lists[0], lists[0] | lists[1]


def parse_item_table(text: str):
    """Parse ``*itm_tbl.txt`` into (items by number, costumes, raw keyvals)."""
    kv = parse_keyval(text)
    items: Dict[int, ItemInfo] = {}
    for i in range(int(kv.get("item.length", "0"))):
        p = "item.%d." % i
        if p + "no" not in kv:
            continue
        raw = {k[len(p):]: v for k, v in kv.items() if k.startswith(p)}
        objsets = [raw["objset.%d" % j] for j in range(int(raw.get("objset.length", "0"))) if "objset.%d" % j in raw]
        objects = []
        for j in range(int(raw.get("data.obj.length", "0"))):
            uid = raw.get("data.obj.%d.uid" % j)
            if uid:
                objects.append((uid, int(raw.get("data.obj.%d.rpk" % j, "-1"))))
        items[int(raw["no"])] = ItemInfo(int(raw["no"]), raw.get("name", ""), objsets, objects,
                                         int(raw.get("sub_id", "-1")), int(raw.get("attr", "0")), raw)
    costumes: Dict[int, List[int]] = {}
    for i in range(int(kv.get("cos.length", "0"))):
        p = "cos.%d." % i
        if p + "id" not in kv:
            continue
        cid = int(kv[p + "id"])
        costumes[cid] = [int(kv[p + "item.%d" % j]) for j in range(int(kv.get(p + "item.length", "0")))
                         if p + "item.%d" % j in kv]
    return items, costumes, kv
