"""DIVA ``key=value`` text files (item tables, skin parameters, ...).

The game sorts keys when loading, but its own files are stored sorted with a
plain byte-wise comparison, so we write them the same way.
"""

from __future__ import annotations

from typing import Dict


def parse_keyval(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.rstrip("\r")
        if not line or line[0] in "#!":
            continue
        key, sep, value = line.partition("=")
        if sep:
            out[key] = value
    return out


def dump_keyval(values: Dict[str, object], header: str = "") -> str:
    lines = []
    if header:
        lines.extend("# " + h for h in header.splitlines())
    for key in sorted(values, key=lambda k: k.encode("utf-8")):
        v = values[key]
        if isinstance(v, float):
            v = ("%.6f" % v).rstrip("0").rstrip(".") if v != 0 else "0"
            if v in ("-0",):
                v = "0"
        lines.append("%s=%s" % (key, v))
    return "\n".join(lines) + "\n"
