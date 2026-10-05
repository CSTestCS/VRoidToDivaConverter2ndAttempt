"""Command line interface.  Run ``python -m vroid2diva --help``."""

from __future__ import annotations

import argparse
import sys

from .convert.builder import BuildOptions
from .convert.game import find_game_dir
from .convert.pipeline import ConvertRequest, convert, list_modules


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vroid2diva",
                                 description="Convert a VRoid (.vrm) model into a Project DIVA Mega Mix+ module mod.")
    ap.add_argument("--vrm", help="VRoid model exported as .vrm")
    ap.add_argument("--game", help="Mega Mix+ folder (the one with DivaMegaMix.exe); auto-detected if omitted")
    ap.add_argument("--module", type=int, help="ID of the module to replace (see --list-modules)")
    ap.add_argument("--list-modules", action="store_true", help="list the game's modules and exit")
    ap.add_argument("--name", default="", help="module name shown in game (default: the VRM's title)")
    ap.add_argument("--out", default="", help="output mod folder (default: <game>/mods/VRoid - <name>)")
    ap.add_argument("--author", default="")
    ap.add_argument("--no-physics", action="store_true", help="do not convert spring bones")
    ap.add_argument("--no-face", action="store_true", help="do not bake facial blend shapes")
    ap.add_argument("--simple-shading", action="store_true", help="use plain BLINN materials instead of toon")
    ap.add_argument("--max-texture-size", type=int, default=2048)
    ap.add_argument("--no-compress", action="store_true", help="store textures uncompressed (large files)")
    args = ap.parse_args(argv)

    game = args.game or find_game_dir()
    if not game:
        ap.error("could not find Mega Mix+; pass --game <folder>")
    if args.list_modules:
        for m in list_modules(game):
            print("%5d  %-8s %s" % (m.id, m.chara, m.name))
        return 0
    if not args.vrm or args.module is None:
        ap.error("--vrm and --module are required (use --list-modules to find a module ID)")
    opts = BuildOptions(style="simple" if args.simple_shading else "toon", physics=not args.no_physics,
                        face=not args.no_face, compress_textures=not args.no_compress,
                        max_texture_size=args.max_texture_size)
    out = convert(ConvertRequest(args.vrm, game, args.module, args.name, args.out, args.author, opts))
    print("\nDone! Your mod is in:\n  %s\nStart the game and pick the module you replaced." % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
