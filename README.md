# VRoid → Project DIVA Mega Mix+ module converter

Turn any **VRoid Studio** model (`.vrm`) into a **module** for *Hatsune Miku: Project DIVA Mega Mix+* (Steam).
Pick your model, pick a module to replace, click **Convert** — the tool writes a ready-to-use
[DivaModLoader](https://github.com/blueskythlikesclouds/DivaModLoader) mod into the game's `mods` folder.

* **Hair, skirt and bust physics** — VRoid spring bones become DIVA *osage* chains, so they swing during songs.
* **Facial animation** — VRoid's blend shapes (A/I/U/E/O, blink, joy, angry, …) are baked onto DIVA's face
  bones, so the game's expressions, blinking and lip sync move your model's face.
* **Eye movement** — the VRoid eye bones are driven by the game's gaze.
* Works with VRM 0.x (VRoid Studio 1.x export) and VRM 1.0 files.
* Nothing from the game is shipped: the tool reads Miku's skeleton binding from *your* copy of the game.

## Quick start

1. Install **DivaModLoader** (copy `dinput8.dll`, `config.toml` and the `mods` folder next to `DivaMegaMix.exe`).
2. Install **Python 3.9+** from <https://www.python.org/downloads/> (tick *Add python.exe to PATH*).
3. In VRoid Studio: **Export → Export as VRM** (VRM 0.0 or 1.0 both work).
4. Double-click **`VRoid2Diva.bat`** (first start installs the three Python packages it needs).
   You can also drag your `.vrm` file onto `VRoid2Diva.bat`.
5. In the window:
   1. choose your `.vrm` file,
   2. check the game folder (found automatically for normal Steam installs),
   3. choose the **module to replace** (type in the search box to filter),
   4. optionally change the name shown in game,
   5. click **Convert and install mod**.
6. Start the game and select the module you replaced. Done!

The new mod is called `VRoid - <name>` in the `mods` folder. It contains a `preview.png` showing your model
bound to the game skeleton — a quick way to check the result before starting the game. Delete the folder (or
disable it in your mod manager) to get the original module back.

> Linux / Steam Deck: run `./vroid2diva.sh` (needs `python3-tk` for the window, or use the command line below).

## Options

| Option | Default | What it does |
| --- | --- | --- |
| Hair / skirt physics | on | Converts spring bones to osage physics chains |
| Facial expressions & lip sync | on | Bakes blend shapes onto DIVA's face bones |
| Toon shading | on | Uses the game's own character shaders (`SKIN` for skin, `CLOTH` for the rest). Turn off for plain `BLINN` materials if something looks wrong |
| Max texture size | 2048 | Textures are resized (power of two) and DXT-compressed |

## Command line

```
python -m vroid2diva --list-modules                        # show module IDs
python -m vroid2diva --vrm MyModel.vrm --module 0 --name "My Model"
python -m vroid2diva --help                                # all options
```

`--game <folder>` overrides the automatic game detection, `--out <folder>` writes the mod somewhere else.

## How it works

```
.vrm ──► VRM loader ──► retarget onto DIVA skeleton ──► meshes / materials / textures ─┐
              │                      ▲                                                 ├─► *_obj.bin / *_tex.bin
              │              reference skeleton read                                    │   in rom/objset/<set>.farc
              │              from the game's CPKs                                       │
              ├─ spring bones ──► osage chains (OSG blocks) + rom/skin_param/ext_skp_*.txt
              └─ blend shapes ──► weights on DIVA face bones
```

* **Skeleton.** The module you replace is loaded from the game's CPK archives. Its skin provides DIVA's bone
  IDs, bind matrices and the expression blocks that define helper bones (e.g. arm twist bones), which are
  copied into the new object. The VRM is turned to face the same way, scaled to the reference height, and its
  arms, hands, fingers and legs are re-posed from VRoid's T-pose into the game's bind pose. Every bone is then
  bound with the game's own bone orientation, so the game's motions rotate the VRoid mesh exactly as they
  rotate Miku.
* **Physics.** Each VRM spring chain becomes an osage chain parented to the matching body bone. VRM stiffness,
  drag and hit radius map onto osage `force`, `air_res` and collision radius; VRM sphere/capsule colliders are
  converted to skin-parameter collisions on the corresponding DIVA bones (only bones listed in the game's
  `bone_data.bin` are used, so the file can never reference a bone the game does not know).
* **Face.** Mega Mix+ animates faces with bones, not morph targets. For each face bone (jaw, upper/lower
  eyelids, mouth corners, lips, brows) the converter finds the VRoid expression that moves that part of the face
  (`aa` for the jaw, `blinkLeft` for the left eyelids, `ih` for mouth corners, `ou` for the lips, `angry` for the
  brows) and weights each vertex by how far it moves in that expression relative to the region's peak
  (rotation angle around the bone pivot for jaw/eyelids, displacement for the others). The game's face motions
  then reproduce the VRoid shapes.
* **Module.** A new character item is added to `chritm_prop.farc` for your model and the chosen module's costume
  is pointed at it; `mod_obj_db.bin` / `mod_tex_db.bin` register the new object and textures, and
  `mod_str_array.toml` renames the module.

## Things to know

* **One module-replacing mod per character.** The mod ships a modified `rom/chritm_prop.farc`; if two mods
  (this one or any other module mod) both replace it, only the highest-priority one takes effect.
* **The module's icon** in the selection screen stays the original one.
* **Proportions** follow Miku's skeleton: very tall/short or chibi models are scaled to Miku's height and
  limbs follow her joint positions.
* **Facial animation is an approximation.** Blend shapes are fitted onto bones; expressions that VRoid builds
  from many blend shapes are reproduced only as far as DIVA's face bones allow. If a face looks wrong, convert
  with *Facial expressions* off (the eyes still follow the gaze).
* **Material look.** Toon shading uses the game's character shaders with only the base color texture; MToon
  outlines, rim light and shade textures are not carried over.

## Verification status

This tool was built without access to a copy of the game, so please treat the first release as a beta:

* All written files (object set, texture set, FARC, `obj_db`, `tex_db`) are read back without errors by
  **MikuMikuLibrary**, the reference implementation of these formats used by the DIVA modding community.
  The game-side behaviour (skinning, osage, skin parameters, item tables) follows the reverse-engineered
  engine in [ReDIVA](https://github.com/korenkonder/ReDIVA).
* Conversion has been run end-to-end on public VRM sample models (VRM 0.x and 1.0) and against a synthetic game
  install; the included preview renderer applies the game's skinning formula to confirm the bind pose and
  the face weights.
* It has **not yet been tested inside Mega Mix+ itself.** If something goes wrong, please open an issue with
  the log text from the converter window and the `preview.png` from the mod folder.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| "No .cpk files found" | Select the folder that contains `DivaMegaMix.exe` |
| Module list is empty | Make sure the game is fully installed/updated; try another module |
| Model is invisible in game | Check that DivaModLoader is installed and the mod is enabled; check for other mods replacing `chritm_prop.farc` |
| Black or glowing materials | Convert again with *Toon shading* off |
| Face distorts | Convert again with *Facial expressions* off |
| Hair goes crazy | Convert again with physics off, or lower the stiffness of that spring in VRoid |

## Development

```
pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

Code layout:

| Path | Purpose |
| --- | --- |
| `vroid2diva/formats/` | DIVA file formats: object sets, textures, FARC, CPK (read-only), databases, key/value text |
| `vroid2diva/vrm/` | glTF/VRM loader (VRM 0.x and 1.0 normalised to one model) |
| `vroid2diva/convert/` | reference skeleton, retargeting, physics, face baking, materials, module writer, pipeline |
| `vroid2diva/preview.py` | software renderer using the game's skinning formula |
| `vroid2diva/gui.py`, `cli.py` | front ends |
| `tests/` | unit and end-to-end tests (synthetic VRM and synthetic game install) |

Formats and engine behaviour are based on MikuMikuLibrary (MIT) and ReDIVA; no game data is included.
