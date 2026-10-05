"""Tests for vroid2diva.  Run with: python -m unittest discover -s tests -t ."""

import os
import shutil
import struct
import tempfile
import unittest

import numpy as np

from vroid2diva.convert.builder import BuildOptions, build_object
from vroid2diva.convert.game import GameData, parse_item_table
from vroid2diva.convert.physics import euler_zyx
from vroid2diva.convert.pipeline import ConvertRequest, convert, list_modules
from vroid2diva.convert.reference import reference_from_objsets
from vroid2diva.convert.retarget import axis_angle
from vroid2diva.formats.databases import read_obj_db, read_tex_db
from vroid2diva.formats.farc import read_farc, write_farc
from vroid2diva.formats.keyval import parse_keyval
from vroid2diva.formats.objset import (BoneInfo, ExpressionBlock, Material, Mesh, Object, ObjectSet,
                                       OsageBlock, OsageNode, Skin, SubMesh, read_objset, write_objset)
from vroid2diva.formats.txp import encode_texture, read_texture_set, write_texture_set
from vroid2diva.util.murmur import murmur_hash
from vroid2diva.vrm.model import load_vrm

from .fake_game import make_fake_game
from .synthetic import make_reference_objset
from .vrm_builder import make_test_vrm

QUIET = lambda *_: None  # noqa: E731


def _read(path):
    with open(path, "rb") as f:
        return f.read()


class FormatTests(unittest.TestCase):
    def test_objset_roundtrip(self):
        n = 12
        mesh = Mesh(name="m", positions=np.random.rand(n, 3).astype(np.float32),
                    normals=np.random.rand(n, 3).astype(np.float32), uv0=np.random.rand(n, 2).astype(np.float32),
                    blend_weights=np.tile([0.75, 0.25, 0, 0], (n, 1)).astype(np.float32),
                    blend_indices=np.tile([1, 0, -1, -1], (n, 1)))
        mesh.submeshes.append(SubMesh(indices=np.arange(12), bone_indices=[0, 1]))
        skin = Skin([BoneInfo("kl_mune_b_wj", 8), BoneInfo("j_v00_000_wj", is_ex=True, parent="kl_mune_b_wj")],
                    [ExpressionBlock(parent_name="kl_mune_b_wj", name="n_x_ex", expressions=["= 0 v 1"]),
                     OsageBlock(parent_name="kl_mune_b_wj", name="e_v00_osg", external_name="c_v00_osg",
                                nodes=[OsageNode("j_v00_000_wj", 0.1)])])
        data = write_objset(ObjectSet([Object("TEST", 5, [mesh], [Material(name="mat")], skin)], [7]))
        back = read_objset(data)
        obj = back.objects[0]
        self.assertEqual((obj.name, obj.id, back.texture_ids), ("TEST", 5, [7]))
        np.testing.assert_allclose(obj.meshes[0].positions, mesh.positions)
        np.testing.assert_array_equal(obj.meshes[0].blend_indices, mesh.blend_indices)
        self.assertEqual(obj.meshes[0].submeshes[0].bone_indices, [0, 1])
        self.assertEqual(obj.skin.bones[1].id & 0x8000, 0x8000)
        self.assertEqual(obj.skin.bones[1].parent, "kl_mune_b_wj")
        osg = [b for b in obj.skin.blocks if isinstance(b, OsageBlock)][0]
        self.assertEqual(osg.external_name, "c_v00_osg")
        self.assertEqual(osg.nodes[0].name, "j_v00_000_wj")

    def test_farc_and_texture(self):
        from PIL import Image
        tex = encode_texture(Image.new("RGBA", (64, 32), (255, 0, 0, 255)), "T", 3, has_alpha=False)
        data = write_texture_set([tex])
        back = read_texture_set(data)
        self.assertEqual(len(back[0].mips), len(tex.mips))
        self.assertEqual((back[0].mips[0].width, back[0].mips[0].height), (64, 32))
        arc = read_farc(write_farc({"a.bin": data, "b.txt": b"x=1\n"}))
        self.assertEqual(arc["a.bin"], data)

    def test_murmur(self):
        self.assertEqual(murmur_hash("abc"), murmur_hash("ABC"))
        self.assertNotEqual(murmur_hash("MIKITM001"), murmur_hash("MIKITM002"))

    def test_euler_zyx(self):
        r = axis_angle([0, 0, 1], 0.3) @ axis_angle([0, 1, 0], -0.2) @ axis_angle([1, 0, 0], 0.7)
        np.testing.assert_allclose(euler_zyx(r), [0.7, -0.2, 0.3], atol=1e-9)


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.vrm = make_test_vrm(os.path.join(cls.tmp, "test.vrm"))
        cls.ref = reference_from_objsets([make_reference_objset()], collision_bones={"j_kao_wj", "kl_mune_b_wj"})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_vrm_loads(self):
        m = load_vrm(self.vrm)
        self.assertIn("head", m.humanoid)
        self.assertTrue(m.springs)
        self.assertIn("aa", m.expressions)

    def test_build_object_binds_to_reference_pose(self):
        from vroid2diva.preview import game_bone_worlds, skin_vertices
        m = load_vrm(self.vrm)
        r = build_object(m, self.ref, BuildOptions(max_texture_size=64), QUIET)
        self.assertTrue(r.obj.meshes)
        self.assertEqual(len(r.chains), 1)
        # In the game's rest pose the hand must sit where the reference hand is.
        worlds = game_bone_worlds(r.obj, self.ref, r.chains)
        hand_ref = self.ref.position("kl_te_l_wj")
        pts = np.concatenate([p for p, _, _, _ in skin_vertices(r.obj, worlds)])
        self.assertLess(np.min(np.linalg.norm(pts - hand_ref, axis=1)), 0.05)
        # Every bone weight sum is 1.
        for mesh in r.obj.meshes:
            np.testing.assert_allclose(mesh.blend_weights.sum(axis=1), 1.0, atol=1e-5)
        # The jaw gets the "aa" blend shape.
        self.assertTrue(any(f["bone"] == "kl_ago_wj" and f["driver"] == "aa" for f in r.face_fit))

    def test_full_pipeline_with_fake_game(self):
        game = make_fake_game(os.path.join(self.tmp, "game"))
        mods = list_modules(game, QUIET)
        self.assertEqual([m.name for m in mods], ["Hatsune Miku", "Kagamine Rin"])
        out = convert(ConvertRequest(self.vrm, game, 0, "My VRoid",
                                     output_dir=os.path.join(self.tmp, "mod"),
                                     options=BuildOptions(max_texture_size=64)), QUIET)
        objset_dir = os.path.join(out, "rom", "objset")
        db = read_obj_db(_read(os.path.join(objset_dir, "mod_obj_db.bin")))
        info = db.sets[0]
        self.assertEqual(info.name, "MIKITM501")
        farc = read_farc(_read(os.path.join(objset_dir, info.archive_file_name)))
        objset = read_objset(farc[info.file_name])
        self.assertEqual(objset.objects[0].name, info.objects[0][1])
        tex_db = read_tex_db(_read(os.path.join(objset_dir, "mod_tex_db.bin")))
        self.assertEqual(set(tex_db), set(objset.texture_ids))
        tbl = read_farc(_read(os.path.join(out, "rom", "chritm_prop.farc")))
        items, costumes, _ = parse_item_table(tbl["mikitm_tbl.txt"].decode())
        self.assertEqual(costumes[0], [501])
        self.assertEqual(items[501].objects[0][0], "MIKITM501_VROID")
        self.assertEqual(items[501].objsets, ["MIKITM501"])
        self.assertIn(b"rinitm_tbl.txt", b"".join(k.encode() for k in tbl))
        skp = parse_keyval(_read(os.path.join(out, "rom", "skin_param", "ext_skp_mikitm501_vroid.txt")).decode())
        self.assertIn("c_v00_osg.root.force", skp)
        self.assertTrue(os.path.isfile(os.path.join(out, "config.toml")))
        self.assertIn('module.0 = "My VRoid"', _read(os.path.join(out, "rom", "lang2", "mod_str_array.toml")).decode())


if __name__ == "__main__":
    unittest.main()
