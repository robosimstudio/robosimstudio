# SPDX-License-Identifier: Apache-2.0
"""Which renderers exist, and what each one can produce.

These moved out of ``test_core.py`` when the backends became
``robosimstudio.solvers.render``: a test module follows the code it tests, and
what is left in core is the contract -- ``RendererBase``, a frame, a frame set --
rather than any implementation of it.

Static: ``supported_channels`` and ``backend_class`` read a class without
constructing one, so none of this needs a model or a GPU. Four of the five
backends are optional extras, which is exactly why resolving them by name has to
be checked: a menu built from ``list_renderers`` must not offer a name that
raises.
"""

from __future__ import annotations

import os
import types
import unittest
from unittest.mock import patch

import numpy as np

from robosimstudio.core.cameras import CameraSpec
from robosimstudio.core.render.base import RendererBase
from robosimstudio.core.render.spec import RenderSpec, merge_render_kwargs
from robosimstudio.solvers.render import backend_class, list_renderers, look_kwargs, supported_channels
from robosimstudio.solvers.render.rtx import usd_film_back
from robosimstudio.solvers.render.tiled import aperture_from_intrinsics

TIER = "fast"
"""No GPU. Gates every commit."""


class TestRendererChannels(unittest.TestCase):
    """What a backend can produce, and what happens when it is asked for more.

    The bug this guards: an offline re-render asked a path tracer for the
    segmentation channel the run happened to be *collected* with, and the
    constructor raised -- so no run recorded with segmentation could be
    re-rendered by RTX at all, which is the whole point of collecting cheap.
    Negotiating needs the answer before the constructor runs, because the
    constructor is what opens the GL context or the ovrtx viewer.
    """

    class _Backend(RendererBase):
        name = "fake"
        supported_channels = ("rgb", "depth")

    def test_a_backend_reports_its_channels_without_being_built(self):
        """``supported_channels`` reads the class, so nothing is constructed."""
        self.assertEqual(supported_channels("tiled"), ("rgb", "depth", "seg"))
        self.assertIs(backend_class("tiled").supported_channels, backend_class("tiled").supported_channels)

    def test_an_unknown_backend_raises(self):
        """A misspelled backend name fails by name, not by AttributeError."""
        with self.assertRaises(ValueError):
            backend_class("raytracer")

    def test_every_advertised_backend_resolves(self):
        """A menu built from ``list_renderers`` cannot offer a name that raises.

        Resolving is not constructing, so this needs none of the optional
        backends installed.
        """
        for backend in list_renderers():
            self.assertTrue(issubclass(backend_class(backend), RendererBase), backend)

    def test_rtx_is_the_one_offered_and_tiled_is_the_unlisted_fallback(self):
        """One renderer by name; the fallback builds but is nobody's choice."""
        from robosimstudio.solvers.render import FALLBACK, PUBLIC, default_renderer

        self.assertEqual(list_renderers(), (PUBLIC,))
        self.assertNotIn(FALLBACK, list_renderers())
        self.assertTrue(issubclass(backend_class(FALLBACK), RendererBase))
        self.assertIn(default_renderer(), (PUBLIC, FALLBACK))

    def test_the_default_is_the_rules_pick_with_nothing_asked(self):
        from robosimstudio.solvers.render import ENV, choose_renderer, default_renderer

        with patch.dict(os.environ, {ENV: ""}):
            self.assertEqual(default_renderer(), choose_renderer()[0])

    def test_the_override_narrows_what_a_camera_asked_for(self):
        """A channel override wins over the camera's own list."""
        camera = CameraSpec(name="cam", width=8, height=8, channels=("rgb", "depth", "seg"))
        renderer = self._Backend([camera], channels=("rgb", "depth"))
        self.assertEqual(renderer.channels["cam"], ("rgb", "depth"))

    def test_asking_for_an_unsupported_channel_still_raises(self):
        """Narrowing is the caller's decision; the backend never silently drops one.

        The renderer refusing is what makes the negotiation explicit -- if it
        quietly returned three channels and filled one with zeros, a recorded
        segmentation map would be indistinguishable from a scene with one label.
        """
        camera = CameraSpec(name="cam", width=8, height=8, channels=("rgb", "seg"))
        with self.assertRaises(ValueError):
            self._Backend([camera])


def _gpu(name: str, arch: int, cuda: bool = True):
    """A Warp device as :func:`rt_cores` reads one."""
    return types.SimpleNamespace(name=name, arch=arch, is_cuda=cuda, alias="cuda:0" if cuda else "cpu")


class TestTheRendererRule(unittest.TestCase):
    """Named in code, else named for the machine, else what the machine can run.

    No GPU and no ovrtx: what the machine has is patched, which is the point --
    the rule has to come out right on an A100 and on a laptop without either
    being in the room.
    """

    def choose(self, requested="", *, env="", installed=True, gpu=None):
        from robosimstudio.solvers.render import ENV, choose_renderer

        gpu = gpu or _gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition", 120)
        with (
            patch.dict(os.environ, {ENV: env}),
            patch("robosimstudio.solvers.render.rtx_available", return_value=installed),
            patch("warp.get_device", return_value=gpu),
        ):
            return choose_renderer(requested)

    def test_a_gpu_with_rt_cores_gets_the_path_tracer(self):
        for name, arch in (("Tesla T4", 75), ("NVIDIA A40", 86), ("NVIDIA L40S", 89), ("NVIDIA GeForce RTX 5090", 120)):
            self.assertEqual(self.choose(gpu=_gpu(name, arch))[0], "rtx", name)

    def test_a_gpu_without_them_gets_tiled_and_says_which(self):
        for name, arch in (("NVIDIA A100-SXM4-80GB", 80), ("NVIDIA H100 80GB HBM3", 90), ("NVIDIA B200", 100)):
            backend, why = self.choose(gpu=_gpu(name, arch))
            self.assertEqual(backend, "tiled", name)
            self.assertIn(name, why)
            self.assertIn("no RT cores", why)

    def test_turing_without_rt_cores_is_told_apart_by_name(self):
        """GTX 16 and MX share sm_75 with the T4 and RTX 20, and have none."""
        self.assertEqual(self.choose(gpu=_gpu("NVIDIA GeForce GTX 1660 Ti", 75))[0], "tiled")

    def test_an_architecture_nobody_listed_gets_tiled(self):
        self.assertEqual(self.choose(gpu=_gpu("NVIDIA Future", 130))[0], "tiled")

    def test_no_ovrtx_is_tiled_whatever_the_gpu(self):
        self.assertEqual(self.choose(installed=False), ("tiled", "ovrtx is not installed"))

    def test_no_cuda_is_tiled(self):
        self.assertEqual(self.choose(gpu=_gpu("cpu", 0, cuda=False))[0], "tiled")

    def test_the_machine_variable_wins_over_the_machine(self):
        self.assertEqual(self.choose(env="tiled"), ("tiled", "ROBOSIMSTUDIO_RENDERER=tiled"))
        self.assertEqual(self.choose(env="rtx", gpu=_gpu("NVIDIA A100", 80))[0], "rtx")

    def test_a_name_in_code_wins_over_the_variable(self):
        self.assertEqual(self.choose("tiled", env="rtx"), ("tiled", "asked for"))

    def test_auto_and_empty_name_nothing(self):
        for requested, env in (("auto", ""), ("", "auto"), ("auto", "auto")):
            self.assertEqual(self.choose(requested, env=env)[0], "rtx", (requested, env))
        self.assertEqual(self.choose("auto", env="tiled")[0], "tiled")

    def test_rtx_named_without_ovrtx_raises_rather_than_swapping(self):
        with self.assertRaisesRegex(ImportError, "ovrtx is not installed"):
            self.choose("rtx", installed=False)
        with self.assertRaisesRegex(ImportError, "ROBOSIMSTUDIO_RENDERER"):
            self.choose(env="rtx", installed=False)

    def test_a_misspelled_name_fails_by_name(self):
        with self.assertRaisesRegex(ValueError, "ROBOSIMSTUDIO_RENDERER='rtxx'"):
            self.choose(env="rtxx")
        with self.assertRaisesRegex(ValueError, "'tiles'"):
            self.choose("tiles")

    def test_a_dotted_path_is_a_name(self):
        self.assertEqual(self.choose("my.renderers.Mine")[0], "my.renderers.Mine")


class TestOneWorldRenderIsNotTheWholeFleetsSignature(unittest.TestCase):
    """``render()`` must be able to call ``render_worlds()`` on its own class.

    The two fleet lines gave ``render_worlds`` two different shapes -- one took
    the worlds to read back, the other computed a world *count* into a local of
    the same name -- and a merge kept the first's signature with the second's
    body. The local shadowed the parameter, so `worlds` was an int by the time
    it was iterated and EVERY tiled render raised

        TypeError: 'int' object is not iterable

    at ``Env.reset``, before a policy saw a frame. The whole fast tier stayed
    green because nothing here drove the backend's own render path; it took a
    gear sweep, which is the one runner that defaults to tiled.

    Read by AST rather than by rendering, so this needs no GPU and no scene: a
    parameter that is assigned inside its own function body is the defect.
    """

    def test_no_backend_shadows_a_render_worlds_parameter(self) -> None:
        import ast
        from pathlib import Path as _Path

        root = _Path(__file__).resolve().parent.parent / "robosimstudio" / "solvers" / "render"
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text())
            for fn in ast.walk(tree):
                if not isinstance(fn, ast.FunctionDef) or fn.name != "render_worlds":
                    continue
                params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
                assigned = {
                    target.id
                    for node in ast.walk(fn)
                    if isinstance(node, ast.Assign)
                    for target in node.targets
                    if isinstance(target, ast.Name)
                }
                clash = sorted(params & assigned - {"self"})
                with self.subTest(backend=path.name):
                    self.assertEqual(clash, [], f"{path.name}: render_worlds assigns its own parameter(s) {clash}")


class TestRenderSpec(unittest.TestCase):
    """One lighting request, five backends that agree on none of it.

    The spec's whole contract is that *unstated means untouched*: the five
    backends' defaults are five different values, not one value written five
    ways, so a spec that filled in neutral-looking numbers would change every
    render in the package the moment it was introduced.
    """

    def test_an_unstated_spec_asks_every_backend_for_nothing(self):
        """The property that makes this safe to introduce at all."""
        spec = RenderSpec()
        for backend in list_renderers():
            self.assertEqual(look_kwargs(backend, spec), {}, backend)

    def test_each_backend_gets_only_what_it_can_honour(self):
        """Translation, not forwarding: the field names differ per backend."""
        spec = RenderSpec(
            light_direction=(0.0, 0.0, -1.0),
            key_intensity=3.0,
            ambient=(0.1, 0.2, 0.3),
            shadows=False,
            environment="studio",
        )
        self.assertEqual(look_kwargs("tiled", spec), {"light_direction": (0.0, 0.0, -1.0), "shadows": False})
        # RTX takes the preset, and the key and ambient that its ``lab`` preset
        # scales its panel and dome by; the directional rig means nothing to it.
        self.assertEqual(
            look_kwargs("rtx", spec), {"environment": "studio", "key_intensity": 3.0, "ambient": (0.1, 0.2, 0.3)}
        )

    def test_a_background_reaches_the_backend_that_can_fill_a_miss(self):
        """The tracer packs a clear colour; rtx bakes an environment and has no colour to hand."""
        spec = RenderSpec(background=(0.2, 0.3, 0.4))
        self.assertEqual(look_kwargs("tiled", spec)["background"], (0.2, 0.3, 0.4))
        self.assertNotIn("background", look_kwargs("rtx", spec))

    def test_the_house_look_is_what_an_unstated_scene_gets(self):
        """``look_for`` is where the all-None spec stops meaning "do nothing".

        The distinction it draws is the load-bearing one: a scene that says
        nothing gets the house look, and a caller that genuinely wants each
        backend's own defaults still has a way to say so -- a bare
        ``RenderSpec()``, which is not the same statement as silence.
        """
        from robosimstudio.core.render.look import STUDIO, look_for
        from robosimstudio.core.render.room import HANGAR_CLOSED, VOID

        self.assertIs(look_for(None), HANGAR_CLOSED)
        mine = RenderSpec(shadows=False)
        self.assertIs(look_for(mine), mine)
        for backend in ("tiled", "rtx"):
            with self.subTest(backend=backend):
                self.assertEqual(look_kwargs(backend, look_for(RenderSpec())), {})
        # And the old default is still sayable, by name.
        self.assertIs(VOID, STUDIO)

    def test_the_default_room_is_invisible_to_every_backend_but_the_path_tracer(self):
        """The whole safety argument for shipping a room by default, as an assertion.

        A backdrop that reached the tiled tracer would change what a policy is
        evaluated on, which is the one thing a change to the *picture* may not
        do. The room is one field -- ``scenery``, which only ``rtx`` reads -- so
        the four other backends are handed byte-identical arguments before and
        after. That is checked here rather than argued, because the argument is
        only as good as the field list it rests on and a fifth field could be
        added to ``RenderSpec`` tomorrow.

        The exact-set assertion on the ``rtx`` side is the other half: the house
        room deliberately states *nothing else*, no lighting preset and no
        environment map, so that adding a room to every scene cannot also change
        how any of them is lit. A room that grew a second field would fail here
        and should.
        """
        from robosimstudio.core.render.look import STUDIO, look_for
        from robosimstudio.core.render.room import HANGAR_CLOSED

        self.assertIsNot(look_for(None), STUDIO, "the default is a room, or this test is vacuous")
        self.assertEqual(look_kwargs("tiled", HANGAR_CLOSED), look_kwargs("tiled", STUDIO))
        self.assertEqual(set(look_kwargs("rtx", HANGAR_CLOSED)) - set(look_kwargs("rtx", STUDIO)), {"scenery"})

    def test_a_tiled_background_packs_little_endian_with_red_low(self):
        """Getting the byte order wrong gives a plausible colour, not an error."""
        from robosimstudio.solvers.render.tiled import _packed_rgba

        self.assertEqual(_packed_rgba((0.0, 0.0, 0.0)), 0xFF000000)
        self.assertEqual(_packed_rgba((1.0, 0.0, 0.0)), 0xFF0000FF)
        self.assertEqual(_packed_rgba((0.0, 0.0, 1.0)), 0xFFFF0000)

    def test_explicit_kwargs_beat_the_spec(self):
        """A kwarg names one renderer; the spec describes the scene."""
        spec = RenderSpec(shadows=True)
        self.assertEqual(merge_render_kwargs(backend_class("tiled"), spec, {"shadows": False}), {"shadows": False})
        self.assertEqual(merge_render_kwargs(backend_class("tiled"), None, {"shadows": False}), {"shadows": False})


class TestRTXFilmBack(unittest.TestCase):
    """The USD camera a path-traced rig is authored with.

    The same round trip as the tiled tracer's, against a different
    parameterisation -- a film back in millimetres rather than film apertures in
    whatever unit cancels -- because it is the same claim: an explicitly
    calibrated camera survives the trip into the renderer *including* its
    principal point, which the shared viewport this replaced could not express
    at all.
    """

    def test_film_back_round_trip_is_sub_pixel(self):
        spec = CameraSpec(name="c", width=640, height=480, fx=525.3, fy=524.1, cx=317.2, cy=238.9)
        film = usd_film_back(spec)
        f, w, h = film["focal_length"], spec.width, spec.height
        h_ap, v_ap = film["horizontal_aperture"], film["vertical_aperture"]
        fx = w * f / h_ap
        fy = h * f / v_ap
        cx = w * 0.5 - 0.5 - film["horizontal_aperture_offset"] * w / h_ap
        cy = film["vertical_aperture_offset"] * h / v_ap + h * 0.5 - 0.5
        k = spec.intrinsics
        for got, want, name in ((fx, k[0, 0], "fx"), (fy, k[1, 1], "fy"), (cx, k[0, 2], "cx"), (cy, k[1, 2], "cy")):
            self.assertLess(abs(got - want), 1e-3, f"{name}: {got} != {want}")

    def test_a_centred_camera_is_offset_by_half_a_pixel(self):
        """Not a rounding artefact: it is the convention gap, and it is signed.

        USD projects onto a continuous image coordinate where pixel *i*'s centre
        sits at ``i + 0.5``; a ``K`` is written for the index. So the film back
        of a perfectly centred camera is *not* zero-offset, and the two halves
        run opposite ways -- the horizontal offset slides the film against the
        principal point and the vertical one with it. Measured against the tiled
        tracer, getting this wrong lands the two backends 1.00 px apart.
        """
        film = usd_film_back(CameraSpec(name="c", width=640, height=480))
        self.assertAlmostEqual(film["horizontal_aperture_offset"] * 640 / film["horizontal_aperture"], -0.5)
        self.assertAlmostEqual(film["vertical_aperture_offset"] * 480 / film["vertical_aperture"], 0.5)


class TestTiledIntrinsics(unittest.TestCase):
    def test_aperture_round_trip_is_sub_pixel(self):
        """K -> film apertures -> K reproduces the intrinsics to well under a pixel.

        The half-pixel terms in the conversion are what this checks: the ray
        helper samples pixel centres, so dropping them shifts every projection
        by half a pixel -- an error that survives a visual check and fails a
        calibration one.
        """
        spec = CameraSpec(name="c", width=640, height=480, fx=525.3, fy=524.1, cx=317.2, cy=238.9)
        p = aperture_from_intrinsics(spec)
        f, w, h = p["focal_length"], spec.width, spec.height
        fx = f * w / p["horizontal_aperture"]
        fy = f * h / p["vertical_aperture"]
        cx = w * 0.5 - 0.5 - fx * p["horizontal_aperture_offset"] / f
        cy = h * 0.5 - 0.5 + fy * p["vertical_aperture_offset"] / f
        k = spec.intrinsics
        for got, want, name in ((fx, k[0, 0], "fx"), (fy, k[1, 1], "fy"), (cx, k[0, 2], "cx"), (cy, k[1, 2], "cy")):
            self.assertLess(abs(got - want), 1e-3, f"{name}: {got} != {want}")


class TestACameraFindsItsOwnWorldsParent(unittest.TestCase):
    """Which body a camera hangs off when the model holds many copies of it.

    A rig bolted to the room is the same rig in every world -- a wave's worlds
    are superimposed at the origin, so they all see the same place. A rig bolted
    to a *wrist* is not, and its parent index is the template's, so without an
    offset every world is photographed through world 0's hand.

    That failure is silent and is worst exactly where it matters: an eye-in-hand
    view is the one a grasp is judged from, so a wave would score sixteen grasps
    off one hand's picture and read as a perception problem.
    """

    @staticmethod
    def _body_q(bodies_per_world: int, worlds: int) -> np.ndarray:
        """Identity rotations, one body per world at ``x = world``."""
        rows = []
        for world in range(worlds):
            for body in range(bodies_per_world):
                rows.append([float(world), float(body), 0.0, 0.0, 0.0, 0.0, 1.0])
        return np.asarray(rows, dtype=np.float64)

    def test_a_world_fixed_camera_ignores_the_offset(self):
        camera = CameraSpec(name="overhead", width=8, height=8)
        body_q = self._body_q(bodies_per_world=2, worlds=3)
        first = camera.pose_on_body(body_q)
        for world in range(3):
            with self.subTest(world=world):
                self.assertTrue(np.allclose(camera.pose_on_body(body_q, body_offset=world * 2), first))

    def test_a_body_parented_camera_follows_its_own_world(self):
        """Each world's parent sits one metre further along x; so does its camera.

        Stated as the *difference* between worlds rather than as an absolute,
        because the camera's own local matrix is in the answer too and asserting
        the composition would be asserting ``pose_on_body``'s arithmetic against
        itself.
        """
        camera = CameraSpec(name="eye_in_hand", width=8, height=8, parent_body=1)
        body_q = self._body_q(bodies_per_world=2, worlds=3)
        poses = [camera.pose_on_body(body_q, body_offset=world * 2) for world in range(3)]
        for world in range(1, 3):
            with self.subTest(world=world):
                shift = poses[world] - poses[0]
                self.assertAlmostEqual(float(shift[0, 3]), float(world), places=6)
                self.assertAlmostEqual(float(shift[1, 3]), 0.0, places=6)
                self.assertTrue(
                    np.allclose(poses[world][:3, :3], poses[0][:3, :3]),
                    "the worlds are superimposed, so only the translation may differ",
                )

    def test_no_offset_is_world_zero(self):
        """The default has to be the single-world answer, or every caller changes."""
        camera = CameraSpec(name="eye_in_hand", width=8, height=8, parent_body=1)
        body_q = self._body_q(bodies_per_world=2, worlds=3)
        self.assertTrue(np.allclose(camera.pose_on_body(body_q), camera.pose_on_body(body_q, body_offset=0)))


if __name__ == "__main__":
    unittest.main()
