# SPDX-License-Identifier: Apache-2.0
"""``robosimstudio_real2sim_agent``'s page itself: a scene file built on a GPU and driven the way ``ctl`` does.

:mod:`tests.test_agent` is the half with no page. This is the other half, with
the loop driven a pass at a time rather than by ``serve`` -- so a test can say
what happens *between* two passes, which is where every command lands:

* a routine steps the scene once per pass whatever the Pause checkbox says,
  and is filmed while it does; ``run N`` is refused while it plays;
* both Reset buttons stop the routine and leave Pause where it was;
* a file that builds and then fails to go into the page, and a panel whose
  drain raises, leave the page serving and say why;
* a helper module beside the scene file is read again on a rebuild, and a
  name ``eval`` binds is one the file's routines see;
* the URL names the port viser actually took.
"""

from __future__ import annotations

import socket
import tempfile
import types
import unittest
from pathlib import Path

TIER = "medium"
"""Builds scenes and starts a viser server on a spare port."""

PORT = 9331
"""Held by a plain socket for the whole class, so the page has to take the next one up."""

SCENE = """import robosimstudio as rss
from helper import SIZE

STEPS = 10
scene = rss.Scene(cell="void", name="live-test")
scene.add(rss.box(name="cube", half=(SIZE,) * 3, at=(0.0, 0.0, 0.1)))


def routine(sim):
    for _ in range(STEPS):
        sim.run(1)
        yield
"""


class TestThePage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from robosimstudio_real2sim_agent.server import LiveServer
        from robosimstudio_studio.session import StudioConfig

        cls.blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cls.blocker.bind(("0.0.0.0", PORT))
        cls.blocker.listen(1)
        cls.tmp = Path(tempfile.mkdtemp(prefix="live-"))
        (cls.tmp / "helper.py").write_text("SIZE = 0.02\n")
        (cls.tmp / "scene.py").write_text(SCENE)
        config = StudioConfig(port=PORT, panels="all", render_every=1, live=False, out=str(cls.tmp / "studio"))
        cls.live = LiveServer(
            scene_path=cls.tmp / "scene.py",
            sock_path=cls.tmp / "ctl.sock",
            run_dir=cls.tmp / "run",
            config=config,
            watch=False,
        )
        cls.live.open_page()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.live.bus.close()
        cls.live.session.close()
        cls.blocker.close()

    def setUp(self) -> None:
        self.live.reload("next test")  # the file's own names back, nothing playing, landed running
        self.assertFalse(self.live.session.paused)

    def play_out(self, passes: int = 200) -> None:
        for _ in range(passes):
            self.live.tick_once()
            if not self.live.script:
                return
        self.fail("the routine did not finish")

    # ------------------------------------------------------------ the port
    def test_the_url_names_the_port_viser_took(self) -> None:
        port = self.live.session.server.get_port()
        self.assertNotEqual(port, PORT, "the port asked for is held; viser should have taken another")
        self.assertEqual(self.live.url, f"http://localhost:{port}")
        self.assertEqual(self.live.cmd_status()["url"], self.live.url)

    # ------------------------------------------------------------ routines
    def test_a_routine_steps_once_per_pass_whatever_pause_says(self) -> None:
        sim = self.live.sim
        start = int(sim.step_count)
        frames = len(self.live.session.video.buffer)
        self.assertTrue(self.live.cmd_play("routine")["ok"])
        self.assertFalse(self.live.session.paused, "Pause is left as it was")
        refused = self.live.cmd_run(5)
        self.assertFalse(refused["ok"])
        self.assertIn("routine is playing", refused["error"])
        self.play_out()
        # Ten from the routine, and the running page's own one on the pass the
        # routine ended in; stepping beside it would have made it twenty.
        self.assertEqual(int(sim.step_count) - start, 10 + 1, "the page stepped beside the routine")
        self.assertGreater(len(self.live.session.video.buffer), frames, "nothing was filmed while it played")
        self.assertFalse(self.live.session.paused, "and the page carries on as it was")

    def test_the_live_reset_stops_a_routine_and_leaves_pause_alone(self) -> None:
        self.live.cmd_play("routine")
        self.live.tick_once()
        self.live.cmd_reset()
        self.live.tick_once()
        self.assertEqual(self.live.script, [])
        self.assertFalse(self.live.session.paused, "a reset mid-routine left the page paused")
        self.assertLessEqual(int(self.live.sim.step_count), 1, "back at the start, bar the running page's own step")

    def test_the_simulation_folders_reset_does_the_same(self) -> None:
        self.live.cmd_play("routine")
        self.live.tick_once()
        self.live.session.reset_requested = True  # what that button's callback does
        self.live.tick_once()
        self.assertEqual(self.live.script, [], "the two Reset buttons differ")

    def test_a_name_eval_binds_is_one_the_routine_reads(self) -> None:
        self.assertTrue(self.live.cmd_eval(code="STEPS = 3")["ok"])
        start = int(self.live.sim.step_count)
        self.live.cmd_play("routine")
        self.play_out()
        self.assertEqual(int(self.live.sim.step_count) - start, 3 + 1, "the routine read the file's STEPS, not eval's")

    # ------------------------------------------------------------ rebuilds
    def test_a_helper_edit_lands_on_the_next_rebuild(self) -> None:
        (self.tmp / "helper.py").write_text("SIZE = 0.0375\n")  # a different length, so no stale .pyc can pass
        try:
            before = self.live.generation
            self.assertTrue(self.live.reload("test")["ok"])
            self.assertEqual(self.live.ns["SIZE"], 0.0375)
            self.assertEqual(self.live.generation, before + 1, "a run in flight is told by the generation")
        finally:
            (self.tmp / "helper.py").write_text("SIZE = 0.02\n")
            self.live.reload("test")

    def test_a_file_that_will_not_go_into_the_page_leaves_it_serving(self) -> None:
        session = self.live.session

        def broken(self_, sim, **kwargs):
            del self_.adopt  # once
            self_.teardown()  # what a real failure half way through has already done
            raise RuntimeError("adopt broke")

        session.adopt = types.MethodType(broken, session)
        reply = self.live.reload("test")
        self.assertFalse(reply["ok"])
        self.assertIn("adopt broke", reply["error"])
        self.assertIn("adopt broke", self.live.cmd_status()["last_error"])
        self.assertEqual(self.live.scene.name, "live-fallback", "a bare scene stands in")
        for _ in range(3):
            self.live.tick_once()
        self.assertTrue(self.live.reload("test")["ok"], "and the next good save goes in")
        self.assertEqual(self.live.last_error, "")

    def test_a_drain_that_raises_is_reported_not_fatal(self) -> None:
        from robosimstudio_studio.panels import Panel

        class Broken(Panel):
            def drain(self) -> None:
                raise ValueError("drain broke")

        panel = Broken(self.live.session)
        self.live.session.panels.append(panel)
        try:
            for _ in range(3):
                self.live.tick_once()
            self.assertIn("drain broke", self.live.last_error)
        finally:
            self.live.session.panels.remove(panel)
        self.live.tick_once()


class TestAResetThatFails(unittest.TestCase):
    """The studio's own half: a reset that raises pauses through the checkbox and says so on the page."""

    def test_the_checkbox_and_the_status_say_it(self) -> None:
        import robosimstudio as rss
        from robosimstudio_studio.session import StudioConfig, StudioSession

        scene = rss.Scene(cell="void", name="reset-fails")
        scene.add(rss.box(name="cube", half=(0.02,) * 3, at=(0.0, 0.0, 0.1)))
        session = StudioSession(scene.build(), config=StudioConfig(port=9332, panels="drive"))
        try:

            def boom():
                raise RuntimeError("no episode")

            session.reset_env = boom
            session.reset_requested = True
            self.assertFalse(session.tick())
            self.assertTrue(session.paused)
            self.assertTrue(session.pause_box.value, "the checkbox does not say the page paused")
            self.assertIn("did not load", session._status_readout.value)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
