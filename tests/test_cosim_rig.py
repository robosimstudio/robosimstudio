# SPDX-License-Identifier: Apache-2.0
"""The tick's state machine, with no engine under it.

``CosimRig`` is what is left of a co-simulation tick once the engine calls are
taken out: which poses this step reads, how a mirror is swept across the
engine's substeps, when a sweep becomes a jump, and how the deformable state
lands back in Newton's particle arrays. All of it is checkable without an
engine, a GPU or a scene, and this is where that is done -- the SuperDex scenes
at the medium tier (``tests/test_superdex_contact.py`` among them) cover the same code with an engine in the loop but cannot say *why* a number
came out, only that it did.
"""

from __future__ import annotations

import unittest

import numpy as np

TIER = "fast"
"""Stubs all the way down."""

IDENT = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


class _Slice:
    def __init__(self, store, key):
        self._store, self._key = store, key

    def assign(self, values):
        self._store[self._key] = np.array(values, copy=True)


class _Array:
    """Enough of a warp array for ``q[start:start+count].assign(...)``."""

    def __init__(self, store, name):
        self._store, self._name = store, name

    def __getitem__(self, key):
        return _Slice(self._store, (self._name, key.start, key.stop))


class _State:
    def __init__(self, body_q=None, store=None):
        self.store = {} if store is None else store
        self.particle_q = _Array(self.store, "q")
        self.particle_qd = _Array(self.store, "qd")
        self._body_q = body_q

    @property
    def body_q(self):
        outer = self

        class _BQ:
            def numpy(self):
                return outer._body_q

        return _BQ()


class _Pipeline:
    def __init__(self, state_0, state_1, snapshot=None, settling=False):
        self.state_0, self.state_1 = state_0, state_1
        self.snapshot, self.settling = snapshot, settling


class _Snapshot:
    def __init__(self, body_q):
        self.body_q = body_q


class _Mirror:
    def __init__(self, body, local=None):
        self.body = body
        self.local = IDENT.copy() if local is None else local


def _rig(bodies=(2,), **kw):
    from robosimstudio.engines.cosim import CosimRig

    return CosimRig(bodies, **kw)


class TestPoseSource(unittest.TestCase):
    def test_fresh_reads_this_step_off_the_state(self):
        body_q = np.tile(IDENT, (4, 1))
        body_q[2, 0] = 0.5
        rig = _rig((2,), pose_source="fresh")
        self.assertAlmostEqual(rig.body_poses(_State(body_q))[2][0], 0.5)

    def test_snapshot_prefers_the_batched_host_copy(self):
        fresh = np.tile(IDENT, (4, 1))
        fresh[2, 0] = 0.5
        stale = np.tile(IDENT, (4, 1))
        stale[2, 0] = 0.1
        rig = _rig((2,), pose_source="snapshot")
        rig.attach_pipeline(_Pipeline(_State(), _State(), snapshot=_Snapshot(stale)))
        self.assertAlmostEqual(rig.body_poses(_State(fresh))[2][0], 0.1, msg="a control period stale, by choice")

    def test_snapshot_falls_back_while_settling(self):
        """A settle steps outside the read-back, so the snapshot is not there yet."""
        fresh = np.tile(IDENT, (4, 1))
        fresh[2, 0] = 0.5
        stale = np.tile(IDENT, (4, 1))
        stale[2, 0] = 0.1
        rig = _rig((2,), pose_source="snapshot")
        rig.attach_pipeline(_Pipeline(_State(), _State(), snapshot=_Snapshot(stale), settling=True))
        self.assertAlmostEqual(rig.body_poses(_State(fresh))[2][0], 0.5)

    def test_no_mirrored_bodies_reads_nothing(self):
        """A deformable-only island must not pay a device read for no bodies."""
        self.assertEqual(_rig(()).body_poses(_State(None)), {})

    def test_world_fixed_bodies_are_not_tracked(self):
        self.assertEqual(_rig((-1, 3, -1)).bodies, (3,))


class TestMirrorSweep(unittest.TestCase):
    def _targets(self, x):
        pose = IDENT.copy()
        pose[0] = x
        return {2: pose}

    def test_it_yields_one_batch_per_substep_ending_on_target(self):
        rig = _rig(substeps=4)
        rig.commit(self._targets(0.0))  # a previous step, so this is a sweep
        mirrors = [_Mirror(2)]
        batches = list(rig.substeps_of(mirrors, self._targets(1.0)))
        self.assertEqual([i for i, _ in batches], [0, 1, 2, 3])
        xs = [poses[0][1][0] for _, poses in batches]
        np.testing.assert_allclose(xs, [0.25, 0.5, 0.75, 1.0])

    def test_the_first_tick_jumps_rather_than_sweeps(self):
        """No previous pose exists, so interpolating from one would sweep the
        mirror across the scene and through whatever is in the way."""
        rig = _rig(substeps=4)
        self.assertTrue(rig.teleport)
        batches = list(rig.substeps_of([_Mirror(2)], self._targets(1.0)))
        xs = [poses[0][1][0] for _, poses in batches]
        np.testing.assert_allclose(xs, [1.0, 1.0, 1.0, 1.0], err_msg="every substep is already at target")

    def test_commit_ends_the_teleport_and_arms_the_next_sweep(self):
        rig = _rig(substeps=2)
        list(rig.substeps_of([_Mirror(2)], self._targets(1.0)))
        rig.commit(self._targets(1.0))
        self.assertFalse(rig.teleport)
        xs = [poses[0][1][0] for _, poses in rig.substeps_of([_Mirror(2)], self._targets(2.0))]
        np.testing.assert_allclose(xs, [1.5, 2.0], err_msg="now it sweeps from where it was")

    def test_mark_teleport_jumps_again(self):
        rig = _rig(substeps=2)
        rig.commit(self._targets(0.0))
        rig.mark_teleport()
        xs = [poses[0][1][0] for _, poses in rig.substeps_of([_Mirror(2)], self._targets(9.0))]
        np.testing.assert_allclose(xs, [9.0, 9.0], err_msg="a reset must not drag the mirror across")

    def test_a_world_fixed_mirror_is_skipped(self):
        """It was posed once at build; a tick has nothing to do for it."""
        rig = _rig(substeps=1)
        batches = list(rig.substeps_of([_Mirror(-1), _Mirror(2)], self._targets(1.0)))
        self.assertEqual(len(batches[0][1]), 1)
        self.assertEqual(batches[0][1][0][0].body, 2)

    def test_the_body_local_transform_is_composed_in(self):
        rig = _rig(substeps=1)
        local = IDENT.copy()
        local[1] = 0.3
        _, poses = next(iter(rig.substeps_of([_Mirror(2, local)], self._targets(1.0))))
        np.testing.assert_allclose(poses[0][1][:3], [1.0, 0.3, 0.0])

    def test_substep_dt_divides_the_control_period(self):
        self.assertAlmostEqual(_rig(substeps=4).substep_dt(1.0 / 60.0), 1.0 / 240.0)


class TestParticleWriteBack(unittest.TestCase):
    def test_it_writes_both_halves_of_the_double_buffer(self):
        """The tick runs outside the captured graph; writing one state lets the
        swap resurrect a stale copy on the next step."""
        rig = _rig(())
        s0, s1 = _State(), _State()
        rig.attach_pipeline(_Pipeline(s0, s1))
        pos = np.arange(9, dtype=np.float64).reshape(3, 3)
        rig.write_particles(_State(), 4, 3, pos, 1.0 / 60.0, np.zeros((3, 3)))
        for store in (s0.store, s1.store):
            np.testing.assert_allclose(store[("q", 4, 7)], pos.astype(np.float32))

    def test_velocity_is_finite_differenced_against_the_last_tick(self):
        rig = _rig(())
        s0 = _State()
        rig.attach_pipeline(_Pipeline(s0, _State()))
        prev = np.zeros((2, 3))
        pos = np.full((2, 3), 0.5)
        got = rig.write_particles(_State(), 0, 2, pos, 0.5, prev)
        np.testing.assert_allclose(s0.store[("qd", 0, 2)], np.full((2, 3), 1.0), atol=1e-6)
        np.testing.assert_allclose(got, pos, err_msg="returns the positions to hold as next prev")

    def test_a_zero_dt_does_not_divide_by_it(self):
        rig = _rig(())
        s0 = _State()
        rig.attach_pipeline(_Pipeline(s0, _State()))
        rig.write_particles(_State(), 0, 1, np.ones((1, 3)), 0.0, np.zeros((1, 3)))
        np.testing.assert_allclose(s0.store[("qd", 0, 1)], np.zeros((1, 3)))

    def test_states_precedence(self):
        rig = _rig(())
        bare = _State()
        self.assertEqual(rig.states(bare), (bare,), "with nothing attached, the handed state")
        a, b = _State(), _State()
        rig.attach_states((a, b))
        self.assertEqual(rig.states(bare), (a, b), "the repro runner's pair")
        s0, s1 = _State(), _State()
        rig.attach_pipeline(_Pipeline(s0, s1))
        self.assertEqual(rig.states(bare), (s0, s1), "a pipeline outranks both")


if __name__ == "__main__":
    unittest.main()
