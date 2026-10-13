# SPDX-License-Identifier: Apache-2.0
"""The ``superdex`` backend class: a host-side engine in a solver's clothes."""

from __future__ import annotations

from typing import Any

from robosimstudio.engines.host import SolverHost

__all__ = ["SolverSuperdex"]


class SolverSuperdex(SolverHost):
    """The host island's solver for SuperDex: opens the bridge, inherits the rest.

    Args:
        model: The finalized Newton model.
        recipe: The manifest's host-island record -- see
            :class:`~robosimstudio.engines.superdex.bridge.SuperdexBridge`.
        substeps: SuperDex steps per control step.
        threads: Worker threads for the engine's one-per-process initialize.
        feedback: ``"auto"`` or ``"off"`` -- contact-wrench harvest for
            two-way mirrors.
        feedback_alpha: EMA coefficient for a *rising* harvested force; 1.0
            disables the ramp. See
            :class:`~robosimstudio.engines.feedback.WrenchAccumulator`.
        feedback_dv_max: The most velocity [m/s] one control step's harvested
            wrench may add to a body; scales the whole wrench, torque with it.
        pose_source: ``"fresh"`` reads this step's poses off the state (zero
            coupling lag, one extra small device read); ``"snapshot"`` reads
            the free batched host copy, one control period stale.
        shell_young: Shell (cloth) elastic modulus [Pa], island-wide, for
            entities the bridge builds as shell actors.
        shell_poisson: Shell Poisson ratio, island-wide.
        mirror_smoothing: ``penalty_smoothing_half_distance`` [m] on the
            mirrored colliders -- the knob that decides whether a gripper
            presses a deformable or sinks into it. See
            :func:`~robosimstudio.engines.superdex.mirror.build_mirror_actors`.
        mirror_friction_falloff: ``friction_falloff_vel`` [m/s] on the
            mirrors. It reaches no contact: against a static collider, which
            every mirror is, SuperDex uses the deformable's own falloff, so a
            hold is set by ``friction_falloff`` on the deformable. See
            :func:`~robosimstudio.engines.superdex.mirror.build_mirror_actors`.
        mirror_threshold: ``penalty_threshold_default`` [m] on the mirrors, or
            ``None`` for the engine's 1 mm -- the stand-off at which a pad starts
            pushing, which the collider's side of a pair decides.
        mirror_padding: ``penalty_threshold_extra_padding`` [m] on the
            mirrors: a stand-off the engine adds against a point-cloud actor
            only. A rod built without its skin rests with its axis on a mirror;
            padding by about its radius lifts it onto its surface.
        mirror_exact_boxes: Mirror boxes as the engine's exact ``BOX`` collider.
        linear_solver: The engine's linear solver by name (``"cg"``,
            ``"minres"``, ``"ldlt"``...), or ``None`` for its own choice.
        preconditioner: Its preconditioner by name (``"block_jacobi"``...).
        newton_iterations: Non-linear iterations per engine step, or ``None``
            for the engine's 4.
        linear_iterations: A cap on the linear solver's iterations per Newton
            step, or ``None`` for the engine's own (it ran 1000 per contact).
    """

    def open(
        self,
        model: Any,
        recipe: dict,
        *,
        substeps: int = 1,
        threads: int = -1,
        feedback: str = "auto",
        feedback_alpha: float = 0.35,
        feedback_dv_max: float = 0.5,
        pose_source: str = "fresh",
        shell_young: float = 1.0e5,
        shell_poisson: float = 0.3,
        mirror_smoothing: float = 5.0e-4,
        mirror_friction_falloff: float = 1.0e-2,
        mirror_threshold: float | None = None,
        mirror_padding: float = 0.0,
        mirror_exact_boxes: bool = True,
        linear_solver: str | None = None,
        preconditioner: str | None = None,
        newton_iterations: int | None = None,
        linear_iterations: int | None = None,
    ) -> Any:
        """The SuperDex bridge, with the island-wide scalars folded into the recipe.

        They ride the recipe rather than an argument because that is what a
        repro bundle carries: the manifest records the recipe, and a twin
        rebuilds the same actors from it and the model alone.
        """
        from .bridge import SuperdexBridge

        return SuperdexBridge(
            model,
            {
                **recipe,
                "feedback_alpha": float(feedback_alpha),
                "feedback_dv_max": float(feedback_dv_max),
                "shell_young": float(shell_young),
                "shell_poisson": float(shell_poisson),
            },
            substeps=int(substeps),
            threads=int(threads),
            feedback=str(feedback),
            pose_source=str(pose_source),
            mirror_smoothing=float(mirror_smoothing),
            mirror_friction_falloff=float(mirror_friction_falloff),
            mirror_threshold=None if mirror_threshold is None else float(mirror_threshold),
            mirror_padding=float(mirror_padding),
            mirror_exact_boxes=bool(mirror_exact_boxes),
            linear_solver=linear_solver,
            preconditioner=preconditioner,
            newton_iterations=newton_iterations,
            linear_iterations=linear_iterations,
        )
