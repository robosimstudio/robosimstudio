# SPDX-License-Identifier: Apache-2.0
"""A new kind of thing

A chain of loose links the library has never heard of, defined in this file: click any link and drag it.

Everything a kind is made of is here, and nothing under ``robosimstudio/`` was
edited to run it: a spec, an entity that builds it, a ``Ref`` with an accessor
of its own, a kernel that runs every substep, and the registration. The page
shows the chain's span, read through that accessor.

**Key features:**

* :func:`robosimstudio.register_domain` to declare the kind
* :func:`robosimstudio.register_entity` and :class:`robosimstudio.Entity` for what builds it
* :meth:`robosimstudio.Entity.handle` and :meth:`robosimstudio.Entity.read` so the page and the measures can reach it
* :meth:`robosimstudio.Entity.contribute_hooks` for a kernel in the step
* :class:`robosimstudio.Ref` subclassed for an accessor, and ``page=`` on :func:`robosimstudio.view` for a readout
"""

from dataclasses import dataclass, field
from typing import Any

import newton
import numpy as np
import warp as wp

import robosimstudio as rss


@wp.kernel
def _drag_kernel(body_qd: wp.array(dtype=wp.spatial_vector), indices: wp.array(dtype=int), damping: float):
    """Viscous drag on every link, as if the chain were falling through syrup.

    ``body_qd`` is the whole model's, robot included, so the kernel visits only
    the chain's own bodies.
    """
    b = indices[wp.tid()]
    body_qd[b] = body_qd[b] * damping


@dataclass(frozen=True, kw_only=True)
class ChainSpec:
    """A plain frozen dataclass. No library base class, and none needed."""

    entity: str = "my_chain"
    name: str = "chain"
    links: int = 24
    half: float = 0.012
    at: tuple[float, float, float] = (0.0, 0.0, 0.35)  # the bottom link, in the work-surface frame
    damping: float = 0.985
    sim: Any = field(
        default_factory=lambda: rss.SimRequirements(
            domain="chain", contacts=rss.contacts_with("surface", "fixtures", "objects")
        )
    )


class ChainRef(rss.Ref):
    """Your accessors, not the library's: ``Entity.ref`` says what a thing offers."""

    def span(self) -> float:
        """End to end [m], off the handle the entity published.

        This read ``pipeline.state_0.body_q`` directly, which is the model's live
        array rather than the batched read-back the rest of the scene shares --
        a number that is right and arrives by a private road. It reads the
        handle now, which is the same road every other accessor takes.
        """
        points = np.asarray(self.pose_of_each())[:, :3]
        return float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0

    def pose_of_each(self) -> np.ndarray:
        handle = self.handle
        return self._scene.pipeline.snapshot.body_q[list(handle.bodies)]


@rss.register_entity("my_chain")
class MyChain(rss.Entity):
    """A stack of free boxes that damp themselves every substep."""

    domain = "chain"
    ref = ChainRef

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        spec = self.spec
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 400.0
        self.bodies: list[int] = []
        self.shapes: list[int] = []
        for i in range(spec.links):
            pose = self.placed(np.array([*spec.at, 0.0, 0.0, 0.0, 1.0], dtype=np.float64))
            pose[2] += i * 2.2 * spec.half
            body = builder.add_body(
                xform=wp.transform(tuple(float(v) for v in pose[:3]), wp.quat(*[float(v) for v in pose[3:7]])),
                label=f"{spec.name}_{i:02d}",
            )
            shape = builder.add_shape_box(
                body, hx=spec.half, hy=spec.half, hz=spec.half, cfg=cfg, label=f"{spec.name}_{i:02d}"
            )
            self.bodies.append(body)
            self.shapes.append(shape)

    def handle(self, model):
        """What the scene publishes about this chain.

        Offering one is the whole of what makes a kind first-class: the views
        derive from it, so a chain that publishes a handle is an *object* -- it
        is clickable, draggable, describable and scoreable on the same terms as
        a box. Declining is still legal and still means "reachable through the
        inspector, not the click map"; it is a statement now rather than a bin
        somebody had to pick.
        """
        del model
        from robosimstudio.scene.handles import EntityHandle

        bodies = tuple(int(b) for b in self.bodies)
        return EntityHandle(name=self.name, body=bodies[0], bodies=bodies, shapes=self.shapes) if bodies else None

    def read(self, form: str, snapshot):
        """Where this chain's state is, for anything that measures it.

        A layout names the *shape* of state, not a place to find one. The chain
        says where its points are and every reader -- a goal, a description, the
        page -- stops guessing.
        """
        if form != "points":
            return None
        return np.asarray(snapshot.body_q[list(self.bodies)][:, :3], dtype=np.float64)

    def contribute_hooks(self, hooks) -> None:
        """A device kernel in a named phase, at a cadence, from outside the library."""
        hooks.add("post_solve", self._drag, name="chain-drag", cadence=1)

    def _drag(self, state, dt) -> None:
        if not self.bodies:
            return
        indices = getattr(self, "_drag_indices", None)
        if indices is None or indices.device != state.body_qd.device:
            indices = self._drag_indices = wp.array(self.bodies, dtype=int, device=state.body_qd.device)
        wp.launch(
            _drag_kernel,
            dim=len(self.bodies),
            inputs=[state.body_qd, indices, float(self.spec.damping)],
            device=indices.device,
        )


def main():
    cell = "void"
    embodiment = "yam_bimanual"
    links = 24

    # A kind nobody here has heard of, composed on exactly the terms ``cloth``
    # is: it declares its role, what it touches by default, and that it is not
    # welded to the world.
    from robosimstudio.specs.physics import DOMAINS

    if "chain" not in DOMAINS:
        rss.register_domain(
            "chain",
            state=("bodies", "joints", "shapes"),
            reads="rigid",
            role="objects",
            contacts=("surface", "fixtures", "objects"),
            summary="A stack of loose links.",
        )

    scene = rss.Scene(cell=cell, name="custom")
    scene.add(rss.robot(embodiment))
    chain = scene.add(ChainSpec(links=links))
    sim = scene.build(control_mode="joint_position")

    # The page reaches a kind it has never heard of on exactly the terms it
    # reaches a box: the chain is a rigid island, so its links are clickable and
    # every one of them is draggable, and the arms can be driven into it. The
    # accessor below is this file's own -- ``ChainRef.span`` is not a library
    # method -- and ``page`` is where a script puts its own readouts, so the
    # panels stay the studio's and this stays a number about *this* chain.
    def page(studio):
        span = studio.gui.add_number("span, end to end (mm)", 0.0, disabled=True)
        studio.each_step(lambda: setattr(span, "value", round(1000.0 * chain.span(), 1)))

    rss.view(sim, page=page)


if __name__ == "__main__":
    main()
