# SPDX-License-Identifier: Apache-2.0
"""Your own soft body

A second implementation of a soft body, written in this file: a block of tetrahedra dropped onto the bench beside a Panda.

``MySoft`` cuts a lattice into six tetrahedra per cell and hands them to
Newton's builder; the library's particle storage does the rest, so a click in
the studio pins a patch you can drag. The stock :class:`robosimstudio.SoftSpec`
describes the block, and its ``entity=`` field asks for this class. Newton's own
VBD solver steps it and no extra is needed: the one physics line drops the entry
that sends soft bodies to SuperDex. Limits: on VBD the hand passes into the
block rather than pressing it (30 mm of travel dents the top 3 mm);
``00_sponge.py`` is the press, on SuperDex.

**Key features:**

* :func:`robosimstudio.register_entity` to register an implementation under a name of its own
* :class:`robosimstudio.Entity` and its ``build``, where the tetrahedra go into Newton's builder
* ``ParticleMixin`` for bind, handle, reset and reading, shared by every kind made of particles
* :class:`robosimstudio.SoftSpec` with ``entity="my_soft"`` to say "use mine"
* :func:`robosimstudio.replace` on ``rss.GENERIC`` to keep the block on Newton
"""

import newton
import numpy as np
import warp as wp

import robosimstudio as rss
from robosimstudio.domains.capabilities import ParticleMixin

CELL_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
CELL_TETS = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])
"""One lattice cell as six tetrahedra round the diagonal from corner 0 to corner 6.

Every cell is cut the same way, so neighbours agree on the diagonal of the face
they share, and every tetrahedron is wound with a positive volume: Newton drops
the others without a word."""


# Loading this file twice in one process under the same module name is safe:
# the registry keeps the first copy of an identical class.
@rss.register_entity("my_soft")
class MySoft(ParticleMixin, rss.Entity):
    """A block of tetrahedra on a regular lattice."""

    domain = "soft"  # the scene's physics arrangement picks the solver for this kind

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        spec, foam = self.spec, self.spec.material
        counts = np.asarray(spec.cells) + 1  # vertices per axis

        # Lattice vertices in the block's own frame: ``pose`` is the middle of the bottom face.
        grid = np.stack(np.meshgrid(*[np.arange(n) for n in counts], indexing="ij"), axis=-1).reshape(-1, 3)
        vertices = grid * (np.asarray(spec.size) / np.asarray(spec.cells)) - (spec.size[0] / 2, spec.size[1] / 2, 0.0)
        # Each cell's eight corner indices, then its six tetrahedra.
        cells = np.stack(np.meshgrid(*[np.arange(n) for n in spec.cells], indexing="ij"), axis=-1).reshape(-1, 1, 3)
        corners = np.ravel_multi_index(tuple(np.moveaxis(cells + CELL_CORNERS, -1, 0)), counts)
        tets = corners[:, CELL_TETS].reshape(-1, 4)

        base = self.placed((*spec.pose, *spec.quat))
        mu, lam = foam.lame  # Young's modulus and Poisson ratio, as the Lame pair the solver stores
        builder.add_soft_mesh(
            pos=wp.vec3(*base[:3]),
            rot=wp.quat(*base[3:]),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*vertex) for vertex in vertices],
            indices=[int(index) for index in tets.ravel()],
            density=foam.density,
            k_mu=mu,
            k_lambda=lam,
            k_damp=foam.stiffness_damping * mu,
            particle_radius=foam.particle_radius,
        )

    def update_probe(self, state, step=None, *, readback=None):
        """This step's particle positions, for whoever observes the block."""
        self.probe.update(state, step)
        return self.probe

    def on_scene_ready(self, pipeline) -> None:
        """The scene can step: record the state a reset returns to."""
        self.snapshot_rest(pipeline)


def main() -> None:
    size = (0.10, 0.07, 0.04)  # [m]
    cells = (10, 7, 4)  # lattice cells per axis: 440 particles, 1680 tetrahedra
    drop = 0.10  # the bottom face starts this far above the bench [m]

    # ``GENERIC`` sends soft bodies to SuperDex, an extra. Without that entry the
    # block falls through to the wildcard, Newton's VBD, on a base install.
    physics = rss.replace(rss.GENERIC, backends=tuple(entry for entry in rss.GENERIC.backends if entry[0] != "soft"))

    scene = rss.Scene(cell="bench_075", physics=physics, name="your_own_soft")
    scene.add(rss.robot("panda"))
    block = scene.add(rss.SoftSpec(name="block", entity="my_soft", size=size, cells=cells, pose=(0.0, 0.0, drop)))
    sim = scene.build(control_mode="joint_position")

    thinnest = size[2]
    for _ in range(60):  # the first second one step at a time, to catch the landing
        sim.run(1).sync()
        column = block.particle_q[:, 2]
        thinnest = min(thinnest, column.max() - column.min())
    sim.run(120).sync()
    column = block.particle_q[:, 2]
    print(
        f"{type(block.entity).__name__} built {len(block.particles)} particles and {sim.model.tet_count} tetrahedra; "
        f"dropped {1000 * drop:.0f} mm it squashed to {1000 * thinnest:.1f} mm on landing "
        f"and stands {1000 * (column.max() - column.min()):.1f} mm tall at rest"
    )

    rss.view(sim)


# What a production implementation adds is in ``soft.py`` beside this file: a
# body read from a tetrahedral mesh file, the flags and the actor record a
# foreign engine needs, anchored and rigid node sets, a settle before the first
# step, and named particle sets a task can pin or grasp.

if __name__ == "__main__":
    main()
