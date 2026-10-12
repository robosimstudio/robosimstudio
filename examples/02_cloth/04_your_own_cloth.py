# SPDX-License-Identifier: Apache-2.0
"""Your own cloth

A second implementation of cloth, written in this file and draped over a ball beside a Panda.

``MyCloth`` puts a particle on every vertex of a grid and two triangles in every
cell through Newton's builder; the library's particle storage does the rest,
from the handle the studio pins to the state a reset returns to. The stock
:class:`robosimstudio.ClothSpec` describes the sheet, and its ``entity=`` field
asks for this class instead of the shipped one. In the studio, click the sheet
to pin a patch and drag it, or close the gripper on its hem and lift it.
Limits: a flat grid only; a fabric's thickness and contact numbers are not read.

**Key features:**

* :func:`robosimstudio.register_entity` to register an implementation under a name of its own
* :class:`robosimstudio.Entity` and its ``build``, where particles and triangles go into Newton's builder
* ``ParticleMixin`` for bind, handle, reset and reading, shared by every kind made of particles
* :class:`robosimstudio.ClothSpec` with ``entity="my_cloth"`` to say "use mine"
"""

import newton
import numpy as np
import warp as wp

import robosimstudio as rss
from robosimstudio.domains.capabilities import ParticleMixin


# Loading this file twice in one process under the same module name is safe:
# the registry keeps the first copy of an identical class.
@rss.register_entity("my_cloth")
class MyCloth(ParticleMixin, rss.Entity):
    """A membrane: a triangle for every half cell, a bending hinge on every shared edge."""

    domain = "cloth"  # the scene's physics arrangement picks the solver for this kind

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        spec, fabric = self.spec, self.spec.material
        (length, width), (nx, ny) = spec.size, spec.cells

        # Grid vertices in the sheet's own frame, one corner at the origin.
        xs, ys = np.meshgrid(np.linspace(0.0, length, nx + 1), np.linspace(0.0, width, ny + 1), indexing="ij")
        vertices = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)

        # Two triangles per cell, the diagonal alternating from cell to cell so
        # the sheet bends alike along both of its own diagonals.
        faces = []
        for i in range(nx):
            for j in range(ny):
                a = i * (ny + 1) + j  # this cell's corners, anticlockwise from (i, j)
                b, c, d = a + ny + 1, a + ny + 2, a + 1
                faces += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]

        # ``pose`` is stated in the work-surface frame; ``placed`` lifts it into the world.
        corner = self.placed((*spec.pose, *spec.quat))
        builder.add_cloth_mesh(
            pos=wp.vec3(*corner[:3]),
            rot=wp.quat(*corner[3:]),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*vertex) for vertex in vertices],
            indices=[int(index) for face in faces for index in face],
            density=fabric.areal_density,  # per area, for a mesh of triangles
            tri_ke=fabric.tri_ke,
            tri_ka=fabric.tri_ka,
            tri_kd=fabric.tri_kd,
            edge_ke=fabric.edge_ke,
            edge_kd=fabric.edge_kd,
            particle_radius=fabric.particle_radius,
        )

    def update_probe(self, state, step=None, *, readback=None):
        """This step's particle positions, for whoever observes the sheet."""
        self.probe.update(state, step)
        return self.probe

    def on_scene_ready(self, pipeline) -> None:
        """The scene can step: record the state a reset returns to."""
        self.snapshot_rest(pipeline)


def main() -> None:
    size = 0.30  # edge of the square sheet [m]
    cells = 20  # quads along an edge: 441 particles, 800 triangles
    radius = 0.07  # of the ball the sheet lands on [m]

    scene = rss.Scene(cell="bench_075", name="your_own_cloth")
    scene.add(rss.robot("panda"))
    # A ball welded to the bench top. Round on purpose: contact is at the sheet's
    # vertices, and a sharp rim shows through between them.
    ball = rss.ShapeSpec(kind="sphere", radius=radius)
    scene.add(rss.StaticSpec(name="ball", shape=ball, xform=(0.0, 0.0, radius, 0.0, 0.0, 0.0, 1.0)))
    sheet = scene.add(
        rss.ClothSpec(
            name="sheet",
            entity="my_cloth",  # the one line that says "use mine"
            size=(size, size),
            cells=(cells, cells),
            pose=(-size / 2, -size / 2, 2 * radius + 0.02),  # the sheet's corner, so its middle is over the ball
        )
    )
    sim = scene.build(control_mode="joint_position")

    sim.run(120).sync()  # two seconds: the sheet drapes over the ball
    hem = sheet.particle_q[:, 2].min() - rss.get_cell("bench_075").surface_z
    print(
        f"{type(sheet.entity).__name__} built {len(sheet.particles)} particles; "
        f"the hem hangs {1000 * hem:.0f} mm above the bench"
    )

    rss.view(sim)


# What a production implementation adds is in ``cloth.py`` beside this file:
# the fabric's own contact and friction numbers written onto the model
# (``apply_model_materials``), a settle before the first step, named particle
# sets a task can pin or grasp, fabrics with a real thickness, the live
# material sliders, and the measures a goal reads.

if __name__ == "__main__":
    main()
