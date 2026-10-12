# Extending

The two things you add from outside the library: a robot,
and a whole kind of thing with its own physics.

<!-- examples:start: written by scripts/example_readmes.py from each script's docstring -->

## Add a robot

![Add a robot](media/00_add_a_robot.webp)

A robot the library never registered, declared in this script over the Panda's MJCF and driven from the studio.

An `rss.EmbodimentSpec` is the whole declaration. `asset`
names the file; each `rss.ArmSpec` says which joints are an
arm (`joint_names`, in order), which body is its wrist (`ee_body_name`),
which registered hand sits on it (`gripper`) and where it starts (`home_q`);
`base_offset` is where the base stands in the work-surface frame. Handing the
spec to `rss.robot` builds it exactly as a registered name
would: an arm panel with a gizmo, IK, and a cube to reach for. Point `asset`
at your own MJCF or URDF and rename the joints.

**Key features:**

* `rss.EmbodimentSpec` for a robot declared inline, never registered
* `rss.ArmSpec` to index an arm out of the file: joints, wrist, hand, home posture
* `rss.AssetSpec` to name the file
* `rss.robot` taking the spec in place of a registry name

```bash
uv run examples/04_extending/00_add_a_robot.py
```

Source: [`00_add_a_robot.py`](00_add_a_robot.py)

## A new kind of thing

![A new kind of thing](media/01_new_kind.webp)

A chain of loose links the library has never heard of, defined in this file: click any link and drag it.

Everything a kind is made of is here, and nothing under `robosimstudio/` was
edited to run it: a spec, an entity that builds it, a `Ref` with an accessor
of its own, a kernel that runs every substep, and the registration. The page
shows the chain's span, read through that accessor.

**Key features:**

* `rss.register_domain` to declare the kind
* `rss.register_entity` and `rss.Entity` for what builds it
* `rss.Entity.handle` and `rss.Entity.read` so the page and the measures can reach it
* `rss.Entity.contribute_hooks` for a kernel in the step
* `rss.Ref` subclassed for an accessor, and `page=` on `rss.view` for a readout

```bash
uv run examples/04_extending/01_new_kind.py
```

Source: [`01_new_kind.py`](01_new_kind.py)

<!-- examples:end -->

## Add a robot

How to put a robot of your own in a scene: from a file, or
registered under a name like the shipped ones.

A robot is two declarations and no code:

- An `EmbodimentSpec`: the file, the arms, where the base
  stands, and how it is built.
- A `GripperSpec`: the fingers, the tool-centre
  point and the pads. Needed only for a hand the shipped registry does not have.

Both are frozen dataclasses. One entity builds every robot from its spec.

### From a file

Name the joints and the wrist body, and the arm builds, solves IK and takes
joint-position commands:

```python
import robosimstudio as rss

arm = rss.ArmSpec(
    name="right",
    joint_names=tuple(f"robot0_joint{i}" for i in range(1, 8)),
    ee_body_name="robot0_link7",
    gripper="franka_hand_robosuite",  # a registered hand; leave it out for a bare arm
)

scene = rss.Scene(cell="void")
scene.add(rss.robot(asset="embodiments/panda/panda_standalone.xml", arms=[arm], at=(0, 0, 0, 0, 0, 0, 1)))
scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.5, 0.0, 0.1)))
sim = scene.build(control_mode="joint_position")
rss.view(sim)    # drag the wrist gizmo, close the fingers on the cube
```

- `asset` is a path under the asset roots. `assets/embodiments/<name>/` is
  where the shipped files are. `.xml` is MJCF and `.urdf` is URDF. A USD stage
  needs the `[usd]` extra.
- `at` is where the base goes, in the work-surface frame: a position
  `(x, y, z)`, or a pose `(x, y, z, qx, qy, qz, qw)`.
- A hand is named from the hand registry, because its finger bodies, travel and
  tool-centre point are measured facts. `./ctl catalog grippers` lists the
  names.

### Registered under a name

A spec registered at import time is a robot `rss.robot` finds by name:

```python
from robosimstudio.robots import register_embodiment

register_embodiment(
    rss.EmbodimentSpec(
        name="my_panda",
        asset=rss.AssetSpec(path="embodiments/panda/panda_standalone.xml"),
        arms=(arm,),
        base_offset=(-0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),  # the mount, in the work-surface frame
    )
)

scene = rss.Scene(cell="bench_075")
scene.add(rss.robot("my_panda"))
scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.0, 0.1)))
sim = scene.build(control_mode="joint_position")
```

`register_embodiment` and `register_gripper` are imported from
`robosimstudio.robots`. The shipped robots are registered the same way, in
`robosimstudio/robots/embodiments/` and `robosimstudio/robots/grippers/`, one
module each. A registered robot stands where its `base_offset` says; `at` on
`rss.robot` is read only when the spec has none.

### The fields that matter

`EmbodimentSpec`:

| Field | Decides |
| --- | --- |
| `asset` | The file. |
| `arms` | One `ArmSpec` per arm. |
| `base_offset` | Where the base stands, in the work-surface frame. |
| `gripper` | The default hand, by registry name. An arm's own `gripper` overrides it. |
| `arm_actuation`, `gripper_actuation` | Servo gains: `ActuationSpec(target_ke, target_kd)`. |
| `hull_colliders` | Replace each link's collider with its convex hull. |
| `decompose_colliders` | Links to decompose into convex pieces instead. Use it on fingertips. |
| `enable_self_collisions`, `cross_arm_collisions` | Whether links collide within an arm, and whether two arms are solid to each other. |
| `gravcomp` | Gravity compensation. |
| `urdf` | The arm as a URDF. IK runs on [PyRoKi](https://github.com/chungmin99/pyroki) from it, checked against the model at build; an arm without one is solved by Newton's own solver. |
| `ik_iterations` | Iterations of the fallback solver, 8 by default. |
| `color`, `finish` | The look. Leave unset to keep the file's own materials. |

`ArmSpec`: `name`, `joint_names` in order (or a `joint_prefix`), `ee_body_name`
for the body the hand bolts onto, `gripper` and `home_q`.

`GripperSpec`, for a hand of your own:

| Field | Decides |
| --- | --- |
| `finger_names`, `finger_open_q`, `finger_closed_q` | The finger joints and their travel. The command channel is 0 for open and 1 for shut. |
| `tcp_body`, `tcp_offset`, `approach_axis`, `close_axis` | The tool-centre point and how the hand approaches and closes. |
| `finger_bodies`, `pad_half_span`, `pad_thickness`, `pad_depth`, `tip_reach` | The pads, as a grasp measures them. |
| `asset`, `mount_body`, `mount_xform` | An attached hand: its own file and where it bolts on. A hand welded into the arm's file leaves these unset. |

### Check it

```python
from robosimstudio.robots.registry import get_embodiment, resolve_gripper

spec = get_embodiment("my_panda")
print([a.name for a in spec.arms], resolve_gripper(spec, spec.arms[0]).name)    # ['right'] franka_hand_robosuite
```

Then stand it in a scene and look: drag the wrist gizmo, and close the fingers
on a box.


## Add a kind of thing

How to add a kind of thing the library does not ship:
another implementation of cloth or soft bodies, or a kind of your own.

### Two routes

- **Reimplement a kind that exists.** Register an `Entity`
  under a name of your own and ask for it from the stock spec with
  `entity="my_cloth"`. The spec, the materials and the studio's cloth panel stay.
- **Add a new kind.** Write a spec, declare the kind with
  `register_domain`, and register the entity that builds it.

Nothing under `robosimstudio/` is edited for either.

### The registration calls

| Call | Registers |
| --- | --- |
| `rss.register_domain` | A kind: the state its things own, the role they play, what they touch |
| `rss.register_entity` | A class that builds a thing, under a name a spec asks for |
| `rss.register_cell` | A place to stand things in |
| `rss.register_backend` | A solver, under an alias `rss.backend` accepts |
| `rss.register_coupler` | A way for two solvers to exchange forces |

Registering the same thing twice with the same value is accepted.

### A second implementation of cloth

```python
import numpy as np
import warp as wp

import robosimstudio as rss
from robosimstudio.domains.capabilities import ParticleMixin


@rss.register_entity("my_cloth")
class MyCloth(ParticleMixin, rss.Entity):
    domain = "cloth"  # the arrangement's cloth solver steps it

    def build(self, builder, episode):
        spec, fabric = self.spec, self.spec.material
        (length, width), (nx, ny) = spec.size, spec.cells
        xs, ys = np.meshgrid(np.linspace(0, length, nx + 1), np.linspace(0, width, ny + 1), indexing="ij")
        vertices = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
        faces = []
        for a in (i * (ny + 1) + j for i in range(nx) for j in range(ny)):
            faces += [(a, a + ny + 1, a + ny + 2), (a, a + ny + 2, a + 1)]
        corner = self.placed((*spec.pose, *spec.quat))  # work-surface frame to world
        builder.add_cloth_mesh(
            pos=wp.vec3(*corner[:3]), rot=wp.quat(*corner[3:]), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(*v) for v in vertices], indices=[int(i) for f in faces for i in f],
            density=fabric.areal_density, tri_ke=fabric.tri_ke, tri_ka=fabric.tri_ka, tri_kd=fabric.tri_kd,
            edge_ke=fabric.edge_ke, edge_kd=fabric.edge_kd, particle_radius=fabric.particle_radius,
        )

    def update_probe(self, state, step=None, *, readback=None):
        self.probe.update(state, step)
        return self.probe

    def on_scene_ready(self, pipeline):
        self.snapshot_rest(pipeline)  # the state a reset returns to


scene = rss.Scene(cell="bench_075")
sheet = scene.add(rss.ClothSpec(name="sheet", entity="my_cloth", size=(0.3, 0.3), cells=(20, 20), pose=(-0.15, -0.15, 0.1)))
sim = scene.build()
sim.run(120).sync()
print(sheet.particle_q.shape)    # (441, 3)
```

`build` puts particles and triangles into Newton's builder. `ParticleMixin`
supplies what every kind made of particles shares: the handle the studio takes
hold of, reading the particles, and reset. `entity="my_cloth"` on the spec is
the one line that says "use mine".

### A kind of your own

```python
from dataclasses import dataclass, field

import numpy as np
import warp as wp

import robosimstudio as rss


@dataclass(frozen=True, kw_only=True)
class StackSpec:
    entity: str = "stack"
    name: str = "stack"
    count: int = 5
    half: float = 0.02
    at: tuple[float, float, float] = (0.0, 0.0, 0.3)
    sim: object = field(default_factory=lambda: rss.SimRequirements(domain="stack", contacts=rss.contacts_with("surface")))


rss.register_domain("stack", state=("bodies", "joints", "shapes"), reads="rigid", role="objects",
                    contacts=("surface",), summary="A column of loose blocks.")


@rss.register_entity("stack")
class Stack(rss.Entity):
    domain = "stack"

    def build(self, builder, episode):
        spec = self.spec
        self.bodies = []
        for i in range(spec.count):
            pose = self.placed(np.array([*spec.at, 0.0, 0.0, 0.0, 1.0]))
            pose[2] += i * 2.2 * spec.half
            body = builder.add_body(xform=wp.transform(tuple(map(float, pose[:3])), wp.quat(*map(float, pose[3:]))), label=f"{spec.name}_{i}")
            builder.add_shape_box(body, hx=spec.half, hy=spec.half, hz=spec.half, label=f"{spec.name}_{i}")
            self.bodies.append(body)

    def handle(self, model):
        from robosimstudio.scene.handles import EntityHandle

        return EntityHandle(name=self.name, body=self.bodies[0], bodies=tuple(self.bodies))


scene = rss.Scene(cell="bench_075")
stack = scene.add(StackSpec(count=5))
sim = scene.build()
sim.run(120).sync()
print(stack.position)    # the bottom block, resting on the bench
```

- The spec is a plain frozen dataclass. `entity` names the class that builds
  it, and `sim` names the kind and what it touches by default.
- `register_domain` says what state the kind owns (`bodies`, `joints`,
  `particles`, `shapes`), how it is read (`rigid` or `cloth`), and its role
  (`objects`, `fixtures` or `deformables`).
- `handle` is what makes the thing clickable on the page.
- For accessors of your own, subclass `Ref` and set
  `ref = MyRef` on the entity. For a kernel that runs every substep, override
  `contribute_hooks`. `examples/04_extending/01_new_kind.py` does both.

### Where the shipped implementations are

`examples/02_cloth/cloth.py` and `examples/03_soft/soft.py` are the files behind
`rss.sheet` and `rss.sponge`, imported as `robosimstudio.domains.cloth` and
`robosimstudio.domains.soft`. The library scans them the first time a kind is
looked up. A folder of your own is scanned too when
`ROBOSIMSTUDIO_DOMAIN_ROOTS` names it, or `robosimstudio.roots.add_domain_root`
is called with it.


**See also:**

- the docstrings of `rss.EmbodimentSpec`, `rss.ArmSpec` and `rss.AssetSpec`
- `robosimstudio/robots/embodiments/panda.py`, a registered robot to read
- `examples/04_extending/00_add_a_robot.py`
- the docstrings of `Entity`, `rss.SimRequirements` and `contacts_with`, and of
  each `rss.register_*` call
- `examples/02_cloth/04_your_own_cloth.py`, `examples/03_soft/02_your_own_soft.py`
  and `examples/04_extending/01_new_kind.py`
