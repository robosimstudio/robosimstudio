# SPDX-License-Identifier: Apache-2.0
"""What the page is currently looking at, and how a click becomes that.

**A :class:`Subject` is anything the page can be about**, and the list is the
compiled scene's own: an object, a static, a deformable, an arm, a region. Not a
class hierarchy -- they have almost nothing in common and the differences are the
content -- but one identity a panel can hold and one name it can show.

**:class:`SceneIndex` is the reverse lookup.** Every handle already carries the
shapes it owns and every cloth its particles. Building the full
``shape -> subject`` map costs one pass over the handles at load and is what
makes a bench, a bin, a receptacle and a robot link selectable at all -- and
:meth:`~robosimstudio.scene.handles.SceneHandles.segmentation_labels`
has been building essentially this table for renderers the whole time.

Typing goes through the document, the one existing path:
``handle.name -> TaskDoc.objects[].type -> ASSETS``. Note it is **not**
``EntityHandle.kind``, which is the *category* (``"hammer"``) and not the
registry key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["KINDS", "SceneIndex", "Subject", "arm_link_bodies"]

KINDS: tuple[str, ...] = ("object", "cloth", "arm", "static", "region")
"""What a subject can be, in the order a scene tree lists them.

Movable things first, because they are what somebody came to click on; scenery
and the boxes a goal is measured against after. Arms sit between the two: they
move, and nobody browses a scene looking for one.
"""


def arm_link_bodies(model: Any, ee_body: int) -> tuple[int, ...]:
    """The moving links of one arm, from its tool centre down to a fixed root.

    :attr:`~robosimstudio.scene.handles.ArmHandle.shapes` cannot answer this.
    It is filled for contact *exclusion*, where "belonging to this arm" means
    the whole robot, and on ``yam_bimanual`` the left and right lists are the
    same 112 shapes -- see :meth:`SceneIndex._shape_owners`, which resolves the
    same ambiguity for clicks by ordering rather than by topology. A *handle*
    cannot be resolved by ordering: putting a wrench on the right forearm has
    to slacken the right arm, not whichever one sorted first.

    So the chain is walked, the way
    :meth:`~robosimstudio.solvers.ik.arm_ik.ArmIK._build_chain_fk` walks it:
    ``joint_child`` maps a body to the joint that drives it, ``joint_parent``
    steps to the next one up, and the walk ends at a body no joint drives.

    Only revolute and prismatic children are returned. A fixed joint's child
    does not move relative to its parent, so a wrench on it is a wrench on the
    mount -- pulling on the base of a welded robot achieves nothing, and
    offering a handle that cannot move is worse than offering none.
    """
    import newton

    joint_type = model.joint_type.numpy()
    joint_parent = model.joint_parent.numpy()
    joint_child = model.joint_child.numpy()
    moving = {int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)}
    child_to_joint = {int(joint_child[j]): j for j in range(model.joint_count)}

    bodies: list[int] = []
    body = int(ee_body)
    seen: set[int] = set()
    while body in child_to_joint and body not in seen:
        seen.add(body)
        joint = child_to_joint[body]
        if int(joint_type[joint]) in moving:
            bodies.append(body)
        body = int(joint_parent[joint])
        if body < 0:
            break
    return tuple(bodies)


@dataclass(frozen=True)
class Subject:
    """One thing the page can be about.

    Attributes:
        kind: One of :data:`KINDS`.
        name: Its name in the compiled scene, which is also the name a document
            and a goal clause use for it.
        handle: The compiled scene's handle -- an ``EntityHandle``,
            ``DeformableHandle``, ``ArmHandle`` or ``RegionHandle``. Where it is.
        entity: The live :class:`~robosimstudio.scene.entity.Entity`,
            for what may be read from it and written to it -- published states
            and tunables. ``None`` for an arm or a region, which are not
            entities, and for an object the scene built without one.
        asset: The registry key of its declaration, or ``""`` when the document
            does not type it. **Not** the category: a hammer's handle says
            ``kind="hammer"`` and its asset is ``"hammer"`` only by coincidence
            of naming, while a bin's are ``"bin"`` and ``"bin"`` and a panel's
            are ``""`` and ``""``. The registry only answers to the second.
    """

    kind: str
    name: str
    handle: Any = None
    entity: Any = None
    asset: str = ""

    @property
    def label(self) -> str:
        """``"cloth · towel"`` -- what a dropdown row and a section header say."""
        return f"{self.kind} · {self.name}"

    @property
    def key(self) -> str:
        """A stable identity across a reset, for re-selecting the same thing.

        Kind and name rather than an index: indices move when a scene recompiles
        and a name does not, and re-selecting by index after a reload is how a
        page ends up showing one object's numbers under another one's heading.
        """
        return f"{self.kind}:{self.name}"


class SceneIndex:
    """Every subject in one compiled scene, and the lookups a click needs.

    Built once per task load. Nothing here reads state -- it is a map from the
    model's own numbering to names, and the model's numbering does not change
    while a model exists.

    Args:
        handles: The compiled scene's
            :class:`~robosimstudio.scene.handles.SceneHandles`.
        scene: The :class:`~robosimstudio.scene.compiler.CompiledScene`,
            for the live entities. Optional: a subject without one still names
            and locates its thing, which is all a test needs.
        doc: The task document, for typing entities against the catalog.
            Optional, and ``None`` is a legitimate state -- a task assembled
            without a document has subjects with no ``asset``.
    """

    def __init__(self, handles: Any, scene: Any = None, doc: Any = None) -> None:
        self.handles = handles
        types = {o.name: o.type for o in getattr(doc, "objects", ())} if doc is not None else {}

        subjects: list[Subject] = []
        for kind, table in (
            ("object", getattr(handles, "objects", {})),
            ("cloth", getattr(handles, "deformables", {})),
            ("arm", getattr(handles, "arms", {})),
            ("static", getattr(handles, "statics", {})),
            ("region", getattr(handles, "regions", {})),
        ):
            for name, handle in sorted(table.items()):
                subjects.append(
                    Subject(
                        kind=kind,
                        name=name,
                        handle=handle,
                        entity=_entity(scene, kind, name),
                        asset=types.get(name, ""),
                    )
                )
        self.subjects: tuple[Subject, ...] = tuple(subjects)

        self._by_key = {s.key: s for s in self.subjects}
        self._by_name: dict[str, Subject] = {}
        for subject in self.subjects:
            # First wins, in KINDS order, so a name shared between a region and
            # the fixture it is attached to resolves to the movable one -- which
            # is what somebody typing a name means. Scenes do not actually
            # collide names today; this decides what happens when one does
            # rather than leaving it to dict order.
            self._by_name.setdefault(subject.name, subject)

        self._by_shape: dict[int, Subject] = {}
        self._by_body: dict[int, Subject] = {}
        work = getattr(handles, "work_arm", "")
        for subject in self._shape_owners(work):
            for shape in np.asarray(getattr(subject.handle, "shapes", ()), dtype=np.int64).ravel():
                self._by_shape.setdefault(int(shape), subject)
            # A chain claims every segment; everything else has one body or none.
            # Empty and absent must behave the same: `EntityHandle.bodies` exists
            # on every handle now and is empty for the ordinary one-body case, so
            # testing `is None` registered no body at all for every plain object.
            bodies = tuple(getattr(subject.handle, "bodies", ()) or ())
            if not bodies:
                body = int(getattr(subject.handle, "body", -1))
                bodies = (body,) if body >= 0 else ()
            for body in bodies:
                self._by_body.setdefault(int(body), subject)

        self._by_particle: dict[int, Subject] = {}
        for subject in self.subjects:
            if subject.kind != "cloth":
                continue
            for particle in np.asarray(getattr(subject.handle, "particles", ()), dtype=np.int64).ravel():
                self._by_particle[int(particle)] = subject

    def _shape_owners(self, work_arm: str) -> tuple[Subject, ...]:
        """Subjects in the order a shape's owner is decided, first claim winning.

        :data:`KINDS` order already does most of it -- an object's shape beats a
        static's -- and one case needs saying out loud.

        **Both arms of a bimanual cell claim every shape the robot has.**
        ``ArmHandle.shapes`` is filled for contact *exclusion*, where "everything
        belonging to this arm" sensibly means the whole robot, and it is measured:
        on ``yam_bimanual`` the left and right lists are the same 112 shapes.
        Read as identity that makes clicking any link ambiguous, and resolving it
        alphabetically would mean a click on the right hand selects the left arm
        on one robot and the right on another -- exactly the class of bug
        ``SceneHandles.work_arm``'s own docstring was written about.

        So the work arm goes first. It is the arm the task is about and the one
        the page acts with, which makes it the honest answer to an ambiguous
        click rather than merely a deterministic one.
        """
        arms = [s for s in self.subjects if s.kind == "arm"]
        rest = [s for s in self.subjects if s.kind != "arm"]
        arms.sort(key=lambda s: (s.name != work_arm, s.name))
        return (
            *[s for s in rest if s.kind in ("object", "cloth")],
            *arms,
            *[s for s in rest if s.kind not in ("object", "cloth")],
        )

    # ------------------------------------------------------------------ lookup
    def at_shape(self, shape: int) -> Subject | None:
        """Whatever owns one model shape.

        An arm owns its links' shapes, a static owns the bench's, and both are
        something the page can be about rather than a body index with no name.
        """
        return self._by_shape.get(int(shape))

    def at_body(self, body: int) -> Subject | None:
        """Whatever owns one body. The fallback when a shape is unknown."""
        return self._by_body.get(int(body))

    def at_particle(self, index: int) -> Subject | None:
        """Which deformable one particle belongs to."""
        return self._by_particle.get(int(index))

    def by_name(self, name: str) -> Subject | None:
        return self._by_name.get(name)

    def by_key(self, key: str) -> Subject | None:
        return self._by_key.get(key)

    def of_kind(self, kind: str) -> tuple[Subject, ...]:
        return tuple(s for s in self.subjects if s.kind == kind)

    def first(self) -> Subject | None:
        """What to select when nothing is selected yet.

        The first movable thing, in :data:`KINDS` order -- an object if there is
        one, else a deformable, else whatever there is. A page that opens on the
        bench has opened on the least interesting thing in the scene.
        """
        return self.subjects[0] if self.subjects else None

    def summary(self) -> str:
        """``"4 objects · 1 cloth · 2 arms"`` -- what the scene section shows."""
        counts = [(kind, len(self.of_kind(kind))) for kind in KINDS]
        return " · ".join(f"{n} {kind}{'s' if n != 1 else ''}" for kind, n in counts if n) or "nothing"

    def __len__(self) -> int:
        return len(self.subjects)


def _entity(scene: Any, kind: str, name: str) -> Any:
    """The live entity behind one handle, when there is one.

    Arms and regions have none -- an arm is the embodiment's and a region is
    geometry -- and that is why a subject's ``entity`` is optional rather than
    something to look up lazily and fail at.
    """
    if scene is None:
        return None
    table = {
        "object": getattr(scene, "objects", {}),
        "cloth": getattr(scene, "deformables", {}),
        "static": getattr(scene, "statics", {}),
    }.get(kind)
    if table is None:
        return None
    # ``by_name`` rather than a view: which view holds a thing is not what this
    # is asking, and keying on one is how a chain stopped being found the day
    # it started publishing a handle.
    return table.get(name) or getattr(scene, "by_name", {}).get(name)
