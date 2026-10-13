# SPDX-License-Identifier: Apache-2.0
"""The studio: one browser page over one compiled scene.

The interactive half of the stack, and the half a person uses -- open a scene,
click a thing and drag it, drive the arms, record what you drove, film it.

    import robosimstudio as rss

    scene = rss.Scene(cell="void")
    scene.add(rss.robot("yam_bimanual"))
    scene.add(rss.box(name="cube", half=(0.025,) * 3, at=(0.0, 0.0, 0.2)))
    rss.view(scene.build())

This is application code that ships: the page itself is library code, and every
example reaches it through :func:`robosimstudio.view`. It is imported explicitly
and never re-exported from a lower package, so nothing else starts a viser
server's dependencies loading.

**Over a task it is the same page with three more panels.** Where the benchmark
tier is installed, the page also loads a registered
task, shows its goal clause by clause and runs the policies it ships. The
headless half is ``scripts/bench_run.py``; both go through the same ``Env`` and
the same rollout shape, which is what makes what you watch here and what a batch measures the
same thing.

    from robosimstudio_studio import StudioConfig, StudioSession

    StudioSession("<suite>/<variant>", config=StudioConfig(port=8080)).run()

That half lives in files of its own (:data:`robosimstudio_studio.seams.ADDONS`),
which a release without the benchmark tier leaves out. This package imports
without them; ``PanelTask``, ``PanelSolution`` and ``PanelGoal`` are resolved on
attribute access where they exist, and asking for a task where they do not
raises :class:`~robosimstudio_studio.seams.NotInThisRelease`, which says so.
"""

from .cameras import PanelCamera
from .controls import PanelRecord, PanelVideo
from .handles import GrabHandle, HandleSet
from .inspector import PanelInspector
from .overlay import PanelOverlay
from .panels import Panel, PanelArm, PanelCloth, PanelObject
from .pick import ScenePicker
from .render_view import PanelRenderView
from .scene_panel import PanelScene
from .seams import NotInThisRelease, lazy_names
from .selection import SceneIndex, Subject
from .session import StudioConfig, StudioSession

__all__ = [
    "GrabHandle",
    "HandleSet",
    "NotInThisRelease",
    "Panel",
    "PanelArm",
    "PanelCamera",
    "PanelCloth",
    "PanelInspector",
    "PanelObject",
    "PanelOverlay",
    "PanelRecord",
    "PanelRenderView",
    "PanelScene",
    "PanelVideo",
    "SceneIndex",
    "ScenePicker",
    "StudioConfig",
    "StudioSession",
    "Subject",
]

# The three panels a task page adds. Part of this package's surface wherever the
# benchmark tier is installed -- ``from robosimstudio_studio import PanelTask``
# works there as it always has -- but resolved on access rather than imported
# here, so the package imports where the tier is absent. They are left out of
# ``__all__`` for the same reason: a star-import must not be what asks for them.
__getattr__ = lazy_names(
    __name__,
    {"PanelGoal": "goal_panel", "PanelSolution": "bench_panels", "PanelTask": "bench_panels"},
)
