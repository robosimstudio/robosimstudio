# SPDX-License-Identifier: Apache-2.0
"""Why the studio renders flat grey, and the rig that fixes it.

Newton's viser viewer lights a scene with **one ambient light and no
environment map** -- ``ViewerViser._setup_scene`` adds
``add_light_ambient("ambient_light")`` and then explicitly clears the HDRI.
Ambient light is directionless by definition: every surface returns the same
brightness whichever way it faces. Nothing in the picture is shaded by its own
geometry, so a crease, a curl, a sleeve draped over a hem, and a 5 mm step
where one layer of towel lies on another all render as one undifferentiated
grey field. On a scene whose entire point is that the cloth now has a
*thickness*, that is the one thing that cannot be seen.

Three things are needed, and they do different jobs:

* **An environment map.** Image-based lighting varies with surface normal, so a
  curved sheet stops being a silhouette and starts having a form. This is the
  single biggest change and it costs nothing per frame -- the map is baked in
  the browser.
* **A key light that casts shadows.** Shadow is what says *on top of* rather
  than *coincident with*. A folded flap resting on the half below it is exactly
  the relationship a flat render cannot express and a contact shadow states
  immediately.
* **A low key angle.** Raking light across a surface turns a millimetres-deep
  step into a centimetres-long shadow. Lighting from overhead washes relief
  out; lighting from the side is what makes thickness legible. This is why the
  key sits at about 35 degrees of elevation rather than above the bench.

Ambient does not go away, it goes *down*: it is the fill that stops the
shadowed side going black, and at the viewer's default of 1.0 it drowns
everything else out.

Nothing here is baked. :class:`PanelLighting` puts the same numbers on sliders,
because "presentable" is a judgement made by looking, and a rig that has to be
edited and restarted to be judged will not be.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ENVIRONMENTS", "LightingRig", "PanelLighting"]

ENVIRONMENTS = ("warehouse", "studio", "apartment", "city", "lobby", "park", "sunset", "dawn", "forest", "night")
"""The HDRIs viser ships, most useful first.

``warehouse`` is viser's own default and is neutral with strong directionality.
``studio`` is flatter and cleaner, which suits a figure for a paper; ``sunset``
and ``dawn`` are warm and low, which exaggerates relief and flatters cloth.
"""


@dataclass(frozen=True)
class LightingRig:
    """A three-point rig plus an environment, in viser's own units.

    Attributes:
        environment: One of :data:`ENVIRONMENTS`, or ``""`` for none.
        environment_intensity: Strength of the image-based lighting.
        background: Draw the HDRI behind the scene as well as lighting with it.
            Off by default -- a photographic backdrop behind a robot arm reads
            as a compositing error rather than as a room.
        ambient: Fill that keeps shadowed faces off black.

            The viewer's own default is 1.0, which is what flattens everything --
            but it is also the only light it has, so cutting it is not free.
            This was shipped at 0.25 on the assumption that the key and the
            environment map would more than make up the difference, and that
            assumption was never verified: the result was a scene nobody could
            see. **A lighting default must not be able to black out the view.**
            0.7 is dimmer than the viewer's own and cannot blank it, and the
            panel goes down to 0 for anyone who wants the contrast and can watch
            what it does.
        key_position: Where the key light sits [m]. A viser directional light
            shines from its position toward the origin, so this is a direction
            as much as a place; its *elevation* is what decides whether relief
            shows.
        key_intensity: Key brightness.
        key_color: Key RGB, 0-255. Very slightly warm.
        shadows: Whether the key casts shadows. **Off, against this module's own
            argument, because measured it does not cast them.** Turning it on
            removes the key's contribution roughly in proportion to how much a
            surface faces the key -- the page keeps 0.87 of its brightness with
            the key at 20 degrees of elevation and only 0.76 at 80, where the
            floor faces it most -- and no shadow appears anywhere in the frame:
            of the pixels that darken, 71 % lose more than 20 levels and 2 %
            lose more than 40, which is a flat attenuation and not a shape. On
            the wiping scene against the path tracer it costs 20.7 MAE.

            **It is not that viser cannot cast a shadow.** A box on a floor,
            lit from the side, gets a correct one: localised, 0.1 % of the
            frame darkened, and the light otherwise intact (0.997 of it kept).
            That holds for plain meshes and for instanced batches, and at both
            1x and 0.2x scene scale. Something about *this* scene defeats it
            instead, and the following are ruled out, each measured with a
            three-second settle: the drawn room, the camera's near and far, the
            scene's scale, viser's own default lights, and the GLB materials.

            So the default is a measurement about this page and not a verdict on
            viser, and the checkbox stays: the moment it renders a shadow here,
            this default is wrong. What is left unexplained is a real loose end
            -- roughly one third of the key's contribution disappears, which is
            what one of three cascade lights shadowing everything would cost --
            and it wants a smaller reproduction before anyone files it.
        fill_position: Fill light position [m], opposite the key.
        fill_intensity: Fill brightness. Well below the key, or the two cancel
            into the flat look this is replacing.
        fill_color: Fill RGB, 0-255. Slightly cool, which separates it from the
            key and keeps the shadowed side from reading as dirty white.
    """

    environment: str = "warehouse"
    environment_intensity: float = 1.0
    background: bool = False
    ambient: float = 0.7
    # About 35 degrees up and off to the front-right: high enough to light the
    # top of the bench, low enough that a fold's step casts across it.
    key_position: tuple[float, float, float] = (2.2, -1.7, 1.9)
    key_intensity: float = 2.4
    key_color: tuple[int, int, int] = (255, 250, 244)
    shadows: bool = False
    fill_position: tuple[float, float, float] = (-1.6, 1.5, 1.2)
    fill_intensity: float = 0.55
    fill_color: tuple[int, int, int] = (226, 234, 255)

    @classmethod
    def from_look(cls, render, *, distance: float = 3.0) -> LightingRig:
        """The scene's own :class:`RenderSpec`, in viser's units.

        The look is backend-neutral on purpose -- every field defaults to
        ``None`` meaning "keep the backend's own" -- so this reads what the
        scene actually stated and leaves the rest of the rig at the values the
        page was tuned to. That matters: the numbers below are the *only* ones
        anybody has checked against a real page, and a partial look should not
        drag the unstated half along with it.

        Three translations, and each is a real difference between a path tracer
        and a rasteriser rather than a unit conversion:

        * **Direction becomes a position.** ``light_direction`` is the way the
          key *travels*; a viser directional light shines from where it sits
          toward the origin, so the position is the direction negated and pushed
          out. The distance is arbitrary and only has to clear the scene.
        * **Ambient is a triple and viser takes a scalar**, so it is the mean --
          which is what :class:`RenderSpec` says backends taking a scalar should
          do. It is also clamped: the look states 1.5 for a tracer that tone-maps
          afterwards, and handing that to the page washes it out.
        * **``shadows`` is not carried across**, though the look states it and
          this used to read it. A rasteriser that cannot cast the shadow should
          not claim it; see :attr:`shadows`.
        * **The environment stays the page's own.** ``RenderSpec.environment``
          names an OVRTX preset (``lab``, ``studio``, ``none``) and viser's are
          photographs of real places; there is no mapping between them, and
          guessing one would light the scene from somewhere the render does not.
        """
        if render is None:
            return cls()
        base = cls.matched()
        direction = getattr(render, "light_direction", None)
        key_position = base.key_position
        if direction is not None:
            length = sum(float(v) ** 2 for v in direction) ** 0.5 or 1.0
            key_position = tuple(-float(v) / length * distance for v in direction)
        fills = getattr(render, "fill_lights", None) or ()
        fill_position, fill_intensity = base.fill_position, base.fill_intensity
        if fills:
            fill_direction, strength = fills[0]
            length = sum(float(v) ** 2 for v in fill_direction) ** 0.5 or 1.0
            fill_position = tuple(-float(v) / length * distance for v in fill_direction)
            fill_intensity = float(strength) * base.key_intensity
        ambient = getattr(render, "ambient", None)
        key = getattr(render, "key_intensity", None)
        return cls(
            environment=base.environment,
            environment_intensity=base.environment_intensity,
            # Deliberately not `shadows`: see the field's own docstring. The look
            # states what the *render* should do, and the page cannot do it --
            # the same reason `environment` is not mapped either.
            shadows=base.shadows,
            background=base.background,
            # Clamped to the page's own default: the look's 1.5 is for a
            # renderer that tone-maps, and viser does not do so for a scene.
            ambient=base.ambient if ambient is None else min(sum(float(v) for v in ambient) / 3.0, base.ambient),
            key_position=key_position,
            key_intensity=base.key_intensity if key is None else float(key) * base.key_intensity,
            key_color=base.key_color,
            fill_position=fill_position,
            fill_intensity=fill_intensity,
            fill_color=base.fill_color,
        )

    @classmethod
    def matched(cls) -> LightingRig:
        """The rig as close to the path tracer as a rasteriser gets.

        The bare defaults above are tuned by looking at a garment on a bench.
        These are tuned against the renderer itself: a grid of ambient x key x
        environment scored on the wiping scene, camera ``overhead``, page against
        an OVRTX frame of the same state at the same size. The minimum is
        interior in ambient and key rather than up against a slider end, which
        is the part that says it is a fit and not a shrug.

        | | mean abs error / 255 |
        |---|---|
        | the page before any of this | 73.0 |
        | materials | 58.4 |
        | room and lights, at the bare defaults | 48.5 |
        | **these numbers** | **14.1** |

        Page mean 144.1 against the render's 141.3. They are lower than the
        bare defaults across the board because those were chosen for a scene
        with nothing in it: a drawn room fills the frame with lit surface, and
        the fill that rescued a bare bench blows out a room.

        Not folded into the defaults, because a scene that states no look still
        gets the page that was tuned by looking at it -- and because a default
        that cannot black out the view is a rule this module learned the hard
        way and should keep.
        """
        return cls(ambient=0.40, key_intensity=1.2, environment_intensity=0.15)

    @classmethod
    def plain(cls) -> LightingRig:
        """Newton's own lighting: one ambient light, no environment, no key.

        The thing to fall back to. Flat, and visible -- which is the right way
        round for a default when the alternative cannot be checked from here.
        """
        return cls(environment="", ambient=1.0, key_intensity=0.0, fill_intensity=0.0, shadows=False)

    def apply(self, server) -> None:
        """Install this rig on a live viser server.

        Safe to call repeatedly: every node is addressed by name, so re-applying
        replaces rather than accumulates. The ambient node deliberately reuses
        the viewer's own ``ambient_light`` name -- that is what turns its 1.0
        down instead of adding a second ambient light beside it.
        """
        scene = server.scene
        if self.environment:
            scene.configure_environment_map(
                hdri=self.environment,
                background=self.background,
                environment_intensity=float(self.environment_intensity),
            )
        else:
            scene.configure_environment_map(hdri=None)

        scene.add_light_ambient("ambient_light", intensity=float(self.ambient))
        scene.add_light_directional(
            "/lighting/key",
            color=tuple(int(c) for c in self.key_color),
            intensity=float(self.key_intensity),
            cast_shadow=bool(self.shadows),
            position=tuple(float(v) for v in self.key_position),
        )
        scene.add_light_directional(
            "/lighting/fill",
            color=tuple(int(c) for c in self.fill_color),
            intensity=float(self.fill_intensity),
            cast_shadow=False,
            position=tuple(float(v) for v in self.fill_position),
        )


class PanelLighting:
    """Sliders for the rig, so it can be judged by looking at it.

    Not a :class:`~robosimstudio_studio.panels.Panel`: it touches
    nothing in the environment and nothing in Warp, only the viser server, so it
    has no ``drain`` step and survives a task being unloaded underneath it.
    Callbacks apply immediately, which is safe for the same reason -- they are
    scene-graph writes on viser's own threads and never enter the solver.

    Args:
        server: The viser server.
        rig: Starting values.
        viewer: The studio viewer, if the garments' baked shading should
            follow the key. The fuzz a garment wears is unlit geometry with a
            Lambert term baked into its colours
            (:mod:`~robosimstudio_studio.garments`), so a key that
            moves without telling it leaves the pile lit from a room that no
            longer exists.
    """

    def __init__(self, server, rig: LightingRig | None = None, *, viewer=None) -> None:
        self.server = server
        self.viewer = viewer
        self.rig = rig if rig is not None else LightingRig()
        self.apply()

    def apply(self) -> None:
        """Install the rig, and point the garments' baked shading at its key."""
        self.rig.apply(self.server)
        if self.viewer is not None and hasattr(self.viewer, "set_garment_light"):
            self.viewer.set_garment_light(self.rig.key_position)

    def build(self, gui) -> None:
        """Register the widgets, in whatever container the caller opened.

        No folder of its own. Both callers already build this inside
        ``Appearance``, which is where a lighting control belongs and is
        already collapsed.
        """
        environment = gui.add_dropdown(
            "Environment", options=("none", *ENVIRONMENTS), initial_value=self.rig.environment or "none"
        )
        intensity = gui.add_slider(
            "Environment", min=0.0, max=3.0, step=0.05, initial_value=self.rig.environment_intensity
        )
        background = gui.add_checkbox("Show backdrop", initial_value=self.rig.background)
        ambient = gui.add_slider("Ambient fill", min=0.0, max=1.5, step=0.05, initial_value=self.rig.ambient)
        key = gui.add_slider("Key", min=0.0, max=6.0, step=0.1, initial_value=self.rig.key_intensity)
        elevation = gui.add_slider("Key elevation", min=5.0, max=85.0, step=1.0, initial_value=self._elevation())
        azimuth = gui.add_slider("Key azimuth", min=-180.0, max=180.0, step=5.0, initial_value=self._azimuth())
        fill = gui.add_slider("Fill", min=0.0, max=3.0, step=0.05, initial_value=self.rig.fill_intensity)
        shadows = gui.add_checkbox("Shadows", initial_value=self.rig.shadows)
        plain = gui.add_checkbox("Newton default (flat)", initial_value=False)

        widgets = (environment, intensity, background, ambient, key, elevation, azimuth, fill, shadows, plain)

        def update(_event=None) -> None:
            import dataclasses

            if plain.value:
                # One click back to a scene that is definitely visible, without
                # restarting. Worth its own control: the failure mode this rig
                # can produce is "nothing renders", and diagnosing that from a
                # blank browser tab is miserable.
                LightingRig.plain().apply(self.server)
                return
            self.rig = dataclasses.replace(
                self.rig,
                environment="" if environment.value == "none" else environment.value,
                environment_intensity=float(intensity.value),
                background=bool(background.value),
                ambient=float(ambient.value),
                key_intensity=float(key.value),
                key_position=_from_angles(float(elevation.value), float(azimuth.value)),
                fill_intensity=float(fill.value),
                shadows=bool(shadows.value),
            )
            self.apply()

        for widget in widgets:
            widget.on_update(update)

    def _elevation(self) -> float:
        import numpy as np

        x, y, z = self.rig.key_position
        return float(np.degrees(np.arctan2(z, np.hypot(x, y))))

    def _azimuth(self) -> float:
        import numpy as np

        x, y, _ = self.rig.key_position
        return float(np.degrees(np.arctan2(y, x)))


def _from_angles(elevation_deg: float, azimuth_deg: float, distance: float = 3.2) -> tuple[float, float, float]:
    """A key-light position from the angles a person actually thinks in.

    Elevation is what matters for showing relief and is impossible to judge from
    a raw xyz, which is why the panel exposes these two and not three
    coordinates. The distance is arbitrary: a directional light has no falloff,
    so only the direction from here to the origin is read.
    """
    import numpy as np

    elevation, azimuth = np.radians(elevation_deg), np.radians(azimuth_deg)
    horizontal = distance * np.cos(elevation)
    return (
        float(horizontal * np.cos(azimuth)),
        float(horizontal * np.sin(azimuth)),
        float(distance * np.sin(elevation)),
    )
