# SPDX-License-Identifier: Apache-2.0
"""A body's mass properties across the things a build does to its shapes.

Adding a shape to a body adds mass to it, and a stated mass is the source's:
:func:`preserve_body_inertia` holds the mass across a collider swap, and
:func:`set_body_mass` restates it afterwards with the inertia scaled to match.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import newton

__all__ = ["preserve_body_inertia", "set_body_mass"]


@contextlib.contextmanager
def preserve_body_inertia(builder: newton.ModelBuilder, body: int) -> Iterator[None]:
    """Hold a body's mass properties across a collider swap.

    **Adding a shape to a body adds mass to it.** That is right when the shape is
    the body's geometry and wrong when it is a *stand-in* for geometry that is
    still there -- swapping a mesh collider for a box leaves the mesh's mass in
    place and adds the box's on top, so the body silently gets heavier. Measured
    on the grocery stage, where two such swaps happen (an axis-aligned box per
    grocery, five boxes for the tote's cavity), against the masses the export
    authors and Isaac itself runs:

        body                  authored    with the swap
        tomato sauce can       0.400 kg     0.787 kg   1.97x
        alphabet soup can      0.500 kg     0.893 kg   1.79x
        parmesan canister      0.200 kg     0.684 kg   3.42x
        mustard bottle         0.300 kg     0.806 kg   2.69x
        milk carton            0.500 kg     1.506 kg   3.01x
        tote                   1.899 kg     6.542 kg   3.45x

    The importer reads ``physics:mass`` exactly right -- straight out of
    ``add_usd`` every one of those is the authored value to four decimals -- so
    the error is entirely ours, and it is the kind that reads as bad physics
    rather than as a bug: a 3x-heavy object needs 3x the grip to hold and 3x the
    impulse to stop, so it slips in the fingers and lands hard.

    Restores mass, centre of mass, inertia and both inverses, so a body left
    massless (a static's shell) stays massless rather than acquiring an inverse.

    Args:
        builder: The builder, pre-finalize.
        body: The body whose mass properties to hold. Negative (the static world
            body, which has no entry) is a no-op.

    Yields:
        Nothing; the properties are restored on exit, including on an exception.
    """
    if int(body) < 0:
        yield
        return
    body = int(body)
    fields = ("body_mass", "body_com", "body_inertia", "body_inv_mass", "body_inv_inertia")
    saved = {f: getattr(builder, f)[body] for f in fields if len(getattr(builder, f, ())) > body}
    try:
        yield
    finally:
        for field, value in saved.items():
            getattr(builder, field)[body] = value


def set_body_mass(builder: newton.ModelBuilder, body: int, mass: float) -> float:
    """Rescale a body to a stated mass, keeping its inertia consistent with it.

    **A spec that can only say "density" cannot reproduce a source that says
    "mass".** Newton derives mass from density times the *shape's* volume, and
    for a mesh that is the volume of the triangles -- not of the box a collider
    approximation will later stand in for. The two are as far apart as the mesh
    is hollow: measured on the industrial tools, a D battery fills 40.7% of its
    oriented bounding box, a multimeter 55.2% and a pair of safety glasses
    1.7%, so a density back-computed from box volume built glasses weighing
    0.86 g against the 40.5 g their export states -- a 47x error, and the kind
    that reads as bad physics rather than as arithmetic.

    Scaling rather than assigning is what keeps the body coherent: for fixed
    geometry at uniform density the inertia tensor is linear in the mass, so
    multiplying both by the same factor lands on the mass asked for with the
    shape's own inertia *distribution* intact. Assigning the mass alone would
    leave a body that weighs one thing and spins like another.

    Args:
        builder: The builder, pre-finalize, after every shape is on the body.
        body: The body to rescale.
        mass: The target mass [kg]. Must be positive.

    Returns:
        The mass the body had before, which is what the geometry implied.
    """
    import numpy as np

    if mass <= 0.0:
        raise ValueError(f"body {body}: mass must be positive, got {mass}")
    before = float(builder.body_mass[body])
    if before <= 0.0:
        raise ValueError(
            f"body {body}: has no mass to rescale (it is {before}); a massless body has no inertia "
            f"distribution to preserve, so its mass cannot be set this way"
        )
    scale = mass / before
    inertia = np.asarray(builder.body_inertia[body], dtype=np.float64) * scale
    builder.body_mass[body] = mass
    builder.body_inv_mass[body] = 1.0 / mass
    builder.body_inertia[body] = type(builder.body_inertia[body])(inertia.reshape(-1))
    inverse = np.zeros(9, dtype=np.float64)
    square = inertia.reshape(3, 3)
    if abs(np.linalg.det(square)) > 0.0:
        inverse = np.linalg.inv(square).reshape(-1)
    builder.body_inv_inertia[body] = type(builder.body_inv_inertia[body])(inverse)
    return before
