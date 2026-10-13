# SPDX-License-Identifier: Apache-2.0
"""Builder fixups that every Newton scene with a collapsed MJCF arm needs.

:func:`rekey_custom_attributes` is the important one. ``collapse_fixed_joints``
renumbers bodies and joints and then rewrites custom attributes that *reference*
an entity (``CustomAttribute.references == "body"``), but attributes *keyed by*
one -- anything whose ``frequency`` is ``BODY``, ``SHAPE`` or ``JOINT``, stored
as ``{entity_index: value}`` -- keep their pre-collapse keys. Their values then
land on whichever entity happens to now hold that index.

The symptom is specific and silent: ``mujoco:gravcomp`` is BODY-frequency, so
on the bimanual YAM 8 of 16 arm bodies end up reading the gate/camera/table
entries and lose their gravity compensation. The position-servoed arm then sags
about 7 cm at reach (17 cm in transient) while every joint target is tracked
correctly, and the failure surfaces much later as scripted grasps closing on
air. Call this immediately after any ``collapse_fixed_joints``, including the
one ``add_mjcf(collapse_fixed_joints=True)`` runs internally -- for that path
use ``add_mjcf_collapsed`` (the scene tier's build helpers), which captures the
remap the importer discards.
"""

from __future__ import annotations

from typing import Any

import newton

__all__ = [
    "assert_even_substeps",
    "rekey_custom_attributes",
    "restate_body_attribute",
    "set_body_attribute",
]

# Frequencies whose CustomAttribute.values dict is keyed by an entity index that
# collapse_fixed_joints renumbers. SHAPE is included for completeness: shapes are
# not renumbered by the collapse itself, but they are reparented, and a future
# collapse that compacts them would break the same way.
_BODY_FREQUENCIES = (newton.Model.AttributeFrequency.BODY,)
_JOINT_FREQUENCIES = (
    newton.Model.AttributeFrequency.JOINT,
    newton.Model.AttributeFrequency.JOINT_DOF,
    newton.Model.AttributeFrequency.JOINT_COORD,
)


def rekey_custom_attributes(
    builder: newton.ModelBuilder,
    collapse_results: dict[str, Any],
    *,
    verbose: bool = False,
) -> dict[str, int]:
    """Re-key BODY/JOINT-frequency custom attributes after a fixed-joint collapse.

    Args:
        builder: The builder ``collapse_fixed_joints`` was called on.
        collapse_results: That call's return value, which carries ``body_remap``
            and ``joint_remap`` (old index to new index; a body merged away is
            absent from ``body_remap`` and present in ``body_merged_parent``).
        verbose: Print one line per attribute that was re-keyed.

    Returns:
        Attribute key to number of values that survived the re-key.

    Note:
        Only dict-valued attributes are touched. A custom-frequency attribute
        stores a list and is indexed by its own counter, which the collapse
        does not renumber.

        A body that was merged into its parent hands its value to that parent
        only if the parent declares none of its own -- the surviving body's own
        entry is authoritative, and a merged child must not overwrite it.
    """
    body_remap = dict(collapse_results.get("body_remap") or {})
    joint_remap = dict(collapse_results.get("joint_remap") or {})
    merged_parent = dict(collapse_results.get("body_merged_parent") or {})

    def surviving_body(old: int) -> int | None:
        """New index of ``old``, following merges up to a body that survived."""
        seen: set[int] = set()
        while old not in body_remap and old in merged_parent and old not in seen:
            seen.add(old)
            old = merged_parent[old]
        new = body_remap.get(old)
        return None if new is None or new < 0 else int(new)

    counts: dict[str, int] = {}
    for key, attr in builder.custom_attributes.items():
        values = attr.values
        if not isinstance(values, dict) or not values:
            continue
        if attr.frequency in _BODY_FREQUENCIES:
            direct = {int(k) for k in values if int(k) in body_remap}
            rekeyed: dict[int, Any] = {}
            for old_index, value in values.items():
                new_index = surviving_body(int(old_index))
                if new_index is None:
                    continue
                # A merged child never displaces the surviving body's own value.
                if int(old_index) not in direct and new_index in rekeyed:
                    continue
                rekeyed[new_index] = value
        elif attr.frequency in _JOINT_FREQUENCIES:
            rekeyed = {}
            for old_index, value in values.items():
                new_index = joint_remap.get(int(old_index))
                if new_index is not None and new_index >= 0:
                    rekeyed[int(new_index)] = value
        else:
            continue
        attr.values = rekeyed
        counts[key] = len(rekeyed)
        if verbose:
            print(f"[builder] re-keyed {key}: {len(values)} -> {len(rekeyed)} values")
    return counts


def restate_body_attribute(builder: newton.ModelBuilder, key: str, value: Any) -> int:
    """Set one BODY-frequency custom attribute on every body in the builder.

    The blanket fallback for :func:`rekey_custom_attributes` when the collapse
    results are not available -- ``add_mjcf(collapse_fixed_joints=True)``
    discards them. It is only correct when every surviving body genuinely wants
    the same value, which is true of the case it exists for: after the YAM's
    loose props are stripped, every remaining body is an arm body and every one
    carries ``gravcomp="1"`` in the MJCF. Prefer
    ``add_mjcf_collapsed``, which does not need the assumption.

    Returns:
        The number of bodies written, or 0 if the attribute is not declared.
    """
    attr = builder.custom_attributes.get(key)
    if attr is None:
        return 0
    attr.values = dict.fromkeys(range(builder.body_count), value)
    return builder.body_count


def set_body_attribute(builder: newton.ModelBuilder, key: str, value: Any, bodies) -> int:
    """Set one BODY-frequency custom attribute on a *named* set of bodies.

    The targeted form of :func:`restate_body_attribute`, and the one to use when
    a scene holds more than a robot. ``mujoco:gravcomp`` on every body of a
    grocery scene would gravity-compensate the groceries, so a can released over
    a bin would hang in the air; on the arm's bodies alone it is the arm holding
    itself up, which is what the real Franka does and what the robosuite MJCFs
    this stack's other suites import declare for themselves.

    Existing entries for bodies outside ``bodies`` are left alone.

    Args:
        builder: The builder, after any fixed-joint collapse -- these keys are
            body indices, so setting them before a collapse renumbers them out
            from under the values.
        key: Attribute name, e.g. ``"mujoco:gravcomp"``.
        value: The value to write.
        bodies: Body indices to write it on.

    Returns:
        How many bodies were written, or 0 if the attribute is not declared
        (which is what happens when nobody registered the solver's custom
        attributes before the import).
    """
    attr = builder.custom_attributes.get(key)
    if attr is None:
        return 0
    bodies = [int(body) for body in bodies]  # once: an iterator would be spent by the count
    values = dict(attr.values) if isinstance(attr.values, dict) else {}
    for body in bodies:
        values[body] = value
    attr.values = values
    return len(bodies)


def assert_even_substeps(substeps: int) -> int:
    """Validate a substep count, which must be even. Returns it.

    The substep loop swaps ``state_0``/``state_1`` each pass, so an odd count
    leaves the frame's result in the *other* buffer. Eager that is merely
    confusing; inside a captured CUDA graph the swap is baked in at capture
    time, so the graph reads one buffer while the caller reads the other and
    the scene appears to run at half rate on stale state.
    """
    substeps = int(substeps)
    if substeps <= 0:
        raise ValueError(f"substeps must be positive, got {substeps}")
    if substeps % 2:
        raise ValueError(f"substeps must be even so the state swap comes out even, got {substeps}")
    return substeps
