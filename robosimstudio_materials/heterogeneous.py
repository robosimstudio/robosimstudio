# SPDX-License-Identifier: Apache-2.0
"""One connected, part-labelled tetrahedral object with source material estimates.

The part labels describe the asset, independent of the Newton solver used to step it.
The current implementation requires a Newton particle solver (VBD in the example).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import newton
import numpy as np
import warp as wp

from robosimstudio.domains.capabilities import ParticleMixin
from robosimstudio.scene.entity import Entity
from robosimstudio.scene.registry import register_entity

__all__ = ["EntityHeterogeneousSoft", "HeterogeneousSoftSpec", "load_labelled_tets"]


def load_labelled_tets(mesh_path: str | Path, params_path: str | Path) -> tuple[np.ndarray, ...]:
    """Read a Medit `.mesh` and source per-tet arrays, preserving element order.

    Negative tet winding is corrected in place because Newton skips such elements.
    No tetrahedron, vertex, or part is removed or reindexed.
    """
    mesh_path = Path(mesh_path)
    tokens = mesh_path.read_text().split()
    try:
        vertex_start = tokens.index("Vertices")
        vertex_count = int(tokens[vertex_start + 1])
        vertex_data = np.asarray(tokens[vertex_start + 2 : vertex_start + 2 + 4 * vertex_count], dtype=np.float64)
        vertices = vertex_data.reshape(vertex_count, 4)[:, :3].copy()
        tet_start = tokens.index("Tetrahedra")
        tet_count = int(tokens[tet_start + 1])
        tet_data = np.asarray(tokens[tet_start + 2 : tet_start + 2 + 5 * tet_count], dtype=np.int64)
        tets = tet_data.reshape(tet_count, 5)[:, :4].copy() - 1
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{mesh_path}: invalid Medit vertex/tetrahedron sections") from exc
    if len(vertices) < 4 or len(tets) < 1 or not np.isfinite(vertices).all():
        raise ValueError(f"{mesh_path}: invalid vertices or empty volume")
    if tets.min() < 0 or tets.max() >= len(vertices):
        raise ValueError(f"{mesh_path}: tetrahedron index outside {len(vertices)} vertices")

    with np.load(params_path) as source:
        required = ("tet_part_labels", "tet_E_nu", "tet_density")
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"{params_path}: missing {missing}")
        labels = np.asarray(source["tet_part_labels"], dtype=np.int32).copy()
        elastic = np.asarray(source["tet_E_nu"], dtype=np.float64).copy()
        density = np.asarray(source["tet_density"], dtype=np.float64).copy()
    if labels.shape != (len(tets),) or elastic.shape != (len(tets), 2) or density.shape != (len(tets),):
        raise ValueError(f"{params_path}: per-tet arrays do not match {len(tets)} tetrahedra")
    if not np.isfinite(elastic).all() or not np.isfinite(density).all():
        raise ValueError(f"{params_path}: material arrays contain non-finite values")
    if np.any(elastic[:, 0] <= 0) or np.any((elastic[:, 1] < 0) | (elastic[:, 1] >= 0.5)):
        raise ValueError(f"{params_path}: E or Poisson ratio is outside the FEM range")
    if np.any(density <= 0):
        raise ValueError(f"{params_path}: density must be positive")

    det = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]])
    if np.any(~np.isfinite(det)) or np.any(np.abs(det) <= 1e-15):
        raise ValueError(f"{mesh_path}: degenerate tetrahedra")
    negative = det < 0
    tets[negative, :2] = tets[negative, 1::-1]
    if np.unique(tets).size != len(vertices):
        raise ValueError(f"{mesh_path}: isolated vertices would have zero mass")
    # A single file can contain several disconnected solids. For a fused
    # heterogeneous object, every tetrahedron must share faces through one
    # volume, and no face may belong to more than two tetrahedra.
    parent = list(range(len(tets)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    face_owner: dict[tuple[int, int, int], int] = {}
    face_counts: dict[tuple[int, int, int], int] = {}
    for index, tet in enumerate(tets):
        for face in (tet[[0, 1, 2]], tet[[0, 1, 3]], tet[[0, 2, 3]], tet[[1, 2, 3]]):
            key = tuple(sorted(int(vertex) for vertex in face))
            count = face_counts.get(key, 0) + 1
            if count > 2:
                raise ValueError(f"{mesh_path}: non-manifold tetrahedron face {key}")
            face_counts[key] = count
            if count == 1:
                face_owner[key] = index
            else:
                parent[root(index)] = root(face_owner[key])
    if len({root(index) for index in range(len(tets))}) != 1:
        raise ValueError(f"{mesh_path}: tetrahedra do not form one face-connected volume")
    return vertices, tets, labels, elastic, density


@dataclass(frozen=True, kw_only=True)
class HeterogeneousSoftSpec:
    """Paths and placement for a single connected labelled FEM object.

    `pose` places the source mesh origin in the work-surface frame. Young's
    modulus, Poisson ratio, and density come exclusively from `params_path`.
    `damping` is a solver setting, not a source material estimate.
    """

    stage: ClassVar[str] = "deformables"
    entity: str = "heterogeneous_soft"
    name: str = "heterogeneous_object"
    mesh_path: str = ""
    params_path: str = ""
    materials_path: str = ""
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: float = 1.0
    damping: float = 5e3
    """Viscous tet damping [Pa·s] -- Newton's ``k_damp`` -- one value for the whole
    mesh, so the loop and the ten-times-softer body resist deformation alike.
    """
    particle_radius: float = 0.0007
    contact_ke: float = 30000.0
    contact_kd: float = 25.0
    contact_mu: float = 0.9
    settle_frames: int = 0
    settle_substeps: int = 0
    particle_sets: tuple = ()

    def __post_init__(self) -> None:
        if not self.mesh_path or not self.params_path or not self.materials_path:
            raise ValueError("heterogeneous soft body needs mesh, parameter, and material paths")
        if self.scale <= 0 or self.particle_radius <= 0 or self.damping < 0:
            raise ValueError("scale and particle radius must be positive; damping cannot be negative")


@register_entity("heterogeneous_soft")
class EntityHeterogeneousSoft(ParticleMixin, Entity):
    """A labelled volume built as one FEM mesh, including its fused part interfaces."""

    spec: HeterogeneousSoftSpec
    domain = "soft"

    def __init__(self, spec: HeterogeneousSoftSpec, *, world: int = 0) -> None:
        super().__init__(spec, world=world)
        self.particles = np.zeros(0, dtype=np.int32)
        self.sets: dict[str, np.ndarray] = {}
        self.probe = None
        self.rest: dict[str, np.ndarray] = {}

    def on_scene_ready(self, pipeline) -> None:
        # This asset has no named selectors or pre-simulation settling.
        self.join_settle(pipeline)

    def update_probe(self, state: newton.State, step: int | None = None, *, readback=None):
        if self.probe is None:
            raise RuntimeError(f"{self.name}: particle probe is not bound")
        if readback is not None and getattr(readback, "particle_q", None) is not None and len(readback.particle_q):
            self.probe.update_arrays(readback.particle_q, readback.particle_qd, step)
        else:
            self.probe.update(state, step)
        return self.probe

    def build(self, builder: newton.ModelBuilder, episode) -> None:
        del episode
        spec = self.spec
        vertices, tets, labels, elastic, density = load_labelled_tets(spec.mesh_path, spec.params_path)
        source = json.loads(Path(spec.materials_path).read_text())
        parts = {int(part["part_index"]): part for part in source["predictions"]}
        if set(parts) != set(np.unique(labels).tolist()):
            raise ValueError(f"{spec.materials_path}: part IDs disagree with tetrahedron labels")
        for part_id, part in parts.items():
            rows = labels == part_id
            expected = (part["youngs_modulus_pa"], part["poisson_ratio"], part["density_kg_m3"])
            actual = np.column_stack((elastic[rows], density[rows]))
            if not np.allclose(actual, expected, rtol=1e-6, atol=0):
                raise ValueError(f"{spec.materials_path}: part {part_id} disagrees with the per-tet arrays")

        young, nu = elastic.T
        mu = young / (2.0 * (1.0 + nu))
        lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        volume = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) * spec.scale**3 / 6.0
        masses = np.zeros(len(vertices), dtype=np.float64)
        np.add.at(masses, tets.ravel(), np.repeat(density * volume / 4.0, 4))
        if np.any(masses <= 0):
            raise ValueError(f"{spec.mesh_path}: zero-mass vertices")

        # Newton's outer triangles are generated from this same tet order. A
        # sorted face key maps each visible triangle back to its source part.
        faces: dict[tuple[int, int, int], list[int]] = {}
        for tet, label in zip(tets, labels, strict=True):
            for face in (tet[[0, 1, 2]], tet[[0, 1, 3]], tet[[0, 2, 3]], tet[[1, 2, 3]]):
                faces.setdefault(tuple(sorted(int(i) for i in face)), []).append(int(label))
        boundary = newton.TetMesh.compute_surface_triangles(tets.ravel()).reshape(-1, 3)
        colors = (
            np.asarray(
                [parts[faces[tuple(sorted(int(i) for i in face))][0]]["part_color_rgb"] for face in boundary],
                dtype=np.float32,
            )
            / 255.0
        )

        placed = self.placed((*spec.pose, *spec.quat))
        particle_start = int(builder.particle_count)
        tet_start = int(builder.tet_count)
        builder.add_soft_mesh(
            pos=wp.vec3(*[float(x) for x in placed[:3]]),
            rot=wp.quat(*[float(x) for x in placed[3:]]),
            scale=float(spec.scale),
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=tets.ravel(),
            density=1.0,
            k_mu=mu,
            k_lambda=lam,
            k_damp=float(spec.damping),
            particle_radius=spec.particle_radius,
            color=colors,
            label=spec.name,
        )
        if builder.tet_count - tet_start != len(tets) or builder.particle_count - particle_start != len(vertices):
            raise RuntimeError(f"{spec.name}: Newton discarded source elements or vertices")
        builder.particle_mass[particle_start : particle_start + len(vertices)] = masses.tolist()
        self.tet_range = (tet_start, int(builder.tet_count))
        self.tet_part_labels = labels
        self.part_materials = parts
        self.source_vertices = vertices
        self.source_tets = tets
        self.source_elastic = elastic
        self.source_density = density
        self.source_mass = masses

    def host_actor(self):
        raise ValueError(f"{self.name}: labelled FEM needs a Newton particle solver; route soft to VBD")

    def apply_model_materials(self, model: newton.Model) -> None:
        # Newton VBD exposes one soft friction coefficient for this model. The
        # source's per-part values stay on `part_materials`; the loop value is
        # used here because the first task grasps the loop.
        model.soft_contact_ke = float(self.spec.contact_ke)
        model.soft_contact_kd = float(self.spec.contact_kd)
        model.soft_contact_mu = float(self.spec.contact_mu)
