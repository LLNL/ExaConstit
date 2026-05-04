"""3D boundary classifier — Phase 3.3.B of the architecture doc.

WHAT
----
``BoundaryClassifier3D`` consumes a 3D ``ParMesh`` + 3D vector
``ParFiniteElementSpace`` (vdim = 3) and produces:

* 8  ``CornerInfo3D`` records (one per box vertex)
* 12 ``EdgeInfo3D`` records (4 edges per axis × 3 axes)
* 6  ``FaceInfo3D`` records (one per box face) with their face-element
  lists already populated as ``QuadFaceElement`` / ``TriFaceElement``
  objects (per-element sentinel-tagged gtdofs + boundary tags applied)

These are pure-Python objects that downstream code consumes without
holding a ParSubMesh reference. Every rank holds the same replicated
classification — same data on rank 0 and rank N-1 — so downstream
constraint assembly is rank-symmetric.

WHY
---
Phase 3.3.C (``ConstraintBuilder3D``) walks these objects to build
nine 1D edge-mortar blocks (via the Phase-3.3.A-generalised
``MortarAssembler2D``) and three 2D face-mortar blocks (via the
Phase-3.2.B ``QuadFaceMortarAssembler`` / ``TriFaceMortarAssembler``).
By splitting "classification" from "assembly", we keep the assembly
layer pure-Python and unit-testable.

DESIGN
------
1. ``ParSubMesh.CreateFromBoundary(parent, all_attrs)`` builds ONE
   submesh holding the entire boundary. The parent-mapping APIs
   (``GetParentVertexIDMap``, ``GetParentElementIDMap``) give us the
   back-mapping in O(1) per vertex / element.

2. **Wirebasket classification by attribute-set cardinality.** For
   each submesh vertex, the set of distinct parent-boundary-attributes
   among its adjacent submesh elements has cardinality:
       3 → box corner   (vertex sits on 3 faces)
       2 → box edge     (vertex sits on 2 faces, i.e. on a face-pair edge)
       1 → face interior (vertex sits on exactly 1 face)
   This generalises naturally to higher-dimensional domains and works
   for both hex and tet meshes since boundary attributes are assigned
   per face element, not per vertex.

3. **AllGather** all per-rank vertex records (coord + per-component
   parent global TDOFs + parent attribute set) so every rank has the
   same global view. AllGather face-element records too, so every
   rank can walk the same `face_elements` list.

4. **Per-face-element gtdof sentinel rewriting.** Once the per-vertex
   classification is known, we rewrite each face element's gtdofs
   list — replacing entries with -1 (corner) or -2 (edge) where
   appropriate, so the Phase-3.2.B assembler drops those rows
   automatically per the ``types_3d`` sentinel convention.

REFERENCES
----------
* MORTAR_PBC_ARCHITECTURE.md §11.8 Phase 3.3.B (this layer).
* MORTAR_PBC_ARCHITECTURE.md §11.6 (face-mortar geometric matching).
* MORTAR_PBC_ARCHITECTURE.md §10.4 (distributed-driver invariants —
  observed here for all collective calls).
* mortar_pbc/boundary_2d.py (the 2D pattern this generalises).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

import numpy as np

# MFEM and mpi4py are imported lazily inside `BoundaryClassifier3D.__init__`
# (and the few methods that actually use them). The bulk of the class —
# all the topology helpers, sentinel rewriting, CCW reordering — is pure
# Python and is unit-testable without a parallel MFEM stack.
if TYPE_CHECKING:
    import mfem.par as mfem  # noqa: F401  (only for type hints below)

from .types_3d import (
    CornerInfo3D,
    EdgeInfo3D,
    FaceInfo3D,
    QuadFaceElement,
    TriFaceElement,
)


__all__ = ["BoundaryClassifier3D"]


# =============================================================================
# Constants — boundary attribute conventions and naming
# =============================================================================
#
# MakeCartesian3D's boundary attribute convention (1-indexed in MFEM):
#     1 = bottom (y = y_min)
#     2 = front  (z = z_min)
#     3 = right  (x = x_max)
#     4 = back   (z = z_max)
#     5 = left   (x = x_min)
#     6 = top    (y = y_max)
#
# (See mortar_pbc/types_3d.py header for the documented convention.)

# Face-label CONVENTIONS used throughout this module. The (label, perp_axis,
# is_mortar) tuples are LOGICAL definitions that don't depend on MFEM's
# internal boundary-attribute numbering. The classifier discovers the
# mapping `attribute integer -> label` at runtime by inspecting actual
# parent-mesh vertex coordinates, NOT by hardcoding to MFEM's
# `MakeCartesian3D` attribute order — which differs between MFEM versions
# and between hex/tet element types.
#
# Canonical labels (this is what we control; mapping to MFEM attrs is
# discovered):
#     "bottom" : at  y_min, perp = y
#     "top"    : at  y_max, perp = y
#     "front"  : at  z_min, perp = z
#     "back"   : at  z_max, perp = z
#     "left"   : at  x_min, perp = x
#     "right"  : at  x_max, perp = x
#
# The (axis, extreme) -> label canonical mapping used by the runtime
# discovery in `_discover_face_label_by_attr`:
_AXIS_EXTREME_TO_LABEL: Dict[Tuple[str, str], str] = {
    ("y", "min"): "bottom",
    ("y", "max"): "top",
    ("z", "min"): "front",
    ("z", "max"): "back",
    ("x", "min"): "left",
    ("x", "max"): "right",
}

# Mortar/nonmortar assignment per face pair. Convention (locked here):
#     mortar = top, right, back     (the "high" side along each axis)
#     nonmortar  = bottom, left, front  (the "low" side along each axis)
# This matches the 2D convention and the 3D RVE literature default.
_FACE_PAIRS: List[Tuple[str, str]] = [
    ("top",   "bottom"),   # y-pair
    ("right", "left"),     # x-pair
    ("back",  "front"),    # z-pair
]
_MORTAR_LABELS: Set[str] = {pair[0] for pair in _FACE_PAIRS}

# Each face's perpendicular axis and parametric axes.
_FACE_AXES: Dict[str, Tuple[str, Tuple[str, str]]] = {
    "bottom": ("y", ("x", "z")),
    "top":    ("y", ("x", "z")),
    "front":  ("z", ("x", "y")),
    "back":   ("z", ("x", "y")),
    "left":   ("x", ("y", "z")),
    "right":  ("x", ("y", "z")),
}

# Box-edge labels: 12 edges, 4 per axis. Naming convention is
# {axis}-{adjacent-face1}-{adjacent-face2} where the two adjacent faces
# are sorted by attribute integer. The classifier exposes the
# attribute-to-label mapping via `self._face_label_by_attr` (built at
# init), so `_edge_label` is now a method, not a module-level function.


# Edge mortar/nonmortar assignment. Convention: an edge is "mortar" if both
# of its adjacent faces are nonmortars, OR if the edge sits at the
# intersection of a mortar and a nonmortar but on the corner-of-corners
# closest to the high-coord side. The simpler workable rule:
#   mortar edge  = both adjacent faces are nonmortars (low-low corner).
#   nonmortar edges  = the other 3 parallel edges (low-high, high-low, high-high).
# This gives 1 mortar + 3 nonmortars per direction × 3 directions = 12 edges,
# 9 mortar-nonmortar constraint pairs. (This convention matches §11.5 of
# the architecture doc.)


# =============================================================================
# Internal record class for AllGather'd boundary-vertex data
# =============================================================================

class _VertexRecord:
    """One record per UNIQUE submesh-vertex (parent_vertex_id key).

    After AllGather, each rank has the full list. Records are
    deduplicated by parent_vertex_id (the parent ParMesh vertex
    index, which is globally unique within a single ParMesh).

    Attributes
    ----------
    parent_vertex_id : int
        Index into parent ParMesh's vertex array.
    coord : (3,) np.float64
        Physical coordinates.
    gtdof_xyz : (3,) np.int64
        Parent global TDOFs of the (x, y, z) components at this vertex.
    parent_attrs : frozenset of int
        Set of parent boundary attributes adjacent to this vertex.
        Cardinality 1 ⇒ face-interior, 2 ⇒ box-edge, 3 ⇒ box-corner.
    """
    __slots__ = ("parent_vertex_id", "coord", "gtdof_xyz", "parent_attrs")

    def __init__(self, pvid: int, coord: np.ndarray,
                 gtdof_xyz: np.ndarray, parent_attrs: frozenset):
        self.parent_vertex_id = int(pvid)
        self.coord = np.asarray(coord, dtype=np.float64)
        self.gtdof_xyz = np.asarray(gtdof_xyz, dtype=np.int64)
        self.parent_attrs = parent_attrs


class _FaceElementRecord:
    """One record per submesh element on the boundary.

    AllGather'd to all ranks so every rank can build the same
    `face_elements` lists.

    Attributes
    ----------
    parent_attr : int
        Which face-attribute (1..6) this element belongs to.
    geometry_kind : str
        "quad" (4 vertices) or "tri" (3 vertices).
    parent_vertex_ids : tuple of int
        Vertex IDs (parent ParMesh indices), in the order MFEM gives
        for the boundary element. The classifier later reorders them
        to CCW viewed from the OUTWARD normal of the face.
    coords : (n, 3) np.float64
        Physical coordinates of the vertices, same order as
        parent_vertex_ids.
    """
    __slots__ = ("parent_attr", "geometry_kind", "parent_vertex_ids", "coords")

    def __init__(self, parent_attr: int, geometry_kind: str,
                 parent_vertex_ids: Tuple[int, ...], coords: np.ndarray):
        self.parent_attr = int(parent_attr)
        self.geometry_kind = geometry_kind
        self.parent_vertex_ids = tuple(int(v) for v in parent_vertex_ids)
        self.coords = np.asarray(coords, dtype=np.float64)


# =============================================================================
# BoundaryClassifier3D
# =============================================================================

class BoundaryClassifier3D:
    """Classify the boundary of a 3D ``ParMesh`` into corners / edges / faces.

    Constructs the classification at __init__ time. After construction:

        * ``classifier.corners``  — Dict[str, CornerInfo3D] (8 entries)
        * ``classifier.edges``    — Dict[str, EdgeInfo3D]   (12 entries)
        * ``classifier.faces``    — Dict[str, FaceInfo3D]   (6 entries)

    The dicts are keyed by label strings. Corner labels are the
    8-char tuples used by ``CornerInfo3D`` ("blf", "brf", "tlf",
    "trb", ...; see types_3d.py for the full list). Edge labels follow
    the ``_edge_label`` method. Face labels are the 6 canonical strings
    keyed in ``_AXIS_EXTREME_TO_LABEL``: "bottom", "top", "front",
    "back", "left", "right". The mapping from MFEM attribute integers
    to these labels is discovered at runtime via
    ``_discover_face_label_by_attr`` and stored as
    ``self._face_label_by_attr``.

    Parameters
    ----------
    pmesh : mfem.ParMesh
        The parent 3D ParMesh.
    fes : mfem.ParFiniteElementSpace
        Vector H1, vdim = 3, on ``pmesh``. Order 1 (linear) for Phase 3.
    tol_rel : float
        Relative tolerance for coordinate comparisons (default 1e-9 of
        bbox diagonal).
    """

    def __init__(
        self,
        pmesh,
        fes,
        *,
        tol_rel: float = 1e-9,
    ) -> None:
        # Lazy imports — see module header. Importing here lets the rest
        # of this module (topology helpers, sentinel rewriting, CCW
        # reordering) be loaded and unit-tested without MFEM/mpi4py
        # available, which is essential for sandboxed test environments.
        from mpi4py import MPI
        import mfem.par as mfem
        # Stash on the instance for use in methods that need them.
        self._MPI = MPI
        self._mfem = mfem

        if pmesh.Dimension() != 3:
            raise ValueError("BoundaryClassifier3D requires a 3D mesh")
        if fes.GetVDim() != 3:
            raise ValueError(
                f"Expected a 3D vector FE space (vdim=3), got vdim={fes.GetVDim()}"
            )
        if fes.GetOrder(0) != 1:
            raise ValueError(
                "BoundaryClassifier3D currently supports order-1 H1 only "
                "(Phase 3 scope). Higher-order is Phase 6+ via §4.11 LOR."
            )

        self.pmesh = pmesh
        self.fes = fes
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nranks = self.comm.Get_size()

        # ---------- Step 1: bbox + tolerance (collective) ----------
        self._compute_bbox()
        bbox_diag = float(np.linalg.norm(self.bbox_max - self.bbox_min))
        self.tol = tol_rel * bbox_diag

        # ---------- Step 1b: discover MFEM's attribute -> label mapping -----
        # Inspect actual parent-mesh boundary elements to determine the
        # attr -> face-label mapping. Hardcoding fails because MFEM's
        # MakeCartesian3D attribute order varies between versions and
        # between hex/tet element types. See `_discover_face_label_by_attr`.
        self._face_label_by_attr: Dict[int, str] = (
            self._discover_face_label_by_attr()
        )
        self._face_attr_by_label: Dict[str, int] = {
            v: k for k, v in self._face_label_by_attr.items()
        }

        # ---------- Step 2: build the boundary ParSubMesh (collective) -----
        self.bdr_submesh = self._build_boundary_submesh()

        # ---------- Step 3: gather per-rank boundary records (collective) -----
        # vertex_records[parent_vertex_id] = _VertexRecord
        self.vertex_records: Dict[int, _VertexRecord] = {}
        self.face_element_records: List[_FaceElementRecord] = []
        self._gather_boundary_records()

        # ---------- Step 4: classify vertices into corner / edge / face -----
        # corner_pvids: list of 8 parent_vertex_ids
        # edge_pvids: dict[edge_label, sorted list of parent_vertex_ids]
        # face_pvids: dict[face_label, set of parent_vertex_ids]
        self.corners: Dict[str, CornerInfo3D] = {}
        self.edges:   Dict[str, EdgeInfo3D]   = {}
        self.faces:   Dict[str, FaceInfo3D]   = {}
        self._build_corners()
        self._build_edges()
        self._build_faces()

    # =========================================================================
    # Step 1 — bbox
    # =========================================================================
    def _compute_bbox(self) -> None:
        """Compute global RVE bounding box via Allreduce."""
        local_min = np.full(3, np.inf, dtype=np.float64)
        local_max = np.full(3, -np.inf, dtype=np.float64)
        for v in range(self.pmesh.GetNV()):
            xyz = np.array(
                [self.pmesh.GetVertexArray(v)[d] for d in range(3)],
                dtype=np.float64,
            )
            local_min = np.minimum(local_min, xyz)
            local_max = np.maximum(local_max, xyz)
        self.bbox_min = np.zeros(3, dtype=np.float64)
        self.bbox_max = np.zeros(3, dtype=np.float64)
        self.comm.Allreduce(local_min, self.bbox_min, op=self._MPI.MIN)
        self.comm.Allreduce(local_max, self.bbox_max, op=self._MPI.MAX)

    # =========================================================================
    # Step 1b — runtime discovery of MFEM's attribute-to-label mapping
    # =========================================================================
    def _discover_face_label_by_attr(self) -> Dict[int, str]:
        """Build {attr: label} by inspecting actual mesh data.

        For each boundary attribute 1..n_attrs, find one parent
        boundary element with that attribute, read its vertex coords,
        determine which axis is invariant (zero spread) and at which
        extreme (matching bbox_min vs bbox_max), then look up the
        canonical label via ``_AXIS_EXTREME_TO_LABEL``.

        Why runtime discovery instead of hardcoding
        --------------------------------------------
        MFEM's ``MakeCartesian3D`` boundary-attribute ordering is NOT
        documented as part of the API contract — it differs between
        MFEM versions and between hex vs tet element types. Hardcoding
        the mapping caused a complete face-element mis-assignment bug
        in Phase 3.3.C: attribute 1 quads (which I called "bottom")
        were actually at z=0 (i.e., front face), causing
        ``match_conforming_face_pairs`` to fail with a centroid-
        coordinate mismatch.

        Discovery is collective-free (every rank sees the same parent
        bdr_attributes; we use `pmesh.GetBdrAttribute` and
        `pmesh.GetVertexArray`), and runs once at init time. The
        result is stored as `self._face_label_by_attr`.

        Robustness notes
        ----------------
        - For meshes with non-axis-aligned boundaries, the "invariant
          axis" criterion fails. This raises explicitly so the user
          knows to extend the classifier (out of scope for Phase 3
          which targets axis-aligned RVEs only).
        - For ranks that don't own any element with a particular
          attribute, we Allreduce-MIN the discovered label across
          ranks (with -1 sentinel for "didn't find one").
        """
        n_attrs = int(self.pmesh.bdr_attributes.Max())
        # Build per-rank attr -> (axis, extreme) by inspection.
        local_findings: Dict[int, Tuple[str, str]] = {}
        for be in range(self.pmesh.GetNBE()):
            attr = int(self.pmesh.GetBdrAttribute(be))
            if attr in local_findings:
                continue
            verts = [int(v) for v in self.pmesh.GetBdrElementVertices(be)]
            coords = np.asarray([
                [self.pmesh.GetVertexArray(v)[d] for d in range(3)]
                for v in verts
            ], dtype=np.float64)
            spread = coords.max(axis=0) - coords.min(axis=0)
            invariant_axis_idx = int(np.argmin(spread))
            invariant_value = float(coords[:, invariant_axis_idx].mean())
            # Determine extreme by comparing to bbox.
            ax_name = ("x", "y", "z")[invariant_axis_idx]
            d_min = abs(invariant_value - self.bbox_min[invariant_axis_idx])
            d_max = abs(invariant_value - self.bbox_max[invariant_axis_idx])
            if d_min < d_max:
                extreme = "min"
            else:
                extreme = "max"
            # Sanity check that the spread of the invariant axis is
            # actually small (axis-aligned mesh requirement).
            if spread[invariant_axis_idx] > self.tol:
                raise RuntimeError(
                    f"BoundaryClassifier3D: boundary attribute {attr} "
                    f"is not axis-aligned. Invariant-axis spread = "
                    f"{spread[invariant_axis_idx]:.3e}, tol = {self.tol:.3e}. "
                    f"Phase 3 supports axis-aligned RVE boundaries only."
                )
            local_findings[attr] = (ax_name, extreme)

        # AllGather across ranks; each (attr -> finding) should be
        # consistent across all ranks that report it. Sanity-check
        # that the union covers all 1..n_attrs.
        all_findings: List[Dict[int, Tuple[str, str]]] = self.comm.allgather(
            local_findings
        )
        merged: Dict[int, Tuple[str, str]] = {}
        for r_dict in all_findings:
            for attr, finding in r_dict.items():
                if attr in merged and merged[attr] != finding:
                    raise RuntimeError(
                        f"BoundaryClassifier3D: inconsistent face-label "
                        f"discovery for attribute {attr}: "
                        f"{merged[attr]} vs {finding} on different ranks."
                    )
                merged[attr] = finding

        if len(merged) != n_attrs:
            missing = sorted(set(range(1, n_attrs + 1)) - set(merged))
            raise RuntimeError(
                f"BoundaryClassifier3D: discovery did not find a "
                f"boundary element for every attribute. Found "
                f"{sorted(merged)}, expected 1..{n_attrs}, missing "
                f"{missing}."
            )

        # Map (axis, extreme) -> canonical label.
        out: Dict[int, str] = {}
        seen_labels: Set[str] = set()
        for attr, (ax, extreme) in merged.items():
            label = _AXIS_EXTREME_TO_LABEL.get((ax, extreme))
            if label is None:
                raise RuntimeError(
                    f"BoundaryClassifier3D: no canonical label for "
                    f"({ax!r}, {extreme!r}) (attr {attr})."
                )
            if label in seen_labels:
                raise RuntimeError(
                    f"BoundaryClassifier3D: two attributes map to the "
                    f"same label {label!r}. Discovery: {merged}"
                )
            seen_labels.add(label)
            out[attr] = label
        return out

    def _edge_label(self, parametric_axis: str,
                    attrs: Tuple[int, int]) -> str:
        """Build an edge label like 'x-bottom-front' from the parametric
        axis and the two adjacent face attributes.

        The two attributes are sorted by integer value, then mapped to
        their face labels via the runtime-discovered mapping.
        """
        f1, f2 = sorted(attrs)
        return (f"{parametric_axis}-{self._face_label_by_attr[f1]}"
                f"-{self._face_label_by_attr[f2]}")

    # =========================================================================
    # Step 2 — boundary ParSubMesh
    # =========================================================================
    def _build_boundary_submesh(self):
        """Build a single ParSubMesh covering the full boundary.

        The submesh holds all 6 face attributes; its parent-vertex map
        is what we use to back-translate to the parent FES TDOFs.

        pyMFEM/MFEM API note (debugged via Robert's macOS run):
        ``ParSubMesh.CreateFromBoundary`` takes an ``Array<int>`` whose
        CONTENTS are the actual attribute values to select — NOT a
        boolean mask of size ``max_attr`` indexed by attr-1. With a
        mask convention `[1, 1, 1, 1, 1, 1]`, MFEM interprets the
        array as "select attribute 1, six times" and returns a submesh
        of just the bottom face (16 elements / 25 vertices for a
        4×4×4 hex). The correct usage is to fill the array with
        ``[1, 2, 3, 4, 5, 6]``, listing each attribute once.
        """
        mfem = self._mfem
        n_bdr_attrs = int(self.pmesh.bdr_attributes.Max())
        # Build an intArray of length n_bdr_attrs; entry i = attribute (i+1).
        bdr_attrs = mfem.intArray(n_bdr_attrs)
        for a in range(1, n_bdr_attrs + 1):
            bdr_attrs[a - 1] = a
        return mfem.ParSubMesh.CreateFromBoundary(self.pmesh, bdr_attrs)

    # =========================================================================
    # Step 3 — gather per-rank vertex / element records, AllGather
    # =========================================================================
    def _gather_boundary_records(self) -> None:
        """Walk submesh elements; build per-rank vertex/element records;
        AllGather; deduplicate by SNAPPED PHYSICAL COORDINATES.

        Why snap-coord keying, not parent_vertex_id keying
        ---------------------------------------------------
        ParMesh's vertex indices are RANK-LOCAL: vertex 27 on rank 0
        is unrelated to vertex 27 on rank 1. AllGather'ing records
        keyed by `parent_vertex_id` therefore collides across ranks
        and produces nonsense merges. The 2D classifier solved this
        the same way: snap physical coordinates to a tolerance grid
        (`round(x / tol)`), use the snapped tuple as the global key,
        and merge per-rank attribute sets and TDOF tuples.

        pyMFEM API notes (verified against pyMFEM 7e99b925 on macOS):
            * ``Mesh.GetElementVertices(i)`` returns the vertex-id list
              directly — UNARY method.
            * ``ParFiniteElementSpace.GetVertexDofs(v)`` returns the
              SCALAR vertex DOF list directly (one element for P1).
              Per-component LDOFs come from ``DofToVDof(s_ldof, c)``,
              which respects byNODES vs byVDIM ordering automatically.
            * ``GetGlobalTDofNumber(ldof)`` is exposed and gives the
              global TDOF directly (matching the 2D classifier's
              proven-at-np=4 pattern). Returns -1 if the LDOF doesn't
              correspond to a true DOF on this rank.
        """
        mfem = self._mfem
        submesh = self.bdr_submesh
        parent_vmap = submesh.GetParentVertexIDMap().ToList()
        parent_emap = submesh.GetParentElementIDMap().ToList()

        # Snap-key for global vertex identity. Snap radius == tol; round
        # to nearest integer in tol-units for set-stable keying.
        snap_unit = self.tol
        def snap_key(xyz: np.ndarray) -> Tuple[int, int, int]:
            return (
                int(round(float(xyz[0]) / snap_unit)),
                int(round(float(xyz[1]) / snap_unit)),
                int(round(float(xyz[2]) / snap_unit)),
            )

        # Optional diagnostic: see what the boundary submesh and parent
        # maps look like before we build records. Surface issues like
        # wrong parent-id sense or unexpected attribute values without
        # source modifications. Toggle with MORTAR_PBC_DEBUG_CLASSIFIER=1.
        import os as _os
        _debug = _os.environ.get("MORTAR_PBC_DEBUG_CLASSIFIER", "") == "1"
        if _debug and self.rank == 0:
            print(f"  [DEBUG] boundary submesh: NE={submesh.GetNE()}, "
                  f"NV={submesh.GetNV()}")
            print(f"  [DEBUG] parent_vmap[:8] = {parent_vmap[:8]}")
            print(f"  [DEBUG] parent_emap[:8] = {parent_emap[:8]}")
            print(f"  [DEBUG] pmesh.GetNBE() = {self.pmesh.GetNBE()} (rank-local), "
                  f"pmesh.GetNE() = {self.pmesh.GetNE()} (rank-local), "
                  f"pmesh.bdr_attributes.Max() = "
                  f"{int(self.pmesh.bdr_attributes.Max())}")
            attr_dist_via_submesh = {}
            for sub_elem_idx in range(submesh.GetNE()):
                pid = parent_emap[sub_elem_idx]
                a = int(self.pmesh.GetBdrAttribute(pid))
                attr_dist_via_submesh[a] = attr_dist_via_submesh.get(a, 0) + 1
            print(f"  [DEBUG] attr distribution via parent_emap: "
                  f"{attr_dist_via_submesh}")

        # Per-rank tally: snap_key -> dict(coord, attrs, gtdofs)
        # gtdofs starts as [-1, -1, -1]; only ranks owning a component
        # fill in a positive index. Across ranks, the AllGather merge
        # picks up any rank's positive value per component.
        local_vert_data: Dict[Tuple[int, int, int], Dict] = {}
        # Per-rank face element records (will dedup post-AllGather).
        local_face_records: List[Tuple] = []

        for sub_elem_idx in range(submesh.GetNE()):
            parent_bdr_id = parent_emap[sub_elem_idx]
            parent_attr = int(self.pmesh.GetBdrAttribute(parent_bdr_id))

            sub_vert_ids = [int(v) for v in submesh.GetElementVertices(sub_elem_idx)]
            elem_coords: List[np.ndarray] = []
            elem_snap_keys: List[Tuple[int, int, int]] = []

            for sv in sub_vert_ids:
                pv = parent_vmap[sv]
                xyz = np.array(
                    [self.pmesh.GetVertexArray(pv)[d] for d in range(3)],
                    dtype=np.float64,
                )
                key = snap_key(xyz)
                elem_coords.append(xyz)
                elem_snap_keys.append(key)
                # Tally the vertex.
                if key not in local_vert_data:
                    # First time we see this vertex on this rank — look
                    # up its TDOFs via the parent FES.
                    scalar_ldofs = [int(d) for d in self.fes.GetVertexDofs(pv)]
                    gtdofs = [-1, -1, -1]
                    if scalar_ldofs:
                        s_ldof = scalar_ldofs[0]    # P1: one scalar DOF / vertex
                        for c in range(3):
                            try:
                                comp_ldof = self.fes.DofToVDof(s_ldof, c)
                            except Exception:
                                # Fallback: byNODES math.
                                n_scalar_tdofs = self.fes.GetNDofs()
                                comp_ldof = c * n_scalar_tdofs + s_ldof
                            if comp_ldof >= 0:
                                g = int(self.fes.GetGlobalTDofNumber(comp_ldof))
                                if g >= 0:
                                    gtdofs[c] = g
                    local_vert_data[key] = {
                        "coord": xyz.copy(),
                        "attrs": {parent_attr},
                        "gtdofs": gtdofs,
                    }
                else:
                    local_vert_data[key]["attrs"].add(parent_attr)

            n_v = len(sub_vert_ids)
            if n_v == 4:
                geom = "quad"
            elif n_v == 3:
                geom = "tri"
            else:
                raise RuntimeError(
                    f"BoundaryClassifier3D: face element with {n_v} vertices "
                    f"(expected 3 or 4); only quad-4 and tri-3 face elements "
                    f"are supported in Phase 3.3."
                )
            local_face_records.append((
                parent_attr,
                geom,
                tuple(elem_snap_keys),    # snap-key tuple for cross-rank dedup
                np.asarray(elem_coords, dtype=np.float64).tolist(),
            ))

        # Pack per-rank vertex data for AllGather (snap_key tuple is
        # hashable & serialisable).
        local_vert_pack = [
            (key, data["coord"].tolist(), sorted(data["attrs"]), data["gtdofs"])
            for key, data in local_vert_data.items()
        ]

        # AllGather (collective; all ranks, NO `if rank == 0:` per §10.4).
        all_vert_packs = self.comm.allgather(local_vert_pack)
        all_face_packs = self.comm.allgather(local_face_records)

        # Merge vertex records by snap-key. For each key:
        #   - union the parent_attrs set across all ranks
        #   - per-component gtdof: take the first positive value
        #     (each TDOF is owned by exactly one rank, but the FES's
        #     ldof->gtdof query returns the same global index from
        #     any rank that knows about the vertex; we keep the first
        #     positive answer encountered).
        # Use a synthetic running parent_vertex_id (just a stable counter)
        # for downstream dataclasses — the actual parent vertex index is
        # rank-local and not meaningful globally, but we need SOME unique
        # int for the dataclass field.
        merged: Dict[Tuple[int, int, int], _VertexRecord] = {}
        for rank_pack in all_vert_packs:
            for key, coord, attr_list, gtdofs_list in rank_pack:
                key_t = tuple(key)
                gtdofs_arr = np.asarray(gtdofs_list, dtype=np.int64)
                if key_t in merged:
                    existing = merged[key_t]
                    existing.parent_attrs = frozenset(
                        existing.parent_attrs | set(attr_list)
                    )
                    for c in range(3):
                        if existing.gtdof_xyz[c] < 0 and gtdofs_arr[c] >= 0:
                            existing.gtdof_xyz[c] = int(gtdofs_arr[c])
                else:
                    merged[key_t] = _VertexRecord(
                        pvid=len(merged),     # stable synthetic id
                        coord=np.asarray(coord, dtype=np.float64),
                        gtdof_xyz=gtdofs_arr.copy(),
                        parent_attrs=frozenset(attr_list),
                    )

        # Validate.
        bad = [(k, rec) for k, rec in merged.items()
               if any(rec.gtdof_xyz[c] < 0 for c in range(3))]
        if bad:
            sample = [
                f"      key={k} coord={rec.coord.tolist()} "
                f"gtdofs={rec.gtdof_xyz.tolist()} attrs={sorted(rec.parent_attrs)}"
                for k, rec in bad[:5]
            ]
            raise RuntimeError(
                f"BoundaryClassifier3D: {len(bad)} boundary vertex(es) did "
                f"not get a TDOF for at least one component across all "
                f"ranks.\n"
                f"  Total merged: {len(merged)}\n"
                f"  Samples (first 5):\n" + "\n".join(sample)
            )

        # Convert merged dict back to {synthetic_pvid -> _VertexRecord}
        # keyed mapping, since the rest of the code uses that interface.
        # Also keep a snap_key -> synthetic_pvid lookup for face-element
        # processing (translates element snap-keys to vertex records).
        self.vertex_records = {rec.parent_vertex_id: rec for rec in merged.values()}
        self._snap_key_to_pvid: Dict[Tuple[int, int, int], int] = {
            k: rec.parent_vertex_id for k, rec in merged.items()
        }

        # Merge face records, dedup by (parent_attr, sorted snap-key tuple).
        # Each boundary face element on the parent mesh appears in
        # exactly one rank's local list, but ranks may have ghost
        # boundary elements at shared faces (the parent_vertex IDs
        # would differ but the snap-keys are the same).
        face_seen: Set[Tuple[int, Tuple[Tuple[int, int, int], ...]]] = set()
        face_records: List[_FaceElementRecord] = []
        for rank_pack in all_face_packs:
            for parent_attr, geom, snap_keys_tuple, coords_list in rank_pack:
                snap_keys = tuple(tuple(k) for k in snap_keys_tuple)
                # Dedup key: attr + sorted(snap_keys).
                dedup_key = (parent_attr, tuple(sorted(snap_keys)))
                if dedup_key in face_seen:
                    continue
                face_seen.add(dedup_key)
                # Build a parent_vertex_ids tuple of synthetic pvids from
                # the snap-key map (preserves face-element local-node order).
                pvids = tuple(self._snap_key_to_pvid[k] for k in snap_keys)
                face_records.append(_FaceElementRecord(
                    parent_attr=parent_attr,
                    geometry_kind=geom,
                    parent_vertex_ids=pvids,
                    coords=np.asarray(coords_list, dtype=np.float64),
                ))
        self.face_element_records = face_records

        if _debug and self.rank == 0:
            from collections import Counter
            cardinality_dist = Counter(
                len(r.parent_attrs) for r in self.vertex_records.values()
            )
            attr_total = Counter()
            for rec in self.face_element_records:
                attr_total[rec.parent_attr] += 1
            print(f"  [DEBUG] post-merge: {len(self.vertex_records)} unique "
                  f"boundary vertices")
            print(f"  [DEBUG] cardinality distribution: {dict(cardinality_dist)}")
            print(f"  [DEBUG] face-element attr distribution: "
                  f"{dict(attr_total)} (total {sum(attr_total.values())})")

    # =========================================================================
    # Step 4a — corners (8 total, |attr_set| == 3)
    # =========================================================================
    def _build_corners(self) -> None:
        """Identify the 8 corner vertices and build CornerInfo3D records.

        Corner vertices have |parent_attrs| == 3. There should be
        exactly 8 of them; coord-match each against the bbox to assign
        a label.
        """
        corner_records = [
            r for r in self.vertex_records.values()
            if len(r.parent_attrs) == 3
        ]
        if len(corner_records) != 8:
            # Diagnostic: tally the |attr_set| distribution and dump the
            # first few records so we can see exactly what the upstream
            # gather actually produced.
            from collections import Counter
            cardinality_dist = Counter(
                len(r.parent_attrs) for r in self.vertex_records.values()
            )
            sample = list(self.vertex_records.values())[:6]
            sample_str = "\n".join(
                f"      pv={r.parent_vertex_id} coord={r.coord.tolist()} "
                f"attrs={sorted(r.parent_attrs)}"
                for r in sample
            )
            raise RuntimeError(
                f"BoundaryClassifier3D: expected 8 corner vertices "
                f"(|attr_set| == 3), found {len(corner_records)}. Mesh "
                f"may not be a topologically axis-aligned box.\n"
                f"  total boundary vertices gathered: {len(self.vertex_records)}\n"
                f"  attr-set cardinality distribution: {dict(cardinality_dist)}\n"
                f"  bbox: min={self.bbox_min.tolist()} max={self.bbox_max.tolist()}\n"
                f"  first 6 vertex records (sample):\n{sample_str}"
            )

        # Coord-match against bbox-corner targets.
        x_min, y_min, z_min = self.bbox_min
        x_max, y_max, z_max = self.bbox_max
        # Label convention per CornerInfo3D: "blf" = bottom-left-front,
        # "brf" = bottom-right-front, ..., 8 labels total.
        # Row 1: bottom (y_min) — blf, brf, blb, brb
        # Row 2: top    (y_max) — tlf, trf, tlb, trb
        # Where: l/r = x_min / x_max; f/b = z_min / z_max.
        corner_targets = {
            "blf": (x_min, y_min, z_min),
            "brf": (x_max, y_min, z_min),
            "blb": (x_min, y_min, z_max),
            "brb": (x_max, y_min, z_max),
            "tlf": (x_min, y_max, z_min),
            "trf": (x_max, y_max, z_min),
            "tlb": (x_min, y_max, z_max),
            "trb": (x_max, y_max, z_max),
        }
        for label, target in corner_targets.items():
            tgt = np.asarray(target, dtype=np.float64)
            best = None
            best_dist = np.inf
            for r in corner_records:
                d = float(np.linalg.norm(r.coord - tgt))
                if d < best_dist:
                    best_dist = d
                    best = r
            if best is None or best_dist > self.tol:
                raise RuntimeError(
                    f"BoundaryClassifier3D: no corner record within tol="
                    f"{self.tol} of target {target} for label {label!r}."
                )
            self.corners[label] = CornerInfo3D(
                label=label,
                coord=best.coord.copy(),
                gtdof_x=int(best.gtdof_xyz[0]),
                gtdof_y=int(best.gtdof_xyz[1]),
                gtdof_z=int(best.gtdof_xyz[2]),
            )

    # =========================================================================
    # Step 4b — edges (12 total, |attr_set| == 2)
    # =========================================================================
    def _build_edges(self) -> None:
        """Identify the 12 box edges and build EdgeInfo3D records.

        Box-edge vertices have |parent_attrs| == 2. Each pair of
        attributes (a1, a2) corresponds to exactly one box edge (4 of
        them are at fixed parametric_axis values).
        """
        # Group |attr_set| == 2 vertices by their (sorted) attr pair.
        edge_groups: Dict[Tuple[int, int], List[_VertexRecord]] = {}
        for r in self.vertex_records.values():
            if len(r.parent_attrs) != 2:
                continue
            key = tuple(sorted(r.parent_attrs))
            edge_groups.setdefault(key, []).append(r)

        if len(edge_groups) != 12:
            raise RuntimeError(
                f"BoundaryClassifier3D: expected 12 distinct (attr1, attr2) "
                f"pairs for box edges, found {len(edge_groups)}."
            )

        for attr_pair, recs in edge_groups.items():
            # Determine the parametric axis: the axis along which the
            # vertices vary (the other two are constant per edge).
            param_axis = self._infer_edge_parametric_axis(recs)
            label = self._edge_label(param_axis, attr_pair)

            # Sort records along the parametric axis (interior nodes
            # only; corners are excluded by the |attr_set| == 2 filter).
            axis_idx = {"x": 0, "y": 1, "z": 2}[param_axis]
            recs_sorted = sorted(recs, key=lambda r: float(r.coord[axis_idx]))

            n_interior = len(recs_sorted)
            coords = np.zeros((n_interior, 3), dtype=np.float64)
            gtdofs_x = np.zeros(n_interior, dtype=np.int64)
            gtdofs_y = np.zeros(n_interior, dtype=np.int64)
            gtdofs_z = np.zeros(n_interior, dtype=np.int64)
            for k, r in enumerate(recs_sorted):
                coords[k] = r.coord
                gtdofs_x[k] = r.gtdof_xyz[0]
                gtdofs_y[k] = r.gtdof_xyz[1]
                gtdofs_z[k] = r.gtdof_xyz[2]

            # Edge connectivity: [(-1, 0), (0, 1), ..., (n-1, -2)].
            elements: List[Tuple[int, int]] = [(-1, 0)]
            for k in range(n_interior - 1):
                elements.append((k, k + 1))
            elements.append((n_interior - 1, -2))

            # Edge bounds along the parametric axis (= corresponding
            # bbox bounds, since the edge spans bbox_min to bbox_max).
            edge_min = float(self.bbox_min[axis_idx])
            edge_max = float(self.bbox_max[axis_idx])

            # Determine the corner labels at the two endpoints. The
            # corner sitting at (edge_min) is the one whose coord at
            # axis_idx equals edge_min and matches the other 2
            # attributes; same for edge_max.
            corner_min_label, corner_max_label = self._endpoint_corners(
                attr_pair, axis_idx, edge_min, edge_max,
            )

            # Mortar/nonmortar assignment per the rule documented above:
            # the mortar edge is the one where both adjacent faces are
            # nonmortars (the "low-low corner" edge along its parametric
            # axis). All other edges are nonmortars.
            f1, f2 = attr_pair
            f1_name = self._face_label_by_attr[f1]
            f2_name = self._face_label_by_attr[f2]
            both_nonmortars = (
                f1_name not in _MORTAR_LABELS and f2_name not in _MORTAR_LABELS
            )
            is_mortar = both_nonmortars

            self.edges[label] = EdgeInfo3D(
                label=label,
                is_mortar=is_mortar,
                parametric_axis=param_axis,
                edge_min=edge_min,
                edge_max=edge_max,
                coords=coords,
                gtdofs_x=gtdofs_x,
                gtdofs_y=gtdofs_y,
                gtdofs_z=gtdofs_z,
                elements=elements,
                corner_min_label=corner_min_label,
                corner_max_label=corner_max_label,
            )

    def _infer_edge_parametric_axis(self, recs: List[_VertexRecord]) -> str:
        """Determine which axis is the parametric one (varies along edge).

        The other two axes have constant values across all `recs`.
        Returns "x", "y", or "z".
        """
        if len(recs) == 0:
            raise RuntimeError("Cannot infer edge axis from empty vertex list")
        if len(recs) == 1:
            # Only one interior node; can't infer from variance. This
            # is a degenerate but valid case (a 1-element-along-edge
            # mesh). Fall back to attr-based: the parametric axis is
            # the one perpendicular to BOTH adjacent face normals.
            attrs = sorted(recs[0].parent_attrs)
            return self._param_axis_from_attrs(tuple(attrs))
        # Variance-based: the parametric axis has the largest spread.
        coords = np.asarray([r.coord for r in recs])
        spread = coords.max(axis=0) - coords.min(axis=0)
        axis_idx = int(np.argmax(spread))
        return ("x", "y", "z")[axis_idx]

    def _param_axis_from_attrs(self, attrs: Tuple[int, int]) -> str:
        """Given two adjacent face attributes, return the edge's parametric axis.

        Each face has a perpendicular axis (its normal direction). The
        edge's parametric axis is perpendicular to BOTH face normals,
        i.e. the unique axis not equal to either face's perp axis.
        """
        f1_name = self._face_label_by_attr[attrs[0]]
        f2_name = self._face_label_by_attr[attrs[1]]
        perp1 = _FACE_AXES[f1_name][0]
        perp2 = _FACE_AXES[f2_name][0]
        if perp1 == perp2:
            raise ValueError(
                f"Faces {f1_name!r} and {f2_name!r} share the same perp "
                f"axis {perp1!r}; they're a mortar-nonmortar pair, not "
                f"adjacent — they don't share an edge."
            )
        for ax in ("x", "y", "z"):
            if ax != perp1 and ax != perp2:
                return ax
        raise RuntimeError("Unreachable")

    def _endpoint_corners(
        self, attr_pair: Tuple[int, int], axis_idx: int,
        edge_min: float, edge_max: float,
    ) -> Tuple[str, str]:
        """Find the corner labels at the two endpoints of an edge.

        An endpoint corner is the (already-built) CornerInfo3D whose
        coord at axis_idx equals edge_min (or edge_max), AND whose
        coord at the OTHER two axes matches the constant values
        defined by attr_pair.
        """
        # Determine the constant coord values at the two non-parametric
        # axes from attr_pair.
        f1_name = self._face_label_by_attr[attr_pair[0]]
        f2_name = self._face_label_by_attr[attr_pair[1]]

        def face_value(face_name: str) -> Tuple[str, float]:
            """Return (perp_axis, plane_value) of the face."""
            perp = _FACE_AXES[face_name][0]
            ax_idx = {"x": 0, "y": 1, "z": 2}[perp]
            if face_name in ("right", "top", "back"):
                return perp, float(self.bbox_max[ax_idx])
            else:
                return perp, float(self.bbox_min[ax_idx])

        perp1, val1 = face_value(f1_name)
        perp2, val2 = face_value(f2_name)

        def find(coord_target: np.ndarray) -> str:
            for label, ci in self.corners.items():
                if (np.abs(ci.coord[0] - coord_target[0]) < self.tol
                        and np.abs(ci.coord[1] - coord_target[1]) < self.tol
                        and np.abs(ci.coord[2] - coord_target[2]) < self.tol):
                    return label
            raise RuntimeError(
                f"No corner found at {coord_target} (attr_pair = {attr_pair})"
            )

        # Build target coords: parametric axis = edge_min/edge_max,
        # other two axes = val1, val2 according to perp1, perp2.
        ax_idx_perp1 = {"x": 0, "y": 1, "z": 2}[perp1]
        ax_idx_perp2 = {"x": 0, "y": 1, "z": 2}[perp2]
        tgt_min = np.zeros(3, dtype=np.float64)
        tgt_max = np.zeros(3, dtype=np.float64)
        tgt_min[axis_idx] = edge_min
        tgt_max[axis_idx] = edge_max
        tgt_min[ax_idx_perp1] = val1
        tgt_max[ax_idx_perp1] = val1
        tgt_min[ax_idx_perp2] = val2
        tgt_max[ax_idx_perp2] = val2
        return find(tgt_min), find(tgt_max)

    # =========================================================================
    # Step 4c — faces (6 total) and per-face element lists
    # =========================================================================
    def _build_faces(self) -> None:
        """Build 6 FaceInfo3D records, each with its face_elements list.

        Per-face-element gtdofs are sentinel-rewritten: -1 for corner
        DOFs, -2 for box-edge DOFs (i.e. shared with another face).
        Boundary tags ("none", "edge-...", "corner-...") are assigned
        based on whether the element shares vertices with face
        boundaries.
        """
        # Build a corner-DOF set for fast O(1) sentinel rewriting.
        # Map: parent global TDOF -> 'corner' or 'edge' (or absent = face-interior).
        sentinel_class: Dict[int, str] = {}
        for r in self.vertex_records.values():
            if len(r.parent_attrs) == 3:
                cls = "corner"
            elif len(r.parent_attrs) == 2:
                cls = "edge"
            else:
                continue
            for c in range(3):
                sentinel_class[int(r.gtdof_xyz[c])] = cls

        # Group face element records by parent attribute.
        per_attr: Dict[int, List[_FaceElementRecord]] = {
            a: [] for a in sorted(self._face_label_by_attr)
        }
        for rec in self.face_element_records:
            per_attr[rec.parent_attr].append(rec)

        for attr in sorted(self._face_label_by_attr):
            face_label = self._face_label_by_attr[attr]
            perp_axis, param_axes = _FACE_AXES[face_label]
            ax_idx = {"x": 0, "y": 1, "z": 2}[perp_axis]
            plane_value = (
                float(self.bbox_max[ax_idx]) if face_label in ("top", "right", "back")
                else float(self.bbox_min[ax_idx])
            )
            is_mortar = face_label in _MORTAR_LABELS

            face_elems: List[object] = []
            n_quad = 0
            n_tri = 0
            interior_gtdofs_x_set: Set[int] = set()
            interior_gtdofs_y_set: Set[int] = set()
            interior_gtdofs_z_set: Set[int] = set()

            for rec in per_attr[attr]:
                # Build per-vertex gtdof tuple with sentinels applied,
                # vertices reordered to CCW-from-outward-normal.
                ordered_pvids, ordered_coords = self._reorder_face_vertices_ccw(
                    rec, face_label, perp_axis, plane_value,
                )
                ordered_gtdofs_with_sentinels: List[int] = []
                for pv in ordered_pvids:
                    vrec = self.vertex_records[pv]
                    primary_gtdof = int(vrec.gtdof_xyz[0])  # x-component primary
                    cls = sentinel_class.get(primary_gtdof, None)
                    if cls == "corner":
                        ordered_gtdofs_with_sentinels.append(-1)
                    elif cls == "edge":
                        ordered_gtdofs_with_sentinels.append(-2)
                    else:
                        ordered_gtdofs_with_sentinels.append(primary_gtdof)
                        interior_gtdofs_x_set.add(int(vrec.gtdof_xyz[0]))
                        interior_gtdofs_y_set.add(int(vrec.gtdof_xyz[1]))
                        interior_gtdofs_z_set.add(int(vrec.gtdof_xyz[2]))

                if rec.geometry_kind == "quad":
                    fe = QuadFaceElement(
                        coords=ordered_coords,
                        gtdofs=tuple(ordered_gtdofs_with_sentinels),  # type: ignore
                        parametric_axes=param_axes,
                        perpendicular_axis=perp_axis,
                        boundary_tag=self._classify_quad_boundary_tag(
                            ordered_gtdofs_with_sentinels,
                        ),
                    )
                    n_quad += 1
                elif rec.geometry_kind == "tri":
                    fe = TriFaceElement(
                        coords=ordered_coords,
                        gtdofs=tuple(ordered_gtdofs_with_sentinels),  # type: ignore
                        parametric_axes=param_axes,
                        perpendicular_axis=perp_axis,
                        boundary_tag=self._classify_tri_boundary_tag(
                            ordered_gtdofs_with_sentinels,
                        ),
                    )
                    n_tri += 1
                else:
                    raise RuntimeError(f"Unknown geometry: {rec.geometry_kind}")
                face_elems.append(fe)

            # Bounding edge labels for this face.
            bounding_edges = self._face_bounding_edge_labels(attr)

            self.faces[face_label] = FaceInfo3D(
                label=face_label,
                is_mortar=is_mortar,
                perpendicular_axis=perp_axis,
                plane_value=plane_value,
                parametric_axes=param_axes,
                n_quad_elements=n_quad,
                n_tri_elements=n_tri,
                submesh=None,   # Optional; we don't hold a ParSubMesh ref here
                face_elements=face_elems,
                interior_gtdofs_x=np.asarray(
                    sorted(interior_gtdofs_x_set), dtype=np.int64),
                interior_gtdofs_y=np.asarray(
                    sorted(interior_gtdofs_y_set), dtype=np.int64),
                interior_gtdofs_z=np.asarray(
                    sorted(interior_gtdofs_z_set), dtype=np.int64),
                bounding_edge_labels=bounding_edges,
            )

    def _reorder_face_vertices_ccw(
        self,
        rec: _FaceElementRecord,
        face_label: str,
        perp_axis: str,
        plane_value: float,
    ) -> Tuple[List[int], np.ndarray]:
        """Reorder a face element's vertices so they are CCW viewed from
        the OUTWARD normal of the face.

        Outward normal direction:
            face = "top"     : +y
            face = "bottom"  : -y
            face = "right"   : +x
            face = "left"    : -x
            face = "back"    : +z
            face = "front"   : -z

        Algorithm: project to 2D in the face's parametric plane, compute
        signed area; if it's negative w.r.t. outward normal, reverse.
        """
        perp_idx = {"x": 0, "y": 1, "z": 2}[perp_axis]
        param_axes = _FACE_AXES[face_label][1]
        a_idx = {"x": 0, "y": 1, "z": 2}[param_axes[0]]
        b_idx = {"x": 0, "y": 1, "z": 2}[param_axes[1]]
        # Outward normal sign: positive if face is at bbox_max along
        # perp axis, negative if at bbox_min.
        outward_pos = face_label in ("top", "right", "back")

        coords = rec.coords  # (n, 3)
        pvids = list(rec.parent_vertex_ids)
        # 2D projection in (a, b) plane.
        pts_2d = coords[:, [a_idx, b_idx]]

        # Compute signed area of the polygon (Shoelace).
        n = pts_2d.shape[0]
        signed_area = 0.0
        for i in range(n):
            x1, y1 = pts_2d[i]
            x2, y2 = pts_2d[(i + 1) % n]
            signed_area += (x1 * y2 - x2 * y1)
        signed_area *= 0.5
        # CCW in the (a, b) plane means signed_area > 0.
        # We want CCW from OUTWARD normal. The (a, b) -> outward-normal
        # right-hand rule: if perp_axis ordering is consistent (cross
        # product a × b = outward), then signed_area > 0 == CCW
        # from outward. The choice of (a, b) per face was set in
        # _FACE_AXES so that this holds for outward = +perp:
        #     top/right/back: cross of param_axes = +perp
        #     bottom/left/front: cross of param_axes = -perp (so we flip)
        # Reflection: when outward is -perp, we need signed_area < 0 to
        # be the "outward CCW" direction. Adjust.
        want_positive = outward_pos
        if want_positive and signed_area < 0:
            pvids = list(reversed(pvids))
            coords = coords[::-1].copy()
        elif (not want_positive) and signed_area > 0:
            pvids = list(reversed(pvids))
            coords = coords[::-1].copy()

        return pvids, coords

    @staticmethod
    def _classify_quad_boundary_tag(sentinels: List[int]) -> str:
        """Map sentinel pattern of a quad-4 face element to a Wohlmuth tag.

        Tag conventions per ``QuadFaceMortarAssembler._quad4_boundary_tag_to_sides``:
            "none"          : no sentinel vertices
            "edge-xi-low"   : local nodes 0 & 3 are sentinels (xi=-1 edge)
            "edge-xi-high"  : local nodes 1 & 2 are sentinels (xi=+1 edge)
            "edge-eta-low"  : local nodes 0 & 1 are sentinels (eta=-1 edge)
            "edge-eta-high" : local nodes 2 & 3 are sentinels (eta=+1 edge)
            "corner-LL"     : nodes 0 (or {0, 1, 3}) are sentinels  (xi-low + eta-low)
            "corner-LR"     : nodes 1 (or {0, 1, 2}) are sentinels  (xi-high + eta-low)
            "corner-UR"     : nodes 2 (or {1, 2, 3}) are sentinels  (xi-high + eta-high)
            "corner-UL"     : nodes 3 (or {0, 2, 3}) are sentinels  (xi-low + eta-high)

        Quad-4 local-node convention (CCW from outward normal):
            node 3 -- node 2     eta=+1
              |          |
            node 0 -- node 1     eta=-1
            xi=-1     xi=+1

        Sentinel patterns and their geometric meanings:
            * 0 sentinels: face-interior quad (no boundary contact).
            * 1 sentinel (corner DOF only): one local node is a box-
              corner. The L-shape formed by that node's two in-element
              neighbours is what determines the corner-XX tag.
            * 2 co-edge sentinels: one full local edge of the quad
              coincides with a face-boundary box-edge.
            * 2 diagonal sentinels: anomalous; doesn't arise on
              MakeCartesian3D meshes but we fall through to 'none'
              with the lumped-positivity guard catching any issue.
            * 3 sentinels (typical corner-of-face quad): two of its
              local edges are on box-edges AND its shared corner is
              the box corner. The single non-sentinel node is the
              "kept" node opposite that corner. Tag = corner-XX with
              XX picked so that the dropped sides match the {xi, eta}
              extents of the sentinel cluster.
            * 4 sentinels: all kept-rows would be dropped; the
              element contributes nothing. 'none' is harmless.
        """
        sentinel_locs = [i for i, s in enumerate(sentinels) if s < 0]
        n = len(sentinel_locs)
        if n == 0:
            return "none"
        if n == 1:
            i = sentinel_locs[0]
            return ("corner-LL", "corner-LR", "corner-UR", "corner-UL")[i]
        if n == 2:
            s = set(sentinel_locs)
            if s == {0, 3}: return "edge-xi-low"
            if s == {1, 2}: return "edge-xi-high"
            if s == {0, 1}: return "edge-eta-low"
            if s == {2, 3}: return "edge-eta-high"
            # Diagonal-pair sentinels ({0, 2} or {1, 3}): anomalous on
            # MakeCartesian3D meshes; lumped-positivity guards integrity.
            return "none"
        if n == 3:
            # Three sentinels = two co-edge sentinel pairs sharing a
            # corner. The 4 cases name the kept node:
            #   kept node 2 (corner-LL drops {xi-low, eta-low}) -> sentinels {0, 1, 3}
            #   kept node 3 (corner-LR drops {xi-high, eta-low}) -> sentinels {0, 1, 2}
            #   kept node 0 (corner-UR drops {xi-high, eta-high}) -> sentinels {1, 2, 3}
            #   kept node 1 (corner-UL drops {xi-low, eta-high}) -> sentinels {0, 2, 3}
            kept = (set(range(4)) - set(sentinel_locs)).pop()
            return ("corner-UR", "corner-UL", "corner-LL", "corner-LR")[kept]
        # 4 sentinels: every row dropped, element contributes nothing.
        return "none"

    @staticmethod
    def _classify_tri_boundary_tag(sentinels: List[int]) -> str:
        """Map sentinel pattern of a tri-3 to its Wohlmuth tag.

        Tag conventions per ``TriFaceMortarAssembler._tri3_boundary_tag_to_drops``:
            "none"     : no sentinel vertices
            "v0"       : vertex 0 sentinel
            "v1"       : vertex 1 sentinel
            "v2"       : vertex 2 sentinel
            "v0-v1"    : vertices 0, 1 sentinels
            "v0-v2"    : vertices 0, 2 sentinels
            "v1-v2"    : vertices 1, 2 sentinels
            "v0-v1-v2" : all 3 sentinels (rare; degenerate)
        """
        sentinel_locs = sorted(i for i, s in enumerate(sentinels) if s < 0)
        if len(sentinel_locs) == 0:
            return "none"
        return "v" + "-v".join(str(i) for i in sentinel_locs)

    def _face_bounding_edge_labels(self, face_attr: int) -> List[str]:
        """Return the 4 edge labels bounding the face with given attribute.

        Each box face has 4 bounding edges; each is shared with one
        adjacent face. The labels follow `_edge_label`.
        """
        face_label = self._face_label_by_attr[face_attr]
        # The 4 adjacent face attributes (those sharing an edge with this face).
        adjacent: List[int] = []
        for other_attr in sorted(self._face_label_by_attr):
            if other_attr == face_attr:
                continue
            other_label = self._face_label_by_attr[other_attr]
            # Two faces share an edge if their perp axes differ.
            if _FACE_AXES[face_label][0] != _FACE_AXES[other_label][0]:
                adjacent.append(other_attr)
        out: List[str] = []
        for other_attr in adjacent:
            other_label = self._face_label_by_attr[other_attr]
            # Parametric axis of the shared edge: perpendicular to BOTH
            # face normals.
            perp1 = _FACE_AXES[face_label][0]
            perp2 = _FACE_AXES[other_label][0]
            for ax in ("x", "y", "z"):
                if ax != perp1 and ax != perp2:
                    out.append(self._edge_label(ax, (face_attr, other_attr)))
                    break
        return out

    # =========================================================================
    # Public helpers for ConstraintBuilder3D (Phase 3.3.C)
    # =========================================================================
    @property
    def n_global_tdofs(self) -> int:
        """Total number of global true-DOFs in the parent FES.

        Used by ConstraintBuilder3D to size the global C matrix.
        Available on every rank because the parent FES knows its own
        global TDOF count without further collectives at access time.
        """
        return int(self.fes.GlobalTrueVSize())

    def gtdof_xyz_lookup(self) -> Dict[int, Tuple[int, int, int]]:
        """Build a lookup gtdof_x → (gtdof_x, gtdof_y, gtdof_z).

        ConstraintBuilder3D uses this to expand the primary-component
        gtdofs stored in ``FaceMortarPairBlock.nonmortar_gtdofs`` /
        ``mortar_gtdofs`` (and in the per-face-element gtdofs tuples)
        into per-component gtdofs for vdim=3 constraint rows.

        The map is built from ``vertex_records``, which holds every
        vertex's full ``gtdof_xyz`` triple. Returned as a fresh dict
        on each call (cheap; ~100 entries on a 4×4×4 RVE).
        """
        out: Dict[int, Tuple[int, int, int]] = {}
        for r in self.vertex_records.values():
            gx = int(r.gtdof_xyz[0])
            gy = int(r.gtdof_xyz[1])
            gz = int(r.gtdof_xyz[2])
            if gx >= 0:
                out[gx] = (gx, gy, gz)
        return out

    def edge_pairs(self) -> List[Tuple[str, str, str]]:
        """Return the 9 mortar-nonmortar edge pairs as (axis, mortar, nonmortar).

        For each parametric axis (x, y, z), there is 1 mortar edge
        (the one with both adjacent faces being nonmortars) and 3 nonmortar
        edges. We pair the mortar against each nonmortar individually,
        producing 9 pairs total.
        """
        mortar_by_axis: Dict[str, str] = {}
        nonmortars_by_axis: Dict[str, List[str]] = {"x": [], "y": [], "z": []}
        for label, e in self.edges.items():
            if e.is_mortar:
                if e.parametric_axis in mortar_by_axis:
                    raise RuntimeError(
                        f"Multiple mortar edges along axis "
                        f"{e.parametric_axis!r}: "
                        f"{mortar_by_axis[e.parametric_axis]!r} and "
                        f"{label!r}"
                    )
                mortar_by_axis[e.parametric_axis] = label
            else:
                nonmortars_by_axis[e.parametric_axis].append(label)
        pairs: List[Tuple[str, str, str]] = []
        for axis in ("x", "y", "z"):
            if axis not in mortar_by_axis:
                raise RuntimeError(f"No mortar edge along axis {axis!r}")
            if len(nonmortars_by_axis[axis]) != 3:
                raise RuntimeError(
                    f"Axis {axis!r}: expected 3 nonmortar edges, found "
                    f"{len(nonmortars_by_axis[axis])}"
                )
            mortar = mortar_by_axis[axis]
            for nonmortar in sorted(nonmortars_by_axis[axis]):
                pairs.append((axis, mortar, nonmortar))
        return pairs

    def face_pairs(self) -> List[Tuple[str, str, str]]:
        """Return the 3 mortar-nonmortar face pairs as (axis, mortar, nonmortar).

        One pair per perpendicular axis. Mortar/nonmortar per the §11.5
        convention: mortar = top, right, back; nonmortar = bottom, left,
        front. Encoded in the classifier's ``_FACE_PAIRS`` constant.
        """
        return [(_FACE_AXES[m][0], m, s) for m, s in _FACE_PAIRS]

    # =========================================================================
    # Diagnostic
    # =========================================================================
    def summary(self) -> str:
        """Human-readable summary, suitable for rank-0 diagnostic prints."""
        lines = ["BoundaryClassifier3D summary:"]
        lines.append(
            f"  bbox: [{self.bbox_min.tolist()}] -> [{self.bbox_max.tolist()}]"
        )
        lines.append(f"  tol:  {self.tol:.3e}")
        lines.append(
            f"  corners ({len(self.corners)}): "
            f"{sorted(self.corners.keys())}"
        )
        lines.append(f"  edges ({len(self.edges)}):")
        for lbl, e in sorted(self.edges.items()):
            lines.append(
                f"    {lbl:30s} axis={e.parametric_axis} "
                f"n_interior={e.n_nodes:4d}  mortar={e.is_mortar}"
            )
        lines.append(f"  faces ({len(self.faces)}):")
        for lbl, f in sorted(self.faces.items()):
            lines.append(
                f"    {lbl:8s}  perp={f.perpendicular_axis} "
                f"n_quad={f.n_quad_elements:4d}  n_tri={f.n_tri_elements:4d}"
                f"  mortar={f.is_mortar}"
            )
        return "\n".join(lines)
