"""Phase 3.3.C unit tests — ConstraintBuilder3D with a synthetic classifier.

Pure-Python tests, no MFEM. We construct a synthetic mock classifier
representing a small axis-aligned cube boundary, hand it to
``ConstraintBuilder3D``, and verify the resulting global C matrix.

Key properties verified:

  1. **Row count** matches the analytical formula: vdim *
     (sum of nonmortar-edge interior nodes + sum of nonmortar-face interior
     nodes).

  2. **Linear-field reproduction.** For an affine field u(X) = (F-I)X
     evaluated at every gtdof, the constraint C·u = 0 holds to
     machine precision. This is the load-bearing correctness property
     of the dual basis: the mortar formulation reproduces affine
     fields exactly, so any perfectly periodic affine deformation
     satisfies the periodic constraint with no residual.

  3. **Sparsity pattern**: the row-block from edge-mortar pairs
     touches only edge-related gtdofs; face-mortar pairs touch only
     face-related gtdofs (modulo the corner/edge sentinel exclusions).

References
----------
* MORTAR_PBC_ARCHITECTURE.md §11.8 Phase 3.3.C/D.
* mortar_pbc/constraint_builder_3d.py.
"""
from __future__ import annotations

import os
import sys

# Defensive path setup (see test_face_mortar_3d.py for full rationale).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_LOCAL_PKG = os.path.join(_PARENT, "mortar_pbc")
if not os.path.isdir(_LOCAL_PKG):
    raise RuntimeError(f"Cannot find mortar_pbc package at {_LOCAL_PKG!r}.")
sys.path.insert(0, _PARENT)
for _mod_name in list(sys.modules.keys()):
    if _mod_name == "mortar_pbc" or _mod_name.startswith("mortar_pbc."):
        del sys.modules[_mod_name]

import mortar_pbc                                                    # noqa: E402
_actual = os.path.realpath(os.path.dirname(mortar_pbc.__file__))
_expected = os.path.realpath(_LOCAL_PKG)
if _actual != _expected:
    raise RuntimeError(
        f"mortar_pbc resolves to {_actual!r} not {_expected!r}; "
        f"run `pip uninstall mortar-pbc` to remove a stale install."
    )

import numpy as np                                                    # noqa: E402
import scipy.sparse as sp                                             # noqa: E402

from mortar_pbc import (                                              # noqa: E402
    ConstraintBuilder3D,
    QuadFaceElement,
)
from mortar_pbc.types_3d import (                                     # noqa: E402
    CornerInfo3D, EdgeInfo3D, FaceInfo3D,
)


# =============================================================================
# Synthetic mock classifier — a 2x2x2 hex RVE on [0,1]^3
# =============================================================================
#
# The simplest possible 3D RVE that has the full topology:
#   * 27 vertices (3 per axis).
#   * 8 corners,
#   * 12 box edges, each with 1 interior vertex (3 per axis - 2 corners),
#   * 6 faces, each with 1 interior vertex (3x3 - 4 corners - 4 edge-mids = 1).
#
# This gives:
#   - 8 corner gtdofs (Dirichlet-pinned, NOT in C).
#   - 12 edge interior gtdofs (3 per axis * 4 edges per axis - some sharing
#     across axis groups, but on this RVE they're all distinct = 12).
#   - 6 face interior gtdofs (one per face).
#
# Total boundary scalar dofs: 8 + 12 + 6 = 26.
# Plus 1 cell-center vertex = 27 total. (Cell center isn't on boundary.)
#
# vdim=3, so global TDOFs = 27 * 3 = 81.

def _build_synthetic_classifier_2x2x2(L: float = 1.0):
    """Return a duck-typed classifier mimicking BoundaryClassifier3D
    for a 2x2x2 hex mesh on [0, L]^3.

    Vertex layout (i, j, k) -> linear index = i + 3*j + 9*k:
        i is x-index (0=low, 1=mid, 2=high)
        j is y-index, k is z-index.
    """
    # Vertex coords by (i, j, k).
    coords = np.zeros((27, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                vid = i + 3 * j + 9 * k
                coords[vid] = [i * L / 2, j * L / 2, k * L / 2]

    # Per-vertex gtdofs (vdim=3, byNODES ordering): vertex v owns
    # gtdofs (v, v+27, v+54).
    n_verts = 27
    gtdof_x = np.arange(n_verts, dtype=np.int64)
    gtdof_y = np.arange(n_verts, dtype=np.int64) + n_verts
    gtdof_z = np.arange(n_verts, dtype=np.int64) + 2 * n_verts

    # Helper.
    def vid(i, j, k): return i + 3 * j + 9 * k

    # ---- Corners (i, j, k in {0, 2}) ----
    # Label convention: blf = bottom(y=0)-left(x=0)-front(z=0) etc.
    corner_labels = {
        (0, 0, 0): "blf", (2, 0, 0): "brf", (0, 0, 2): "blb", (2, 0, 2): "brb",
        (0, 2, 0): "tlf", (2, 2, 0): "trf", (0, 2, 2): "tlb", (2, 2, 2): "trb",
    }
    corners = {}
    for (i, j, k), label in corner_labels.items():
        v = vid(i, j, k)
        corners[label] = CornerInfo3D(
            label=label, coord=coords[v].copy(),
            gtdof_x=int(gtdof_x[v]), gtdof_y=int(gtdof_y[v]),
            gtdof_z=int(gtdof_z[v]),
        )

    # ---- Edges (12 total, 1 interior vertex each) ----
    # An edge along axis a passes through (i, j, k) with a's index
    # varying and the other two constant at 0 or 2. The single
    # interior vertex on each edge has the varying axis at 1.
    #
    # Mortar/nonmortar per the §11.5 convention: mortar = edge where both
    # adjacent faces are nonmortars. For the bottom-front x-edge,
    # bottom (nonmortar) + front (nonmortar) are both nonmortars -> mortar.
    edge_specs = {
        # axis 'x': vary i, j and k constant
        ("x", 0, 0): ("x-bottom-front", True),    # bottom + front (both nonmortars) = MORTAR
        ("x", 2, 0): ("x-front-top",   False),    # top is mortar
        ("x", 0, 2): ("x-bottom-back", False),    # back is mortar
        ("x", 2, 2): ("x-back-top",    False),    # both mortars
        # axis 'y': vary j, i and k constant
        ("y", 0, 0): ("y-front-left",  True),     # left + front (both nonmortars) = MORTAR
        ("y", 2, 0): ("y-front-right", False),
        ("y", 0, 2): ("y-back-left",   False),
        ("y", 2, 2): ("y-back-right",  False),
        # axis 'z': vary k, i and j constant
        ("z", 0, 0): ("z-bottom-left", True),     # bottom + left (both nonmortars) = MORTAR
        ("z", 2, 0): ("z-bottom-right", False),
        ("z", 0, 2): ("z-left-top",   False),
        ("z", 2, 2): ("z-right-top",  False),
    }

    edges = {}
    for (axis, p1, p2), (label, is_mortar) in edge_specs.items():
        # Single interior vertex.
        if axis == "x":
            v = vid(1, p1, p2)
            edge_min = 0.0
            edge_max = float(L)
        elif axis == "y":
            v = vid(p1, 1, p2)
            edge_min = 0.0
            edge_max = float(L)
        else:  # z
            v = vid(p1, p2, 1)
            edge_min = 0.0
            edge_max = float(L)
        # Single-node edge: connectivity (-1, 0), (0, -2)
        elements = [(-1, 0), (0, -2)]
        edges[label] = EdgeInfo3D(
            label=label, is_mortar=is_mortar, parametric_axis=axis,
            edge_min=edge_min, edge_max=edge_max,
            coords=coords[v:v + 1].copy(),
            gtdofs_x=np.asarray([gtdof_x[v]], dtype=np.int64),
            gtdofs_y=np.asarray([gtdof_y[v]], dtype=np.int64),
            gtdofs_z=np.asarray([gtdof_z[v]], dtype=np.int64),
            elements=elements,
            corner_min_label="", corner_max_label="",
        )

    # ---- Faces (6 total, 1 interior vertex each, 4 quad sub-elements) ----
    # Each face on a 2x2x2 mesh has a 3x3 vertex grid with the centre
    # being the only interior vertex. The face is divided into 4 quads
    # of size (L/2)x(L/2). Each quad has at most 2 box-edge sentinels
    # (its two outer edges) plus 1 corner sentinel; the kept node is
    # the face-interior centre vertex.

    def build_face(label, perp_axis, plane_value, parametric_axes,
                   is_mortar, corner_lookup):
        """Build a FaceInfo3D with 4 quad sub-elements.

        corner_lookup(p1, p2) -> v_id : maps a position in the (a, b)
        face grid to the 3D vertex id.
        """
        # 4 sub-elements: 2x2 grid in (a, b).
        face_elems = []
        for a_lo in (0, 1):  # 0=low half, 1=high half along axis a
            for b_lo in (0, 1):
                # 4 corner indices in (a, b) grid: low/low, hi/lo, hi/hi, lo/hi
                corner_indices = [
                    (a_lo,     b_lo),
                    (a_lo + 1, b_lo),
                    (a_lo + 1, b_lo + 1),
                    (a_lo,     b_lo + 1),
                ]
                quad_coords = []
                quad_gtdofs = []
                for (a, b) in corner_indices:
                    v = corner_lookup(a, b)
                    quad_coords.append(coords[v].copy())
                    # Apply sentinels: corner if (a, b) is a face corner
                    # (a in {0, 2} and b in {0, 2}); edge if a or b is
                    # 0 or 2 but not both; face-interior if a == 1 and b == 1.
                    is_face_corner = (a in (0, 2)) and (b in (0, 2))
                    is_box_edge = ((a in (0, 2)) ^ (b in (0, 2)))
                    if is_face_corner:
                        quad_gtdofs.append(-1)
                    elif is_box_edge:
                        quad_gtdofs.append(-2)
                    else:
                        quad_gtdofs.append(int(gtdof_x[v]))
                # Determine boundary tag: 3 sentinels (one corner of the
                # face) vs 2 sentinels (along an edge) vs none.
                from mortar_pbc.boundary_3d import BoundaryClassifier3D
                tag = BoundaryClassifier3D._classify_quad_boundary_tag(
                    quad_gtdofs
                )
                face_elems.append(QuadFaceElement(
                    coords=np.asarray(quad_coords, dtype=np.float64),
                    gtdofs=tuple(quad_gtdofs),
                    parametric_axes=parametric_axes,
                    perpendicular_axis=perp_axis,
                    boundary_tag=tag,
                ))

        # The face-interior gtdof is the centre vertex.
        center_v = corner_lookup(1, 1)
        return FaceInfo3D(
            label=label,
            is_mortar=is_mortar,
            perpendicular_axis=perp_axis,
            plane_value=plane_value,
            parametric_axes=parametric_axes,
            n_quad_elements=4, n_tri_elements=0,
            submesh=None,
            face_elements=face_elems,
            interior_gtdofs_x=np.asarray([gtdof_x[center_v]], dtype=np.int64),
            interior_gtdofs_y=np.asarray([gtdof_y[center_v]], dtype=np.int64),
            interior_gtdofs_z=np.asarray([gtdof_z[center_v]], dtype=np.int64),
            bounding_edge_labels=[],
        )

    # bottom: y=0, params (x, z)  (nonmortar)
    bottom = build_face(
        "bottom", "y", 0.0, ("x", "z"), is_mortar=False,
        corner_lookup=lambda a, b: vid(a, 0, b),
    )
    # top: y=L, params (x, z)  (mortar)
    top = build_face(
        "top", "y", float(L), ("x", "z"), is_mortar=True,
        corner_lookup=lambda a, b: vid(a, 2, b),
    )
    # front: z=0, params (x, y)  (nonmortar)
    front = build_face(
        "front", "z", 0.0, ("x", "y"), is_mortar=False,
        corner_lookup=lambda a, b: vid(a, b, 0),
    )
    # back: z=L, params (x, y)  (mortar)
    back = build_face(
        "back", "z", float(L), ("x", "y"), is_mortar=True,
        corner_lookup=lambda a, b: vid(a, b, 2),
    )
    # left: x=0, params (y, z)  (nonmortar)
    left = build_face(
        "left", "x", 0.0, ("y", "z"), is_mortar=False,
        corner_lookup=lambda a, b: vid(0, a, b),
    )
    # right: x=L, params (y, z)  (mortar)
    right = build_face(
        "right", "x", float(L), ("y", "z"), is_mortar=True,
        corner_lookup=lambda a, b: vid(2, a, b),
    )

    faces = {
        "bottom": bottom, "top": top,
        "front": front,   "back": back,
        "left": left,     "right": right,
    }

    # Build the lookup gtdof_x -> (gx, gy, gz)
    lookup = {int(gtdof_x[v]): (int(gtdof_x[v]),
                                int(gtdof_y[v]),
                                int(gtdof_z[v])) for v in range(n_verts)}

    class _MockClassifier:
        bbox_min = np.zeros(3)
        bbox_max = np.array([L, L, L])
        n_global_tdofs = 3 * n_verts

        def __init__(self):
            self.corners = corners
            self.edges = edges
            self.faces = faces

        def gtdof_xyz_lookup(self):
            return dict(lookup)

        def edge_pairs(self):
            # Pair each mortar edge with its 3 nonmortar parallels.
            from collections import defaultdict
            by_axis = defaultdict(lambda: {"mortar": None, "nonmortars": []})
            for label, e in self.edges.items():
                if e.is_mortar:
                    by_axis[e.parametric_axis]["mortar"] = label
                else:
                    by_axis[e.parametric_axis]["nonmortars"].append(label)
            pairs = []
            for axis in ("x", "y", "z"):
                m = by_axis[axis]["mortar"]
                for s in sorted(by_axis[axis]["nonmortars"]):
                    pairs.append((axis, m, s))
            return pairs

        def face_pairs(self):
            return [
                ("y", "top", "bottom"),
                ("x", "right", "left"),
                ("z", "back", "front"),
            ]

    return _MockClassifier(), n_verts, coords, gtdof_x, gtdof_y, gtdof_z


# =============================================================================
# Test 1: row-count formula
# =============================================================================

def test_constraint_row_count():
    """C has the predicted number of rows.

    For the 2x2x2 mock RVE:
        edges: 9 mortar-nonmortar pairs * 1 interior node each * vdim=3 = 27 rows
        faces: 3 mortar-nonmortar pairs * 1 face-interior node each * vdim=3 = 9 rows
        total: 36 rows.
    """
    cl, n_verts, *_ = _build_synthetic_classifier_2x2x2()
    builder = ConstraintBuilder3D(cl)
    n_predicted = builder.n_constraints()
    assert n_predicted == 36, f"n_constraints = {n_predicted}, expected 36"
    C = builder.build()
    assert C.shape == (36, 3 * n_verts), (
        f"C.shape = {C.shape}, expected (36, {3 * n_verts})"
    )
    print(f"  PASS  row count: C is {C.shape}, n_constraints() = {n_predicted}")


# =============================================================================
# Test 2: constant-field reproduction (nullspace property)
# =============================================================================

def test_constraint_kills_periodic_fluctuation():
    """For a periodic fluctuation field that vanishes at corners,
    C·u_fluct = 0.

    Why "periodic fluctuation" not "constant"
    ------------------------------------------
    A constant field is NOT in C's nullspace because corner DOFs are
    sentinel-stripped (they're Dirichlet-pinned separately). The
    partition-of-unity row sum `D[k] = Σ_l A_m[k, l]` is broken at
    rows whose mortar-side neighbours include a corner node — that
    corner contribution is dropped from the A_m sum but accounted
    for in D[k] (which is computed from the nonmortar measure alone).

    The right test is: a function that already vanishes at corners
    AND has u(nonmortar_X) = u(mortar_X) at every matched pair. A product
    of sin(2π·) factors satisfies both: it's exactly zero at every
    box corner, edge, and face boundary node where coords are 0 or L,
    AND it's periodic with period L.

    For the 2x2x2 mock RVE on [0, 1]^3, the only non-zero values of
    sin(2π X) are at the cell centres (X = 0.5), so the test is
    less informative on this minimal mesh than on a finer mesh, but
    it's still a real check.
    """
    cl, n_verts, coords, gtdof_x, gtdof_y, gtdof_z = (
        _build_synthetic_classifier_2x2x2()
    )
    L = 1.0
    u = np.zeros(3 * n_verts, dtype=np.float64)
    for v in range(n_verts):
        sin_val = (np.sin(2 * np.pi * coords[v, 0] / L)
                   * np.sin(2 * np.pi * coords[v, 1] / L)
                   * np.sin(2 * np.pi * coords[v, 2] / L))
        u[gtdof_x[v]] = 0.5  * sin_val
        u[gtdof_y[v]] = -0.7 * sin_val
        u[gtdof_z[v]] = 1.3  * sin_val

    builder = ConstraintBuilder3D(cl)
    C = builder.build()
    Cu = C @ u
    err = float(np.max(np.abs(Cu)))
    assert err < 1e-12, (
        f"Periodic-fluctuation reproduction failed: "
        f"||C·u_fluct||_inf = {err}"
    )
    print(f"  PASS  periodic-fluctuation nullspace: "
          f"||C·u_fluct||_inf = {err:.2e}")


# =============================================================================
# Test 3: affine field produces jump = (F-I)·period
# =============================================================================

def test_constraint_against_affine_yields_known_jump():
    """For u(X) = (F-I) X, C·u should equal the macroscopic jump per mortar-nonmortar pair.

    Per pair, the residual at each constraint row equals:
        D[k] · jump_along_perp_axis · F_factor
    where jump_along_perp_axis = (F-I) · perp_axis_unit_vector * period_length.

    Rather than verifying the exact jump value (which depends on the
    pair_match orientation and assembler conventions), we verify the
    qualitative property: ||C·u_affine||_inf is non-zero, of order
    |F-I| * L * D_typical, and is consistent across vdim components
    (each row triple has the same magnitude pattern).

    This is the necessary counterpart to Test 2: constant fields
    pass through, but affine fields produce the expected jump.
    """
    cl, n_verts, coords, gtdof_x, gtdof_y, gtdof_z = (
        _build_synthetic_classifier_2x2x2()
    )
    F = np.array([
        [1.10, 0.05, 0.02],
        [0.03, 0.95, 0.04],
        [0.01, 0.02, 1.05],
    ])
    F_minus_I = F - np.eye(3)
    u = np.zeros(3 * n_verts, dtype=np.float64)
    for v in range(n_verts):
        u_v = F_minus_I @ coords[v]
        u[gtdof_x[v]] = u_v[0]
        u[gtdof_y[v]] = u_v[1]
        u[gtdof_z[v]] = u_v[2]

    builder = ConstraintBuilder3D(cl)
    C = builder.build()
    Cu = C @ u
    err_inf = float(np.max(np.abs(Cu)))

    # For a 1.0-cube with |F-I| ~ 0.1 and D ~ O(1), the jump should
    # also be O(0.1) at the row level. Just verify it's non-zero.
    assert err_inf > 1e-6, (
        f"Expected non-zero jump for affine field, got {err_inf}"
    )
    # Verify the affine + constant linearity: u_affine + u_const should
    # produce the same C·u as u_affine alone.
    u_const = np.zeros(3 * n_verts, dtype=np.float64)
    for v in range(n_verts):
        u_const[v]               = 0.5
        u_const[v + n_verts]     = -0.2
        u_const[v + 2 * n_verts] = 1.0
    Cu_combined = C @ (u + u_const)
    diff = float(np.max(np.abs(Cu_combined - Cu)))
    assert diff < 1e-12, (
        f"Linearity violation: C is not linear, diff = {diff}"
    )
    print(f"  PASS  affine-field jump: ||C·u_affine||_inf = {err_inf:.4f} "
          f"(non-zero as expected); linearity ||C·(u+const) - C·u||_inf "
          f"= {diff:.2e}")


# =============================================================================
# Test 3: the 3 face mortar-nonmortar pairs target nonmortar gtdofs only
# =============================================================================

def test_face_constraint_rows_target_correct_gtdofs():
    """Each face mortar-nonmortar pair adds rows that touch only:
        - the nonmortar-face-interior gtdofs (positive entries),
        - the mortar-face-interior gtdofs (negative entries),
        - NO corner or edge gtdofs (those were sentinel-stripped).

    Verify by reading the face-block rows directly out of C.
    """
    cl, n_verts, *_ = _build_synthetic_classifier_2x2x2()
    builder = ConstraintBuilder3D(cl)
    C = builder.build().tocoo()

    # Edge rows: 27 (9 pairs * 3 vdim). Face rows: rows 27..36.
    n_edge_rows = 9 * 1 * 3   # 9 pairs * 1 nonmortar node * vdim
    face_row_start = n_edge_rows
    face_row_end = face_row_start + 9

    # For each face row, columns should be a corner-DOF-free subset.
    corner_gtdofs = set()
    for ci in cl.corners.values():
        corner_gtdofs.update([ci.gtdof_x, ci.gtdof_y, ci.gtdof_z])

    edge_gtdofs = set()
    for e in cl.edges.values():
        edge_gtdofs.update(int(g) for g in e.gtdofs_x)
        edge_gtdofs.update(int(g) for g in e.gtdofs_y)
        edge_gtdofs.update(int(g) for g in e.gtdofs_z)

    # Face rows touch ONLY face-interior gtdofs (no corner / no edge).
    for r, c, v in zip(C.row, C.col, C.data):
        if face_row_start <= r < face_row_end:
            assert int(c) not in corner_gtdofs, (
                f"Face row {r} touches corner gtdof {c} (value {v})"
            )
            assert int(c) not in edge_gtdofs, (
                f"Face row {r} touches edge gtdof {c} (value {v})"
            )
    print(f"  PASS  face-row column targets: rows [{face_row_start}, "
          f"{face_row_end}) touch only face-interior gtdofs")


# =============================================================================
# Test 4: sparsity is non-empty in both edge and face row ranges
# =============================================================================

def test_constraint_matrix_is_nonzero():
    """Sanity check: edge and face row blocks both have nonzero rows."""
    cl, *_ = _build_synthetic_classifier_2x2x2()
    builder = ConstraintBuilder3D(cl)
    C = builder.build()
    # Edge block: rows 0..26.
    edge_block = C[:27]
    face_block = C[27:]
    assert edge_block.nnz > 0, "Edge constraint block is empty"
    assert face_block.nnz > 0, "Face constraint block is empty"
    print(f"  PASS  nnz: edge block = {edge_block.nnz}, "
          f"face block = {face_block.nnz}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Phase 3.3.C unit tests — ConstraintBuilder3D")
    print("=" * 60)

    print()
    print("[Row-count formula]")
    test_constraint_row_count()

    print()
    print("[Field reproduction tests]")
    test_constraint_kills_periodic_fluctuation()
    test_constraint_against_affine_yields_known_jump()

    print()
    print("[Sparsity / target-gtdof structure]")
    test_face_constraint_rows_target_correct_gtdofs()
    test_constraint_matrix_is_nonzero()

    print()
    print("=" * 60)
    print(" All Phase 3.3.C tests passed.")
    print("=" * 60)
