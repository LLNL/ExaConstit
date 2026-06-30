"""Linear-elastic + Dirichlet utilities for the 3D mortar PBC prototype.

WHAT
----
Phase 3.1 building blocks for 3D RVEs:

    * ``assemble_linear_elastic_K_hypre(pmesh, fes, E, nu)``
        Assembles the small-strain linear-elastic stiffness K via
        ``ElasticityIntegrator`` and returns the distributed
        ``HypreParMatrix``. Dimension-generic; works in 2D or 3D
        unchanged because the integrator and ParBilinearForm pick up
        the dimension from ``fes``.

    * ``apply_linear_part(fes, F_macro)``
        Project u_lin(X) = (F_macro - I) X onto ``fes`` and return the
        local-rank true-DOF numpy array. Generalised from the 2D
        version (which hard-coded vdim=2 and a 2-vector EvalValue)
        to handle any dimension.

    * ``find_corners_3d(pmesh, fes, tol_rel)``
        Identify the 8 corners of a 3D box RVE by their reference-frame
        coordinates and return ``CornerInfo3D`` records gathered
        across MPI ranks. The 3D analog of the corner-discovery part
        of ``BoundaryClassifier2D``.

    * ``apply_dirichlet_to_distributed_K(K_hyp, f_par, ess_global_tdofs, fes)``
        Eliminate corner-DOF rows/cols on the distributed K and zero
        the corresponding entries of f. Dimension-generic; lifted
        verbatim from the 2D example script (where it has been
        battle-tested at np = 1, 2, 4, 8) but exposed as a package-level
        function so 3D drivers can use it without copy-pasting.

WHY
---
Phase 3.1 is "3D mesh + linear-elastic patch test, NO mortar". It
exercises the 3D mesh handling, FES, Dirichlet, ParaView output, and
``compute_volume_averaged_F`` consistency check on hex AND tet meshes.
This module gives the 3D driver script everything it needs aside from
the mortar machinery (which Phase 3.1 doesn't touch).

DESIGN NOTES
------------
* These functions are intentionally dimension-generic where possible.
  The ``apply_linear_part`` helper takes ``F_macro`` and works for
  ``F_macro.shape == (2, 2)`` or ``(3, 3)`` — same code path. The
  ``assemble_linear_elastic_K_hypre`` helper has been tested in 2D
  against ``ElasticityIntegrator`` and works in 3D unchanged because
  the integrator infers dimension from the FES.

* ``apply_dirichlet_to_distributed_K`` was originally in
  ``examples/patch_test_2d.py`` (and its multi-step heterogeneous
  cousins). Moving it into the package was a deferred refactor; Phase
  3.1 forces our hand because we need it for the 3D driver too.
  The 2D drivers can either keep their local copy (no breakage) or
  switch to the package version in a follow-up clean-up.

REFERENCES
----------
* MORTAR_PBC_ARCHITECTURE.md §11.8 (Phase 3.1 description).
* ``examples/patch_test_2d.py`` for the 2D versions of these helpers
  that this module generalises.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from mpi4py import MPI

import mfem.par as mfem

from .types_3d import CornerInfo3D


# =============================================================================
# Linear-elastic K assembly (dimension-generic)
# =============================================================================

def assemble_linear_elastic_K_hypre(
    pmesh: mfem.ParMesh,
    fes: mfem.ParFiniteElementSpace,
    E: float = 70.0e3,
    nu: float = 0.3,
) -> mfem.HypreParMatrix:
    """Assemble the small-strain linear-elastic tangent K as a HypreParMatrix.

    Identical to the 2D version in patch_test_2d.py, but works in 3D
    unchanged because ``ElasticityIntegrator`` and ``ParBilinearForm``
    both infer the spatial dimension from the FES.

    Parameters
    ----------
    pmesh : mfem.ParMesh
        Parallel mesh (2D or 3D).
    fes : mfem.ParFiniteElementSpace
        Vector H1 space with vdim = pmesh.Dimension().
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    K_hyp : mfem.HypreParMatrix
        Distributed stiffness matrix, ready to be eliminated with
        ``apply_dirichlet_to_distributed_K`` and consumed by the
        saddle-point solver via ``Mult``.

    Notes
    -----
    For heterogeneous RVEs, replace ``ConstantCoefficient`` with
    ``PWConstCoefficient`` and pass per-element-attribute Lamé
    parameters. The 2D heterogeneous patch tests demonstrate the
    pattern; the 3D version follows the same recipe with the
    integrator unchanged.
    """
    mu = 0.5 * E / (1.0 + nu)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lam_coef = mfem.ConstantCoefficient(lam)
    mu_coef = mfem.ConstantCoefficient(mu)

    a = mfem.ParBilinearForm(fes)
    a.AddDomainIntegrator(mfem.ElasticityIntegrator(lam_coef, mu_coef))
    a.Assemble()
    a.Finalize()
    K_hyp = a.ParallelAssemble()
    # Note: ``ParallelAssemble`` returns a freshly-allocated HypreParMatrix
    # that copies the data into HYPRE arrays, so returning it after ``a``
    # goes out of scope is safe in current MFEM (>= 4.0).
    # Cf. mfem/mfem#793 for the underlying lifetime concern.
    return K_hyp


# =============================================================================
# u_lin = (F - I) X projection (dimension-generic)
# =============================================================================

def apply_linear_part(
    fes: mfem.ParFiniteElementSpace,
    F_macro: np.ndarray,
) -> np.ndarray:
    """Compute u_lin(X) = (F - I) X at every nodal coordinate.

    Returns the result as a *local-rank* true-DOF numpy array (the
    portion of TDOFs owned by this rank).

    Parameters
    ----------
    fes : mfem.ParFiniteElementSpace
        Vector H1 space; vdim must equal F_macro.shape[0].
    F_macro : (d, d) ndarray
        Macroscopic deformation gradient. ``d`` is 2 or 3.

    Returns
    -------
    u_lin_local : (n_local_tdofs,) float64 ndarray
        Local-rank true-DOF vector containing the projected u_lin.

    Notes
    -----
    This is the dimension-generic generalisation of the 2D version in
    ``patch_test_2d.py``. The 2D version subclassed
    ``VectorPyCoefficient`` with vdim=2 and a hardcoded 2-vector
    ``EvalValue``; here we close over ``vdim`` and ``F_minus_I`` so the
    same code path handles 2D and 3D.

    The pyMFEM ``VectorPyCoefficient`` idiom requires subclassing (not
    constructor injection of a callable). We therefore define a small
    local subclass with the closed-over data on ``self``.
    """
    vdim = fes.GetVDim()
    if F_macro.shape != (vdim, vdim):
        raise ValueError(
            f"F_macro must be ({vdim}, {vdim}); got {F_macro.shape}"
        )
    F_minus_I = (F_macro - np.eye(vdim)).astype(np.float64)

    class LinearPartCoefficient(mfem.VectorPyCoefficient):
        """u_lin(X) = (F - I) X at point X (vdim-generic)."""

        def __init__(self, A_mat: np.ndarray):
            super().__init__(int(A_mat.shape[0]))
            self.A = A_mat

        def EvalValue(self, x):
            # Return the d-vector (F-I) X at this Gauss / nodal point.
            # ``x`` is a sequence-like of length ``vdim``; we return a
            # plain Python list to be agnostic to pyMFEM build details.
            return [
                float(sum(self.A[i, j] * x[j] for j in range(self.A.shape[1])))
                for i in range(self.A.shape[0])
            ]

    coef = LinearPartCoefficient(F_minus_I)
    gf = mfem.ParGridFunction(fes)
    gf.ProjectCoefficient(coef)

    tv = mfem.Vector()
    gf.GetTrueDofs(tv)
    return np.array(tv.GetDataArray(), dtype=np.float64).copy()


# =============================================================================
# Corner identification for 3D box RVEs
# =============================================================================

# 8 corner labels per the convention documented in CornerInfo3D:
#   first letter:  b/t -> y_min/y_max
#   second letter: l/r -> x_min/x_max
#   third letter:  f/b -> z_min/z_max
_CORNER_LABELS_3D: Tuple[str, ...] = (
    "blf", "brf", "tlf", "trf",
    "blb", "brb", "tlb", "trb",
)


def _corner_target_coord(label: str, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Map a corner label to its target reference-frame coordinate."""
    y_letter, x_letter, z_letter = label[0], label[1], label[2]
    return np.array([
        bbox_max[0] if x_letter == "r" else bbox_min[0],
        bbox_max[1] if y_letter == "t" else bbox_min[1],
        bbox_max[2] if z_letter == "b" else bbox_min[2],
    ], dtype=np.float64)


def _get_my_first_tdof(fes: mfem.ParFiniteElementSpace, rank: int) -> int:
    """Return this rank's first global true-DOF index, robustly across
    pyMFEM exposure variations.

    See ``examples/patch_test_2d.py::_get_my_first_tdof`` for the full
    rationale on why this isn't trivially ``GetTrueDofOffsets()[0]``.
    """
    if hasattr(fes, "GetMyTDofOffset"):
        return int(fes.GetMyTDofOffset())
    offs = fes.GetTrueDofOffsets()
    arr = np.asarray(offs, dtype=np.int64)
    if arr.ndim == 0:
        return int(arr)
    if arr.size == 2:
        return int(arr[0])
    return int(arr[rank])


def find_corners_3d(
    pmesh: mfem.ParMesh,
    fes: mfem.ParFiniteElementSpace,
    tol_rel: float = 1e-9,
) -> Dict[str, CornerInfo3D]:
    """Identify the 8 corners of a 3D box RVE and return them as a dict
    keyed by label.

    Parameters
    ----------
    pmesh : mfem.ParMesh
        Parallel mesh; must be 3D.
    fes : mfem.ParFiniteElementSpace
        Vector H1 space with vdim = 3, ordering byNODES (the prototype
        convention; byVDIM would also work but requires the visualiser
        defensive check).
    tol_rel : float, default 1e-9
        Relative tolerance (vs. bounding-box diagonal) for matching
        a vertex coordinate to a corner location.

    Returns
    -------
    corners : dict[str, CornerInfo3D]
        8 entries keyed by label ("blf", "brf", ..., "trb"); each
        CornerInfo3D has the corner's coord and global TDOF indices
        for x, y, z displacement components.

    Notes
    -----
    Algorithm (mirrors ``BoundaryClassifier2D._build_corners_and_edges``):

        1. Allreduce the local bbox to get the global bbox.
        2. Each rank walks its local boundary vertices; if a vertex
           coordinate matches one of the 8 corner targets within ``tol``
           and the rank owns the vertex's TDOFs, record the global
           TDOFs.
        3. AllGather the (label -> (gtdof_x, gtdof_y, gtdof_z)) records
           and merge: each corner is owned by exactly one rank, so the
           merge is just "take the first non-(-1, -1, -1) record".

    This function is the 3D analog of the corner-discovery part of
    ``BoundaryClassifier2D``. We don't subclass the existing classifier
    because Phase 3.1 doesn't need edges or faces, and we want the 3.1
    deliverable to be locally testable without the full 3D classifier.
    """
    if pmesh.Dimension() != 3:
        raise ValueError(
            f"find_corners_3d requires a 3D mesh; got dim {pmesh.Dimension()}"
        )
    if fes.GetVDim() != 3:
        raise ValueError(
            f"find_corners_3d requires vdim=3 FES; got {fes.GetVDim()}"
        )

    comm: MPI.Intracomm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ----- Step 1: global bbox -----
    local_min = np.full(3, np.inf, dtype=np.float64)
    local_max = np.full(3, -np.inf, dtype=np.float64)
    for v in range(pmesh.GetNV()):
        xyz = np.array([pmesh.GetVertexArray(v)[d] for d in range(3)], dtype=np.float64)
        local_min = np.minimum(local_min, xyz)
        local_max = np.maximum(local_max, xyz)
    bbox_min = np.zeros(3, dtype=np.float64)
    bbox_max = np.zeros(3, dtype=np.float64)
    comm.Allreduce(local_min, bbox_min, op=MPI.MIN)
    comm.Allreduce(local_max, bbox_max, op=MPI.MAX)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    tol = tol_rel * bbox_diag

    # ----- Step 2: walk vertices, match against corner targets -----
    targets: Dict[str, np.ndarray] = {
        label: _corner_target_coord(label, bbox_min, bbox_max)
        for label in _CORNER_LABELS_3D
    }

    my_first_tdof = _get_my_first_tdof(fes, rank)
    my_n_tdof = fes.GetTrueVSize()

    # local_records: label -> (gtdof_x, gtdof_y, gtdof_z) | absent
    local_records: Dict[str, Tuple[int, int, int]] = {}

    # Build a vertex-to-TDOF lookup. For an H1 vector FES with linear
    # elements, GetVertexDofs(v) returns the SCALAR vertex DOF indices.
    # For a vector FES the scalar->vector mapping depends on the
    # ordering: byNODES means component c at scalar DOF s lives at
    # (c * n_scalar_tdofs + s); byVDIM means at (s * vdim + c).
    # We use ``DofToVDof`` for byNODES/byVDIM-agnostic conversion.
    for v in range(pmesh.GetNV()):
        xyz = np.array(
            [pmesh.GetVertexArray(v)[d] for d in range(3)], dtype=np.float64
        )
        # Try to match this vertex to a corner target.
        matched_label = None
        for label, target in targets.items():
            if np.linalg.norm(xyz - target) < tol:
                matched_label = label
                break
        if matched_label is None:
            continue

        # Found a corner vertex on this rank. Resolve its component
        # TDOFs. Per pyMFEM, ``GetVertexDofs(v)`` on a vector FES returns
        # the scalar DOFs; we use ``DofToVDof`` to map (scalar_dof,
        # component) to the correct LDOF for the FES's ordering.
        scalar_ldofs = [int(d) for d in fes.GetVertexDofs(v)]
        if not scalar_ldofs:
            continue  # nothing owned for this vertex on this rank
        s_ldof = scalar_ldofs[0]  # P1: one scalar DOF per vertex

        # Map scalar LDOF -> per-component LDOF -> global TDOF.
        gtdofs = [-1, -1, -1]
        for comp in range(3):
            try:
                comp_ldof = fes.DofToVDof(s_ldof, comp)
            except Exception:
                # Fallback: byNODES math (matches our prototype convention).
                # This shouldn't be needed in modern pyMFEM but kept defensive.
                n_scalar_tdofs = fes.GetNDofs()
                comp_ldof = comp * n_scalar_tdofs + s_ldof

            # LDOF -> TDOF (handles nonmortar DOFs and sign).
            t = fes.GetLocalTDofNumber(comp_ldof)
            if t < 0:
                continue  # not owned on this rank
            gtdofs[comp] = my_first_tdof + int(t)

        # Only record if this rank actually owns at least one component.
        if any(g >= 0 for g in gtdofs):
            local_records[matched_label] = tuple(gtdofs)  # type: ignore[assignment]

    # ----- Step 3: AllGather and merge across ranks -----
    all_records = comm.allgather(local_records)

    corners: Dict[str, CornerInfo3D] = {}
    for label in _CORNER_LABELS_3D:
        merged_gtdofs = [-1, -1, -1]
        for rec in all_records:
            if label in rec:
                comp_gtdofs = rec[label]
                for c in range(3):
                    if comp_gtdofs[c] >= 0 and merged_gtdofs[c] < 0:
                        merged_gtdofs[c] = comp_gtdofs[c]
        if any(g < 0 for g in merged_gtdofs):
            raise RuntimeError(
                f"Corner '{label}' at {targets[label]} has missing TDOFs after "
                f"AllGather merge: {merged_gtdofs}. This likely means the "
                f"mesh doesn't have a vertex at this corner (non-axis-aligned "
                f"box?), or the tol_rel is too tight."
            )
        corners[label] = CornerInfo3D(
            label=label,
            coord=targets[label].copy(),
            gtdof_x=merged_gtdofs[0],
            gtdof_y=merged_gtdofs[1],
            gtdof_z=merged_gtdofs[2],
        )

    return corners


# =============================================================================
# Dirichlet handling on the distributed K (dimension-generic)
# =============================================================================

def apply_dirichlet_to_distributed_K(
    K_hyp: mfem.HypreParMatrix,
    f_par: mfem.Vector,
    ess_global_tdofs: Sequence[int],
    fes: mfem.ParFiniteElementSpace,
    *,
    f_at_essential: Sequence[float] | None = None,
) -> None:
    """Eliminate essential-DOF rows/cols on the distributed K and set
    the corresponding entries of f to the prescribed essential values.
    Modifies both ``K_hyp`` and ``f_par`` in place.

    Dimension-generic: identical algorithm in 2D and 3D.

    Parameters
    ----------
    K_hyp : mfem.HypreParMatrix
        Distributed stiffness; modified in place
        (``EliminateRowsCols``).
    f_par : mfem.Vector
        Distributed RHS; modified in place. Essential entries set to
        ``f_at_essential`` (or 0 if not provided).
    ess_global_tdofs : sequence of int
        Global TDOF indices of essential DOFs (e.g. all 24 corner TDOFs
        in 3D = 8 corners × 3 components).
    fes : mfem.ParFiniteElementSpace
        FE space, used to figure out this rank's TDOF range.
    f_at_essential : sequence of float, optional
        Prescribed values at the essential TDOFs, in the SAME ORDER as
        ``ess_global_tdofs``. If None (default), essential entries are
        zeroed (homogeneous Dirichlet, e.g. for the Phase 1 patch test
        with u_tilde = 0 at corners).

    Notes
    -----
    For Method-D PBC the Dirichlet values are u_lin[corner] = (F - I) X,
    NOT zero. The caller computes these via ``apply_linear_part`` and
    extracts the corner entries; this helper then writes them into the
    distributed RHS at the right TDOF positions.

    Crucial gotcha (documented in §6.4 of MORTAR_PBC_ARCHITECTURE.md):
    ``EliminateRowsCols`` zeros the *full* corner row of K, including
    the off-diagonal coupling K_uc into free DOFs. To preserve the
    consistency of the RHS for non-zero Dirichlet, the caller must
    add ``K_uc @ u_corner`` to f BEFORE calling this function. The
    pattern in the patch test is:

        b_lhs = K_full.Mult(u_lin)         # action on u_corner-extended u
        f -= b_lhs                          # subtract: f -> f - K_uc u_c
        # K_uc set to 0 by EliminateRowsCols below
        apply_dirichlet_to_distributed_K(K, f, ess_tdofs, fes,
                                         f_at_essential=u_corner_values)
        # f at corners is now u_corner_values; identity rows of K
        # produce u = u_corner_values at convergence.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    my_first_tdof = _get_my_first_tdof(fes, rank)
    my_n_tdof = fes.GetTrueVSize()

    local_indices: list[int] = []
    local_vals: list[float] = []
    for i, gd in enumerate(ess_global_tdofs):
        gd_int = int(gd)
        if my_first_tdof <= gd_int < my_first_tdof + my_n_tdof:
            local_indices.append(gd_int - my_first_tdof)
            local_vals.append(
                float(f_at_essential[i]) if f_at_essential is not None else 0.0
            )

    ess_tdof_arr = mfem.intArray(local_indices)
    K_hyp.EliminateRowsCols(ess_tdof_arr)

    f_np = np.asarray(f_par.GetDataArray(), dtype=np.float64, copy=False)
    for local_idx, val in zip(local_indices, local_vals):
        f_np[local_idx] = val


# =============================================================================
# Convenience: build the Newton-step residual at u_init = u_lin
# =============================================================================

def newton_residual_at_u_lin(
    K_hyp: mfem.HypreParMatrix,
    u_lin_local: np.ndarray,
) -> mfem.Vector:
    """Compute the equilibrium residual r1 = K · u_lin at the warm-start
    initial iterate u_init = u_lin, before any Dirichlet elimination.

    Parameters
    ----------
    K_hyp : mfem.HypreParMatrix
        Distributed stiffness (NOT yet eliminated).
    u_lin_local : (n_local_tdofs,) ndarray
        u_lin = (F-I) X, projected onto the FE space and held as a
        local-rank true-DOF numpy array.

    Returns
    -------
    r1_par : mfem.Vector
        Distributed residual r1 = K · u_lin.

    Notes
    -----
    Mirrors the 2D pattern in ``examples/patch_test_2d.py``:

        u_lin_par = numpy_to_mfem_vector(u_lin_local)
        f_par = mfem.Vector(fes.GetTrueVSize())
        K_hyp.Mult(u_lin_par, f_par)
        # Then apply_dirichlet_to_distributed_K to zero corner entries.

    Why "residual" naming: in the Newton-step interpretation of the
    Method-D linear solve (§7.4 of MORTAR_PBC_ARCHITECTURE.md), we
    start at u_init = u_lin, compute r1 = F_int(u_init) - f_ext = K ·
    u_init - 0 = K · u_lin, eliminate Dirichlet, then solve K · du =
    -r1 with du_corner = 0, and update u = u_init + du. For a
    homogeneous patch test, K · u_lin = 0 in the interior (the
    linear-elastic operator on an affine field is zero), so r1 = 0
    after Dirichlet elimination, du = 0, and u = u_lin exactly.

    For heterogeneous RVEs, r1 ≠ 0 in the interior because the
    spatially-varying stiffness produces non-zero stress under uniform
    F; mortar PBC fixes the result by adding the constraint coupling.
    """
    u_lin_par = mfem.Vector(u_lin_local.tolist())
    r1_par = mfem.Vector(u_lin_par.Size())
    K_hyp.Mult(u_lin_par, r1_par)
    return r1_par


def collect_corner_tdofs(corners: Dict[str, CornerInfo3D]) -> list[int]:
    """Flatten the 8 corners into a list of 24 essential global TDOFs."""
    out: list[int] = []
    for label in _CORNER_LABELS_3D:
        c = corners[label]
        out.extend([int(c.gtdof_x), int(c.gtdof_y), int(c.gtdof_z)])
    return out


def find_all_boundary_tdofs(
    pmesh: mfem.ParMesh,
    fes: mfem.ParFiniteElementSpace,
) -> list[int]:
    """Return the GLOBAL TDOFs of every boundary node, all spatial components.

    Used by the Phase 3.1 patch test (homogeneous full-Dirichlet
    validation): the affine field u_lin = (F-I)X is the unique
    minimum-energy solution iff Dirichlet is imposed on the ENTIRE
    boundary. Pinning only the 8 corners leaves the rest of ∂Ω with
    natural (zero-traction) Neumann, which is incompatible with the
    constant stress σ = C : sym(F-I); the solver then finds a non-affine
    field that satisfies σ·n = 0 on the free boundary.

    Implementation
    --------------
    1. Build `ess_bdr` array marking ALL boundary attributes essential.
    2. `fes.GetEssentialTrueDofs(ess_bdr, list)` returns local TDOFs on
       this rank that lie on the boundary, with all vector components
       included automatically (vdim-aware).
    3. Convert local TDOFs to global by adding this rank's `_get_my_first_tdof`
       offset.

    The returned list contains GLOBAL TDOF indices owned by this rank
    only. After AllGather across ranks, the union is the full essential
    set; for `apply_dirichlet_to_distributed_K`, each rank passes its
    local-owned subset (the helper filters by rank-ownership anyway,
    so passing AllGather'd globals also works).

    Parameters
    ----------
    pmesh : mfem.ParMesh
    fes : mfem.ParFiniteElementSpace
        Vector H1 space; vdim sets how many components per boundary node.

    Returns
    -------
    list[int]
        Global TDOFs (this rank's owned subset). Each value is in
        ``[my_first_tdof, my_first_tdof + my_n_tdof)``.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Mark all boundary attributes essential. ParMesh.bdr_attributes is
    # an mfem.intArray; we read its size, build a same-size mask, all 1s.
    n_bdr_attrs = int(pmesh.bdr_attributes.Max())
    ess_bdr = mfem.intArray(n_bdr_attrs)
    ess_bdr.Assign(1)

    # GetEssentialTrueDofs fills `ess_tdof_list` with local TDOFs on this
    # rank lying on the marked boundary, including every vector component.
    ess_tdof_list = mfem.intArray()
    fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    # Convert to global. Use the same offset helper as elsewhere in this
    # module so behaviour is consistent across drivers.
    offset = _get_my_first_tdof(fes, rank)
    local_tdofs = ess_tdof_list.ToList()  # numpy/python list view
    return [int(t) + offset for t in local_tdofs]


def collect_boundary_tdof_values(
    boundary_global_tdofs: Sequence[int],
    u_lin_local: np.ndarray,
    fes: mfem.ParFiniteElementSpace,
) -> list[float]:
    """For each global TDOF in ``boundary_global_tdofs``, return its
    u_lin value from this rank's local TDOF array.

    Used to build the ``f_at_essential`` argument for
    ``apply_dirichlet_to_distributed_K`` when the Dirichlet values are
    u_lin = (F-I)X (Phase 3.1 full-boundary case) or u_lin[corner]
    (Method-D PBC case at the 8 corners).

    Returns a list aligned with ``boundary_global_tdofs``; entries for
    TDOFs not owned by this rank are zero (the helper filters on its
    own anyway).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    my_first = _get_my_first_tdof(fes, rank)
    my_n = fes.GetTrueVSize()

    vals: list[float] = []
    for gd in boundary_global_tdofs:
        gd_int = int(gd)
        if my_first <= gd_int < my_first + my_n:
            vals.append(float(u_lin_local[gd_int - my_first]))
        else:
            vals.append(0.0)
    return vals
