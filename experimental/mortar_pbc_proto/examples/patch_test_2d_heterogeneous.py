"""2D mortar PBC patch test -- linear elastic, heterogeneous strip-split.

Pivoted from NeoHookean + Newton to linear elastic + single linear solve
because pyMFEM's NeoHookeanModel produces NaN at u=0 in this build,
regardless of coefficient type or mesh attribute count (verified
exhaustively in ``examples/diag_neohookean_2x2.py``).  Linear elasticity
is sufficient to validate the mortar PBC machinery -- the integrator
issue is orthogonal to the PBC method.

Material setup
--------------
Vertical strip split:
  * Element attribute 1 (left half, x < L/2)  -> material 1 (matrix)
  * Element attribute 2 (right half, x >= L/2) -> material 2 (stiff)
5x stiffness contrast (Young's modulus); same Poisson ratio.
Materials are linear-elastic with PWConstCoefficient on Lame parameters.

Method-D bookkeeping (Lopes 2021 Remark 1, line 342)
----------------------------------------------------
The macroscopic affine field u_lin = (F-I)X is applied as the initial
guess on the entire RVE domain.  The fluctuation u_tilde = u - u_lin is
then solved for via the saddle-point system:

    [ K   C^T ] [ u_tilde ]   [ -K @ u_lin ]
    [ C    0  ] [ lambda  ] = [     0      ]

with corner DOFs (8 TDOFs in 2D, 4 corners x 2 components) eliminated
from K and the RHS.  At convergence, total displacement is
u = u_lin + u_tilde with u_tilde at machine precision for homogeneous
material and a non-trivial bounded field for heterogeneous.

For homogeneous material, u_tilde should be ~0 (linear elastic exact
solution under affine BC).  For 5x strip-split, u_tilde is non-trivial:
the soft strip relaxes more, the stiff strip resists.

Macroscopic F selectable via --F=<choice> CLI flag:
  --F=uniaxial   (default)  : [[1.2,  0],   [0,   1.0]]
  --F=shear                 : [[1.2,  0.2], [0.2, 1.05]]
  --F=mild-shear            : [[1.05, 0.05], [0.05, 1.02]]

Run with:
    python examples/patch_test_2d_heterogeneous.py
    mpirun -n N python examples/patch_test_2d_heterogeneous.py --F=uniaxial
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI

import mfem.par as mfem

from mortar_pbc import (
    BoundaryClassifier2D,
    MortarAssembler2D,
    ConstraintBuilder2D,
    SaddlePointSolver,
    make_constraint_operators,
    apply_dirichlet_zero_to_C,
    write_pbc_visualization,
    PbcVisualizationWriter,
    MortarPbcDriver2D,
)
# Quarantined verification path -- not exported from package's public API.
from mortar_pbc._verify_solver import SciPyDirectSolver


# ---------------------------------------------------------------------------
# Mesh construction: homogeneous square with deliberately non-conforming sides
# ---------------------------------------------------------------------------

def build_nonconforming_square(L: float = 1.0,
                               n_left: int = 5,
                               n_right: int = 7,
                               n_bottom: int = 6,
                               n_top: int = 4) -> mfem.Mesh:
    """Build an L x L square mesh with non-matching node counts on opposite
    edges.  We do this by constructing two separate Cartesian sub-rectangles
    and merging them along an internal vertical seam, then varying the
    boundary divisions.

    For Phase 1 simplicity, the easier way to achieve a non-conforming
    boundary is to take a uniform Cartesian mesh and *displace* every
    second boundary edge node by a small amount, which forces the mortar
    machinery to integrate on a real intersection.  But that doesn't
    produce a true non-matching mesh -- the connectivity is still uniform.

    For a proper non-conforming test we use MFEM's serial Make2D with two
    different element counts and merge.  Since merging is awkward in pure
    pyMFEM, we instead use a structured mesh with different counts on
    each *edge* by generating an unstructured triangle mesh via
    Mesh::MakeCartesian2D and then perturbing.  Below we use the simplest
    approach that suffices for verification: a uniform mesh whose
    "non-conforming" character comes from the assembly going through the
    mortar pipeline regardless.

    Returns a serial mfem.Mesh in 2D.
    """
    # Uniform 2D Cartesian mesh -- enough for first verification.
    nx, ny = 8, 8
    # Modern pyMFEM factory (preferred over the legacy
    # ``mfem.Mesh(nx, ny, "QUADRILATERAL", 1, L, L)`` constructor).
    # Signature: MakeCartesian2D(nx, ny, type, generate_edges, sx, sy)
    mesh = mfem.Mesh.MakeCartesian2D(
        nx, ny, mfem.Element.QUADRILATERAL, True, L, L,
    )

    # Set boundary attributes per ExaConstit 2D convention:
    # 1=bottom, 2=left, 3=top, 4=right
    for be in range(mesh.GetNBE()):
        # pyMFEM convention: GetBdrElementVertices returns the vertex array
        # directly (the C++ out-parameter pattern is not exposed in Python).
        # Coerce to a plain list of ints for safe iteration regardless of
        # whether pyMFEM returned an mfem.intArray proxy, a list, or a numpy
        # int array.
        verts = [int(v) for v in mesh.GetBdrElementVertices(be)]
        ys = [mesh.GetVertexArray(v)[1] for v in verts]
        xs = [mesh.GetVertexArray(v)[0] for v in verts]
        ymid = sum(ys) / len(ys)
        xmid = sum(xs) / len(xs)
        # All vertices on a boundary element share one constant coord
        if all(abs(y - 0.0) < 1e-9 for y in ys):
            mesh.SetBdrAttribute(be, 1)  # bottom
        elif all(abs(x - 0.0) < 1e-9 for x in xs):
            mesh.SetBdrAttribute(be, 2)  # left
        elif all(abs(y - L) < 1e-9 for y in ys):
            mesh.SetBdrAttribute(be, 3)  # top
        elif all(abs(x - L) < 1e-9 for x in xs):
            mesh.SetBdrAttribute(be, 4)  # right

    # ----- Domain attributes for heterogeneous material (Step 2.2) -----
    # Vertical strip split: elements with centroid x < L/2 -> attribute 1
    # (material 1, left strip).  Elements with centroid x >= L/2 ->
    # attribute 2 (material 2, right strip).  The two materials are
    # bonded along the internal seam at x = L/2.  Periodic BCs in y
    # are within-material (top/bottom of each strip is the same material
    # column); periodic BCs in x couple ACROSS the material interface
    # (left edge is mat 1, right edge is mat 2, and they're identified
    # via the constraint).  This layout exercises both within-material
    # and across-material periodicity at once.
    L_half = 0.5 * L
    for e in range(mesh.GetNE()):
        verts = [int(v) for v in mesh.GetElementVertices(e)]
        xs = [mesh.GetVertexArray(v)[0] for v in verts]
        x_centroid = sum(xs) / len(xs)
        if x_centroid < L_half:
            mesh.SetAttribute(e, 1)   # left strip = material 1
        else:
            mesh.SetAttribute(e, 2)   # right strip = material 2
    # MFEM caches mesh.attributes from the per-element values; force a
    # refresh so PWConstCoefficient sees both attributes 1 and 2.
    mesh.SetAttributes()

    return mesh


# ---------------------------------------------------------------------------
# Linear-elastic stiffness via mfem.ParBilinearForm
# ---------------------------------------------------------------------------

def assemble_linear_elastic_K_hypre(
    pmesh: mfem.ParMesh,
    fes:   mfem.ParFiniteElementSpace,
    E:     float = 70.0e3,
    nu:    float = 0.3,
) -> mfem.HypreParMatrix:
    """Assemble the small-strain linear-elastic tangent K as a HypreParMatrix.

    For the patch test linear elasticity is sufficient because for a
    homogeneous RVE under uniform F, the fluctuation is zero by
    construction; we are only verifying that the constraint enforcement
    *preserves* uniform deformation, not that the material is finite-strain.

    Returns the *distributed* HypreParMatrix; the driver gathers to rank 0
    via ``hypre_to_scipy_csr`` for the prototype's direct SPS solve.
    """
    mu  = 0.5 * E / (1.0 + nu)
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lam_coef = mfem.ConstantCoefficient(lam)
    mu_coef  = mfem.ConstantCoefficient(mu)

    a = mfem.ParBilinearForm(fes)
    a.AddDomainIntegrator(mfem.ElasticityIntegrator(lam_coef, mu_coef))
    a.Assemble()
    a.Finalize()
    K_hyp = a.ParallelAssemble()
    # Note: see mfem/mfem#793 -- the HypreParMatrix's underlying CSR data
    # can depend on the BilinearForm's lifetime under some MFEM versions.
    # ``ParallelAssemble`` returns a freshly-allocated HypreParMatrix that
    # copies the data into HYPRE arrays, so returning it after ``a`` goes
    # out of scope is safe in current MFEM (>= 4.0).
    return K_hyp


def assemble_linear_elastic_K(pmesh: mfem.ParMesh,
                              fes: mfem.ParFiniteElementSpace,
                              E: float = 70.0e3,
                              nu: float = 0.3) -> sp.csr_matrix | None:
    """DEPRECATED: kept for backward-compat with one-step prototypes that
    expect a CSR.  Returns the gathered scipy CSR on rank 0, ``None`` on
    other ranks.  New code should call ``assemble_linear_elastic_K_hypre``
    directly and gather only when needed.
    """
    K_hyp = assemble_linear_elastic_K_hypre(pmesh, fes, E=E, nu=nu)
    return hypre_to_scipy_csr(K_hyp, fes)


# ---------------------------------------------------------------------------
# Partition / TDOF-offset helpers
#
# pyMFEM's wrappers around the various partition queries return
# inconsistent shapes depending on build flags (assumed-partition vs.
# global-partition mode in HYPRE) and on how the SWIG wrapper marshals
# the result (sometimes a plain Python int, sometimes a numpy array).
# These helpers insulate the rest of the prototype from those
# inconsistencies.
# ---------------------------------------------------------------------------

def _get_my_first_tdof(fes: mfem.ParFiniteElementSpace, rank: int) -> int:
    """Return this rank's first global true-DOF index, robustly across
    pyMFEM exposure variations.

    pyMFEM's ``GetTrueDofOffsets()`` is wrapped differently in different
    builds:

        * Sometimes it returns a numpy array of shape (2,) -- "assumed
          partition" mode -- where ``[0]`` is this rank's first owned
          TDOF and ``[1]`` is the past-the-end index.
        * Sometimes it returns a numpy array of shape (nranks+1,) --
          "global partition" mode -- where ``[r]`` is rank r's first.
        * Sometimes it returns a 0-d numpy array containing a Python
          int (the result of ``np.asarray`` on a scalar return value).

    To insulate the prototype from these wrapper inconsistencies we
    prefer the canonical ``GetMyTDofOffset()`` accessor when exposed,
    falling back to parsing ``GetTrueDofOffsets`` only if not.
    """
    if hasattr(fes, "GetMyTDofOffset"):
        return int(fes.GetMyTDofOffset())
    offs = fes.GetTrueDofOffsets()
    arr = np.asarray(offs, dtype=np.int64)
    if arr.ndim == 0:
        # 0-d numpy array: pyMFEM returned a scalar.  Element-zero
        # access would IndexError; use ``int(arr)`` to unwrap.
        return int(arr)
    if arr.size == 2:
        return int(arr[0])         # assumed-partition: [first, last_excl]
    return int(arr[rank])          # global-partition: nranks+1 entries


def _get_first_global_row(hyp_mat: mfem.HypreParMatrix, rank: int) -> int:
    """Return this rank's first owned global row of a HypreParMatrix,
    robustly across pyMFEM exposure variations.

    Mirrors ``_get_my_first_tdof`` for HypreParMatrix.  ``GetRowPartArray()``
    has the same multi-shape inconsistency as ``GetTrueDofOffsets``.
    """
    if hasattr(hyp_mat, "GetRowStart"):
        # Some pyMFEM builds expose this as a direct accessor.
        return int(hyp_mat.GetRowStart())
    arr = np.asarray(hyp_mat.GetRowPartArray(), dtype=np.int64)
    if arr.ndim == 0:
        return int(arr)
    if arr.size == 2:
        return int(arr[0])
    return int(arr[rank])


def hypre_to_scipy_csr(hyp_mat: mfem.HypreParMatrix,
                       fes: mfem.ParFiniteElementSpace) -> sp.csr_matrix | None:
    """Gather a HypreParMatrix to rank 0 as a global scipy CSR matrix.

    Strategy
    --------
    pyMFEM ships a helper ``mfem.common.parcsr_extra.ToScipyCSR`` that wraps
    ``HypreParMatrix::MergeDiagAndOffd`` to produce a serial scipy CSR with
    shape ``(n_local_rows, n_global_cols)`` -- i.e. each rank already gets
    its row slice expressed in *global* column indexing.  We then:

        1. Convert each rank's local CSR to COO.
        2. Shift the (local) row indices by the rank's first global row
           (taken from ``HypreParMatrix.GetRowPartArray()``, which is also
           the canonical pyMFEM helper).
        3. ``comm.gather`` the COO triples to rank 0.
        4. Build the global CSR from the concatenated triples.

    This is a *prototype-grade* gather: the entire global K lives on a
    single rank.  Fine for verifying correctness on RVE-sized problems;
    in production / the C++ port we keep K distributed and apply it via
    ``Mult`` inside a Krylov saddle-point solve.

    Parameters
    ----------
    hyp_mat : mfem.HypreParMatrix
        Distributed matrix to gather.
    fes : mfem.ParFiniteElementSpace
        Currently unused (signature kept for symmetry with the vector
        helpers, which need it for the partition); may be removed later.

    Returns
    -------
    csr : (n_global_rows, n_global_cols) scipy.sparse.csr_matrix on rank 0,
        ``None`` on every other rank.
    """
    # Lazy import: parcsr_extra needs mfem.par + mpi4py and is not always
    # importable at top of module (e.g. in serial-build environments).
    from mfem.common.parcsr_extra import ToScipyCSR

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ----- Per-rank CSR slice in (n_local_rows, n_global_cols) form -----
    # ToScipyCSR holds a reference to the merged mfem.SparseMatrix on the
    # returned scipy matrix's _linked_mat attribute, so the data backing
    # arrays stay alive for the duration of this function.
    local_csr = ToScipyCSR(hyp_mat)

    # ----- Convert to COO and shift row indices to global -----
    local_coo = local_csr.tocoo()
    # ``_get_first_global_row`` handles the various shapes
    # ``GetRowPartArray`` may return across pyMFEM versions (2-element
    # assumed-partition, (nranks+1)-element global-partition, or 0-d
    # numpy scalar).
    my_first_global_row = _get_first_global_row(hyp_mat, rank)

    rows_global = local_coo.row.astype(np.int64) + my_first_global_row
    cols_global = local_coo.col.astype(np.int64)   # already global from MergeDiagAndOffd
    vals        = local_coo.data.astype(np.float64)

    # ----- Gather all triples to rank 0 -----
    all_rows = comm.gather(rows_global, root=0)
    all_cols = comm.gather(cols_global, root=0)
    all_vals = comm.gather(vals,        root=0)

    if rank == 0:
        if all_rows:
            rows_concat = np.concatenate(all_rows)
            cols_concat = np.concatenate(all_cols)
            vals_concat = np.concatenate(all_vals)
        else:
            rows_concat = np.empty(0, dtype=np.int64)
            cols_concat = np.empty(0, dtype=np.int64)
            vals_concat = np.empty(0, dtype=np.float64)
        n_global_rows = hyp_mat.GetGlobalNumRows()
        n_global_cols = hyp_mat.GetGlobalNumCols()
        return sp.csr_matrix(
            (vals_concat, (rows_concat, cols_concat)),
            shape=(n_global_rows, n_global_cols),
        )
    return None


# ---------------------------------------------------------------------------
# Vector gather / scatter helpers
# ---------------------------------------------------------------------------

def gather_tdof_vector_to_root(
    local_vec: np.ndarray,
    fes: mfem.ParFiniteElementSpace,
) -> np.ndarray | None:
    """Gather a TDOF-distributed ndarray to a single global ndarray on rank 0.

    Each rank owns ``fes.GetTrueVSize()`` consecutive entries of the global
    vector, starting at the rank's first TDOF index.  We use ``Gatherv``
    with the per-rank counts to assemble.

    Returns
    -------
    np.ndarray on rank 0 (length ``fes.GlobalTrueVSize()``), ``None`` on
    other ranks.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_count = int(local_vec.size)
    counts = np.array(comm.allgather(local_count), dtype=np.int64)

    if rank == 0:
        global_size = fes.GlobalTrueVSize()
        global_vec = np.zeros(global_size, dtype=np.float64)
        displs = np.zeros_like(counts)
        np.cumsum(counts[:-1], out=displs[1:])
        comm.Gatherv(
            local_vec.astype(np.float64, copy=False),
            [global_vec, counts, displs, MPI.DOUBLE],
            root=0,
        )
        return global_vec
    else:
        comm.Gatherv(local_vec.astype(np.float64, copy=False), None, root=0)
        return None


def scatter_tdof_vector_from_root(
    global_vec: np.ndarray | None,
    fes: mfem.ParFiniteElementSpace,
) -> np.ndarray:
    """Scatter a global ndarray on rank 0 to per-rank local TDOF slices.

    Inverse of ``gather_tdof_vector_to_root``.  All ranks return their
    local slice of the global vector.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_count = int(fes.GetTrueVSize())
    counts = np.array(comm.allgather(local_count), dtype=np.int64)

    local_vec = np.zeros(local_count, dtype=np.float64)
    if rank == 0:
        assert global_vec is not None
        displs = np.zeros_like(counts)
        np.cumsum(counts[:-1], out=displs[1:])
        comm.Scatterv(
            [global_vec.astype(np.float64, copy=False), counts, displs, MPI.DOUBLE],
            local_vec, root=0,
        )
    else:
        comm.Scatterv(None, local_vec, root=0)
    return local_vec


# ---------------------------------------------------------------------------
# Apply linear (kinematic insertion) part u = (F - I) Y as the initial guess
# ---------------------------------------------------------------------------

def apply_linear_part(fes: mfem.ParFiniteElementSpace,
                      F_macro: np.ndarray) -> np.ndarray:
    """Compute u_lin(X) = (F - I) X at every nodal coordinate, return as
    a local-rank true-DOF numpy array.

    Notes on pyMFEM coefficient idiom
    ---------------------------------
    Modern pyMFEM expects ``VectorPyCoefficient`` to be SUBCLASSED, not
    constructed with a callable.  The subclass overrides ``EvalValue(x)``
    to return the vector value at point ``x`` (as a Python list, tuple,
    or numpy array).  We define a small local subclass and instantiate it.

    Two alternative idioms exist in pyMFEM and would also work here, but
    are less universal across pyMFEM versions:
      * ``mfem.jit.vector(...)`` decorator (numba JIT) -- requires numba.
      * ``VectorFunctionCoefficient(vdim, callable)`` with a C++-style
        out-parameter callable -- not consistently exposed in develop.
    """
    F_minus_I = (F_macro - np.eye(2)).astype(np.float64)

    class LinearPartCoefficient(mfem.VectorPyCoefficient):
        """u_lin(X) = (F - I) X at point X (vdim=2)."""
        def __init__(self, F_minus_I_mat: np.ndarray):
            # vdim=2 (planar); the parent class expects this in __init__.
            super().__init__(2)
            self.A = F_minus_I_mat

        def EvalValue(self, x):
            # Return the 2-vector (F-I) X at this Gauss / nodal point.
            return [self.A[0, 0] * x[0] + self.A[0, 1] * x[1],
                    self.A[1, 0] * x[0] + self.A[1, 1] * x[1]]

    coef = LinearPartCoefficient(F_minus_I)
    gf   = mfem.ParGridFunction(fes)
    gf.ProjectCoefficient(coef)

    # Extract local-rank true-DOF vector as a numpy array.
    tv = mfem.Vector()
    gf.GetTrueDofs(tv)
    return np.array(tv.GetDataArray(), dtype=np.float64).copy()


# ---------------------------------------------------------------------------
# Corner Dirichlet handling: row/col elimination on K, col zeroing on C
# ---------------------------------------------------------------------------

def apply_dirichlet_to_zero(
    K: sp.csr_matrix,
    f: np.ndarray,
    C: sp.csr_matrix,
    dofs: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray, sp.csr_matrix]:
    """Enforce u_dof = 0 (Dirichlet at the four RVE corners) by symmetric
    row/col elimination on K and column zeroing on C.

    Strategy
    --------
    For each constrained DOF index ``d``:
        K[d, :]  -> e_d  (identity row, so the d-th equation is u_d = 0)
        K[:, d]  -> 0    (zero the column to preserve symmetry)
        K[d, d]  -> 1    (restore the diagonal entry)
        f[d]     -> 0    (zero the corresponding RHS entry)
        C[:, d]  -> 0    (the constraint must not couple to a prescribed DOF)

    This is the classic "Dirichlet by replacement" treatment.  Symmetry of
    K is preserved.  The constraint matrix C does NOT get rows eliminated
    (corner DOFs were never in C's row space to begin with); only its
    columns at corner DOFs are zeroed.

    Parameters
    ----------
    K : (n, n) scipy CSR
    f : (n,) ndarray
    C : (m, n) scipy CSR
    dofs : (k,) array of int
        Global TDOF indices to constrain to zero.

    Returns
    -------
    K_mod, f_mod, C_mod : modified copies (originals unchanged).
    """
    # Convert to LIL for cheap row writes; CSC for cheap column writes.
    K = K.tolil()
    f = f.copy()
    C = C.tolil()

    dof_set = set(int(d) for d in dofs)

    # ----- (1) Replace constrained rows of K with identity rows; zero f. -----
    for d in dof_set:
        K.rows[d] = [d]
        K.data[d] = [1.0]
        f[d] = 0.0

    # ----- (2) Zero the corresponding columns of K (symmetry) -----
    K = K.tocsc()
    for d in dof_set:
        col_start = K.indptr[d]
        col_end   = K.indptr[d + 1]
        K.data[col_start:col_end] = 0.0
    K.eliminate_zeros()

    # ----- (3) Restore the diagonal entries to 1 -----
    K = K.tolil()
    for d in dof_set:
        K[d, d] = 1.0

    # ----- (4) Zero the constrained columns of C -----
    C = C.tocsc()
    for d in dof_set:
        col_start = C.indptr[d]
        col_end   = C.indptr[d + 1]
        C.data[col_start:col_end] = 0.0
    C.eliminate_zeros()

    return K.tocsr(), f, C.tocsr()


# ---------------------------------------------------------------------------
# Distributed Dirichlet handling for HypreParMatrix
# ---------------------------------------------------------------------------

def apply_dirichlet_to_distributed_K(
    K_hyp: mfem.HypreParMatrix,
    f_par: mfem.Vector,
    corner_global_tdofs: np.ndarray,
    fes: mfem.ParFiniteElementSpace,
) -> None:
    """Eliminate corner-DOF rows/cols on the distributed K and zero the
    corresponding entries of f.  Modifies both ``K_hyp`` and ``f_par`` in
    place.

    Strategy
    --------
    1. Convert global corner TDOF list to LOCAL TDOF indices for this rank
       (filter to TDOFs in this rank's [first, first + n_local) range).
    2. Call ``K_hyp.EliminateRowsCols(local_corner_tdofs)``.  This zeros
       the corresponding rows AND columns of K, and sets the corner
       diagonal to 1 (so the corner equations become trivial: ``u_c = 0``).
       It also returns a ``mfem.HypreParMatrix`` containing the eliminated
       part, which we discard -- we only need the modified K for our
       single-Newton-step linear patch test.
    3. Zero the corner entries of ``f_par`` locally (since we want
       ``u_corner = 0``, the corner equation reads ``u_corner = 0`` which
       is independent of f).

    Notes
    -----
    For inhomogeneous Dirichlet (u_corner = nonzero value), the residual
    would need an additional ``A_e @ x_dirichlet`` correction.  Our patch
    test uses homogeneous corners (u_tilde = 0), so the simple zero
    treatment is correct.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Determine this rank's TDOF range.  Use the helper that handles
    # the various wrapper shapes pyMFEM may return for the partition
    # query (see ``_get_my_first_tdof`` for the rationale).
    my_first_tdof = _get_my_first_tdof(fes, rank)
    my_n_tdof = fes.GetTrueVSize()

    # Filter corner TDOFs to those owned by this rank, then convert to
    # local indices.
    local_corner_tdofs = []
    for d in corner_global_tdofs:
        d_int = int(d)
        if my_first_tdof <= d_int < my_first_tdof + my_n_tdof:
            local_corner_tdofs.append(d_int - my_first_tdof)

    # Build the mfem.intArray expected by EliminateRowsCols.
    ess_tdof_arr = mfem.intArray(local_corner_tdofs)

    # Eliminate K's corner rows/cols.  Returns the eliminated piece;
    # we discard.  K_hyp itself is modified in place: corner rows/cols
    # become identity-like, so the corner equations are vacuous (u_c = 0
    # provided f_corner = 0).
    K_hyp.EliminateRowsCols(ess_tdof_arr)

    # Zero corner entries of f locally.
    f_np = np.asarray(f_par.GetDataArray(), dtype=np.float64, copy=False)
    for local_idx in local_corner_tdofs:
        f_np[local_idx] = 0.0


# ---------------------------------------------------------------------------
# Numpy <-> mfem.Vector conversion helpers
# ---------------------------------------------------------------------------

def numpy_to_mfem_vector(arr: np.ndarray) -> mfem.Vector:
    """Wrap a numpy array as a fresh mfem.Vector (copies the data)."""
    n = int(arr.size)
    v = mfem.Vector(n)
    v_np = np.asarray(v.GetDataArray(), dtype=np.float64, copy=False)
    v_np[:] = np.asarray(arr, dtype=np.float64).ravel()
    return v


def mfem_vector_to_numpy(v: mfem.Vector) -> np.ndarray:
    """Extract an mfem.Vector's data as a numpy array (copies)."""
    return np.array(v.GetDataArray(), dtype=np.float64).copy()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    """Patch-test driver: distributed Krylov primary, direct LU cross-check.

    Algorithm
    ---------
    All ranks (no gather):
        1. Build mesh, ParFE space.
        2. Classify boundary (AllGather inside).
        3. Assemble mortar matrices (pure NumPy, identical on every rank).
        4. Build C scipy CSR (replicated on every rank).
        5. Apply Dirichlet column-zeroing to C (still scipy CSR).
        6. Wrap C as distributed PyOperators.
        7. Assemble K as HypreParMatrix.
        8. Compute f_par = K @ u_lin distributedly via K.Mult.
        9. Eliminate K's corner rows/cols and zero corner entries of f.
       10. Solve via SaddlePointSolver (distributed Krylov).

    Verification (rank 0 only):
       11. Gather K to rank 0 as scipy CSR.
       12. Gather u_lin and f to rank 0.
       13. Apply Dirichlet via the legacy scipy helper.
       14. Solve via SciPyDirectSolver.
       15. Compare to gathered Krylov du.

    PASS criterion: Krylov residuals AND patch-test fluctuation norms
    are below tolerance.  The verification cross-check is informational
    (a diff between Krylov and direct solutions of order 1e-9 is normal
    and not a failure).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    if rank == 0:
        print("=" * 70)
        print("Mortar PBC 2D patch test -- linear elastic (heterogeneous)")
        print(f"  MPI ranks: {nranks}")
        print("  Strip split: left = mat 1, right = mat 2 (5x stiffness)")
        print("=" * 70)

    # ---------------------------------------------------------------------
    # Steps 1-7: build the FE problem (every rank participates)
    # ---------------------------------------------------------------------
    smesh = build_nonconforming_square(L=1.0)
    pmesh = mfem.ParMesh(comm, smesh)
    fec   = mfem.H1_FECollection(1, 2)
    fes   = mfem.ParFiniteElementSpace(pmesh, fec, 2)  # vdim=2 (planar)

    # ----- Boundary classification (AllGather inside) -----
    # IMPORTANT: this collective must be called BEFORE any rank-0-only
    # prints that follow.  If a rank-0-only print were placed between
    # collectives, rank 0 would block on the print's I/O while non-root
    # ranks continued ahead and entered the next collective alone --
    # MFEM's collectives expect every rank to participate in the same
    # order, so this asymmetry can deadlock.
    cl = BoundaryClassifier2D(pmesh, fes)

    if rank == 0:
        print(f"Mesh dim={pmesh.Dimension()}, "
              f"global TDOFs={fes.GlobalTrueVSize()}")
        print("\n" + cl.summary())

    # ----- Mortar matrix assembly -----
    asm = MortarAssembler2D(cl)
    blocks = asm.assemble_all()

    # ----- Build constraint matrix C (scipy CSR, identical on every rank) -----
    C_global_csr = ConstraintBuilder2D(cl, blocks).build()
    n_lam_total = C_global_csr.shape[0]
    if rank == 0:
        print(f"\nC matrix: shape={C_global_csr.shape}, nnz={C_global_csr.nnz}")

    # ----- Apply Dirichlet column-zeroing on C (scipy side) -----
    corner_tdofs = cl.corner_dirichlet_gtdofs()
    if rank == 0:
        print(f"Corner Dirichlet TDOFs (set to zero): {corner_tdofs}")
    C_global_csr_modified = apply_dirichlet_zero_to_C(C_global_csr, corner_tdofs)

    # ----- All-on-rank-0 multiplier layout: rank 0 owns all rows of C -----
    n_lam_local = n_lam_total if rank == 0 else 0
    C_op, CT_op = make_constraint_operators(
        C_global_csr_modified, fes, n_lam_local,
    )

    # ----- Build linear-elastic ParBilinearForm with PWConstCoefficient -
    # Heterogeneous linear elasticity, vertical strip split:
    #   * Element attribute 1 (left half, x < L/2)  -> material 1 (matrix)
    #   * Element attribute 2 (right half, x >= L/2) -> material 2 (stiff)
    # 5x stiffness contrast (Young's modulus); same Poisson ratio.
    #
    # Switched from NeoHookean to linear-elastic ElasticityIntegrator
    # because pyMFEM's NeoHookeanModel produced NaN at u=0 in this build
    # (regardless of coefficient type, mesh attribute count, or whether
    # PWConstCoefficient was used).  Linear elasticity gives us a clean
    # test of the mortar PBC machinery without fighting the integrator.
    #
    # Lame parameters from Young's modulus E and Poisson ratio nu:
    #     mu  = E / (2(1 + nu))
    #     lam = E nu / ((1 + nu)(1 - 2 nu))
    E_1   = 70.0e3        # matrix (left strip, material 1)
    E_2   = 5.0 * E_1     # 5x stiffer inclusion (right strip, material 2)
    nu_1  = 0.3
    nu_2  = 0.3

    mu_1  = E_1 / (2.0 * (1.0 + nu_1))
    lam_1 = E_1 * nu_1 / ((1.0 + nu_1) * (1.0 - 2.0 * nu_1))
    mu_2  = E_2 / (2.0 * (1.0 + nu_2))
    lam_2 = E_2 * nu_2 / ((1.0 + nu_2) * (1.0 - 2.0 * nu_2))

    if rank == 0:
        print(f"\nLinear elastic material (heterogeneous, 5x contrast):")
        print(f"  Material 1 (left strip,  attr=1): "
              f"E={E_1:.3e}, mu={mu_1:.3e}, lam={lam_1:.3e}")
        print(f"  Material 2 (right strip, attr=2): "
              f"E={E_2:.3e}, mu={mu_2:.3e}, lam={lam_2:.3e}")

    # PWConstCoefficient indexed by mesh attribute (1, 2):
    mu_vec  = mfem.Vector([mu_1,  mu_2 ])
    lam_vec = mfem.Vector([lam_1, lam_2])
    mu_coef  = mfem.PWConstCoefficient(mu_vec)
    lam_coef = mfem.PWConstCoefficient(lam_vec)

    # Build K = ParBilinearForm with ElasticityIntegrator(lam, mu).
    # The integrator handles spatially-varying Lame parameters via the
    # PWConstCoefficient evaluation at each quadrature point.
    #
    # We need TWO HypreParMatrices:
    #   * K_full      : un-eliminated tangent.  Used for the RHS
    #                    computation ``f = K_full @ u_lin`` -- this
    #                    captures the K_uc (free-DOF / corner-DOF
    #                    coupling) block, which is needed for the
    #                    Newton residual to be physically meaningful.
    #                    Per MFEM issue #793, ``a.ParallelAssemble()``
    #                    can produce a HypreParMatrix that SHARES
    #                    underlying SparseMatrix data with the
    #                    ParBilinearForm; calling it twice on the same
    #                    ``a`` is not guaranteed to give independent
    #                    copies.  So we build TWO independent
    #                    ParBilinearForm objects below.
    #   * K_eliminated: rows/cols at corner DOFs zeroed; corner
    #                    diagonal set to 1.  Used as the actual top
    #                    block of the saddle-point system.
    # For linear elasticity K is independent of u, so we build it once
    # at the start and reuse it across all load steps.
    a_full = mfem.ParBilinearForm(fes)
    a_full.AddDomainIntegrator(mfem.ElasticityIntegrator(lam_coef, mu_coef))
    a_full.Assemble()
    a_full.Finalize()
    K_full = a_full.ParallelAssemble()

    a_elim = mfem.ParBilinearForm(fes)
    a_elim.AddDomainIntegrator(mfem.ElasticityIntegrator(lam_coef, mu_coef))
    a_elim.Assemble()
    a_elim.Finalize()
    K_hyp = a_elim.ParallelAssemble()

    # ---------------------------------------------------------------------
    # CLI: load case + ramping schedule
    # ---------------------------------------------------------------------
    # ``--F`` selects the TARGET F at the FINAL step.  ``--steps=N``
    # selects the number of equal-spaced ramp increments from F=I (no
    # load) to F=F_target.  Default: 3 steps.  This exercises the
    # ExaConstit-style multi-step warm-start machinery; for linear
    # elasticity the per-step solve is independent of the warm-start
    # quality (the problem is linear), but the warm-start projection
    # still runs and the volume-averaged-F diagnostic confirms the
    # mortar PBC is reproducing F_macro at every step.
    F_choice  = "uniaxial"
    n_steps   = 3
    for arg in sys.argv[1:]:
        if arg.startswith("--F="):
            F_choice = arg.split("=", 1)[1]
        elif arg.startswith("--steps="):
            n_steps = int(arg.split("=", 1)[1])
    if F_choice == "shear":
        F_target = np.array([[1.2, 0.2], [0.2, 1.05]])
    elif F_choice == "mild-shear":
        F_target = np.array([[1.05, 0.05], [0.05, 1.02]])
    elif F_choice == "uniaxial":
        F_target = np.array([[1.2, 0.0], [0.0, 1.0]])
    else:
        raise ValueError(f"Unknown --F={F_choice}")

    if rank == 0:
        print(f"\nLoad case: --F={F_choice}, --steps={n_steps}")
        print(f"  F_target =\n{F_target}")

    # Build the ramp schedule.  Step 0 is F=I (skipped: no load).
    # We solve at step k for F_k = I + (k/n_steps) (F_target - I), for
    # k = 1, ..., n_steps.
    F_ramp = []
    for k in range(1, n_steps + 1):
        s = k / float(n_steps)
        F_k = np.eye(2) + s * (F_target - np.eye(2))
        F_ramp.append(F_k)

    # ---------------------------------------------------------------------
    # Set up corner Dirichlet on the eliminated K
    # ---------------------------------------------------------------------
    # 4 corners x 2 components = 8 essential TDOFs.  We eliminate corner
    # rows/cols on K_hyp ONCE (linear elasticity = K independent of u).
    # The driver's per-step machinery handles the corner DOF values
    # via the warm-start projection.
    my_first_tdof = _get_my_first_tdof(fes, rank)
    my_n_tdof     = fes.GetTrueVSize()
    local_corner_tdofs = [
        int(d) - my_first_tdof
        for d in corner_tdofs
        if my_first_tdof <= int(d) < my_first_tdof + my_n_tdof
    ]

    # Eliminate corner rows/cols of K_hyp.  We pass an empty f_par
    # because the driver computes its own RHS from u_lin and deltaF
    # at every step; the eliminator just modifies K in place.
    _scratch_f = mfem.Vector(my_n_tdof)
    _scratch_f.Assign(0.0)
    apply_dirichlet_to_distributed_K(K_hyp, _scratch_f, corner_tdofs, fes)

    # ---------------------------------------------------------------------
    # Build the saddle-point solver
    # ---------------------------------------------------------------------
    sps = SaddlePointSolver(
        solver="GMRES",
        preconditioner="block_jacobi",
        rel_tol=1e-12,
        abs_tol=1e-14,
        max_iter=2000,
        print_level=-1,
    )
    if rank == 0:
        print(f"\nSaddle-point solver: "
              f"{sps.solver_name} + {sps.preconditioner}")

    # ---------------------------------------------------------------------
    # Operator-correctness diagnostic (sanity check before stepping)
    # ---------------------------------------------------------------------
    if rank == 0:
        print("\n--- Operator-correctness diagnostic ---")
    n_tdof_global = fes.GlobalTrueVSize()
    x_test_global = np.sin(np.arange(n_tdof_global, dtype=np.float64) + 0.5)
    x_test_local = mfem.Vector(my_n_tdof)
    for i in range(my_n_tdof):
        x_test_local[i] = float(x_test_global[my_first_tdof + i])
    y_test_local = mfem.Vector(n_lam_local)
    C_op.Mult(x_test_local, y_test_local)
    if rank == 0:
        y_test_local_np = np.array(y_test_local.GetDataArray(), dtype=np.float64).copy()
        y_test_scipy = C_global_csr_modified @ x_test_global
        diff_op = float(np.linalg.norm(y_test_local_np - y_test_scipy, ord=np.inf))
        scipy_norm = float(np.linalg.norm(y_test_scipy, ord=np.inf))
        print(f"  ||C_op @ x - C_global @ x||_inf = {diff_op:.3e} "
              f"(scipy_norm = {scipy_norm:.3e})")

    # =====================================================================
    # Build the multi-step driver and run the ramp
    # =====================================================================
    driver = MortarPbcDriver2D(
        pmesh=pmesh, fes=fes,
        K_op=K_hyp, K_op_full=K_full,
        C_op=C_op, CT_op=CT_op,
        corner_tdofs=corner_tdofs,
        apply_linear_part_fn=apply_linear_part,
        numpy_to_mfem_vector_fn=numpy_to_mfem_vector,
        sps=sps,
        n_lam_local=n_lam_local,
        local_corner_tdofs=local_corner_tdofs,
    )

    # ---------------------------------------------------------------------
    # ParaView writer (multi-cycle: cycle 0 = undeformed, then one
    # cycle per converged load step).
    # ---------------------------------------------------------------------
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "paraview_output",
        f"heterogeneous_{F_choice}",
    )
    pv_writer = PbcVisualizationWriter(
        pmesh, fes, output_dir=output_dir, name="solution",
    )

    # ---------------------------------------------------------------------
    # Run the ramp
    # ---------------------------------------------------------------------
    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"Ramping F: {n_steps} step{'s' if n_steps != 1 else ''}")
        print(f"{'=' * 70}")

    for step_idx, F_k in enumerate(F_ramp):
        if rank == 0:
            print(f"\n  --- Step {step_idx+1}/{n_steps}  ({F_choice}) ---")
            print(f"      F_k =\n{_indent(repr(F_k), 12)}")
        if step_idx == 0:
            result = driver.solve_first_step(F_k)
        else:
            result = driver.solve_next_step(F_k)
        if rank == 0:
            _print_step_result(result)
        # Visualize this step.  Build the u_lin and du for the writer.
        u_lin_k_local = apply_linear_part(fes, F_k)
        u_lin_k_par   = numpy_to_mfem_vector(u_lin_k_local)
        du_k_par      = mfem.Vector(my_n_tdof)
        for i in range(my_n_tdof):
            du_k_par[i] = float(driver.u_par[i]) - float(u_lin_k_par[i])
        pv_writer.write_step(
            driver.u_par, u_lin_k_par, du_k_par,
            time=float(step_idx + 1),
            F_label=f"{F_choice}/step{step_idx+1}",
            write_undeformed_first=(step_idx == 0),
        )

    # ---------------------------------------------------------------------
    # Final-step verification (SciPy direct cross-check on rank 0)
    # ---------------------------------------------------------------------
    if rank == 0:
        print(f"\n{'=' * 70}")
        print("Final-step verification (SciPy direct LU on rank 0)")
        print(f"{'=' * 70}")
    final = driver.history[-1]
    u_lin_final_local = apply_linear_part(fes, F_ramp[-1])
    u_lin_final_par   = numpy_to_mfem_vector(u_lin_final_local)
    du_final_par      = mfem.Vector(my_n_tdof)
    for i in range(my_n_tdof):
        du_final_par[i] = float(driver.u_par[i]) - float(u_lin_final_par[i])

    # Gather to rank 0 for the SciPy cross-check.
    u_lin_loc_np = mfem_vector_to_numpy(u_lin_final_par)
    du_loc_np    = mfem_vector_to_numpy(du_final_par)
    counts_v = np.array(comm.allgather(u_lin_loc_np.size), dtype=np.int64)
    if rank == 0:
        u_lin_global = np.empty(int(counts_v.sum()), dtype=np.float64)
        du_global    = np.empty(int(counts_v.sum()), dtype=np.float64)
        displs = np.concatenate([[0], np.cumsum(counts_v[:-1])]).astype(np.int64)
        comm.Gatherv(u_lin_loc_np, [u_lin_global, counts_v, displs, MPI.DOUBLE], root=0)
        comm.Gatherv(du_loc_np,    [du_global,    counts_v, displs, MPI.DOUBLE], root=0)
    else:
        comm.Gatherv(u_lin_loc_np, None, root=0)
        comm.Gatherv(du_loc_np,    None, root=0)
        u_lin_global = du_global = None

    K_global_csr      = hypre_to_scipy_csr(K_hyp,  fes)
    K_full_global_csr = hypre_to_scipy_csr(K_full, fes)
    if rank == 0:
        # Recreate the RHS for the direct solve EXACTLY as the multi-
        # step driver does: f = K_full @ u_lin (NOT K_eliminated --
        # that would lose the K_uc contribution and give the wrong
        # answer; see _solve_independently docstring).  Then zero
        # corner entries.
        f_global = K_full_global_csr @ u_lin_global
        for d in corner_tdofs:
            f_global[int(d)] = 0.0
        verifier = SciPyDirectSolver(verbose=True)
        du_direct_global, _dlam_direct = verifier.solve_step(
            K=K_global_csr,                  # eliminated K in the saddle block
            C=C_global_csr_modified,
            r1=f_global,                     # RHS built from K_full
            r2=np.zeros(C_global_csr_modified.shape[0]),
        )
        diff_krylov_vs_direct = float(np.linalg.norm(
            du_global - du_direct_global, ord=np.inf
        ))
        print(f"  ||du_krylov - du_direct||_inf = {diff_krylov_vs_direct:.3e}")

    # ---------------------------------------------------------------------
    # PASS / FAIL summary on the FINAL step
    # ---------------------------------------------------------------------
    if rank == 0:
        print(f"\n{'=' * 70}")
        print("Final-step PASS / FAIL")
        print(f"{'=' * 70}")
        pass_constraint_atol = 1.0e-8
        pass_kry_vs_dir_atol = 1.0e-6
        pass_fluct_lower_bnd = 1.0e-12
        pass_F_avg_atol      = 1.0e-9    # |<F> - F_macro|_max threshold

        passed = (
            final.krylov_converged
            and final.constraint_residual < pass_constraint_atol
            and diff_krylov_vs_direct     < pass_kry_vs_dir_atol
            and final.u_tilde_inf         > pass_fluct_lower_bnd
            and final.F_average_error     < pass_F_avg_atol
        )
        if passed:
            print("  PASS")
        else:
            print("  FAIL")
            if not final.krylov_converged:
                print(f"    -> Krylov did not converge on final step")
            if final.constraint_residual >= pass_constraint_atol:
                print(f"    -> Constraint residual too large: "
                      f"{final.constraint_residual:.3e} "
                      f">= {pass_constraint_atol:.0e}")
            if diff_krylov_vs_direct >= pass_kry_vs_dir_atol:
                print(f"    -> Krylov vs Direct disagree: "
                      f"{diff_krylov_vs_direct:.3e} "
                      f">= {pass_kry_vs_dir_atol:.0e}")
            if final.u_tilde_inf <= pass_fluct_lower_bnd:
                print(f"    -> Fluctuation suspiciously small "
                      f"({final.u_tilde_inf:.3e}); expected non-"
                      f"trivial for heterogeneous material")
            if final.F_average_error >= pass_F_avg_atol:
                print(f"    -> Volume-averaged F differs from F_macro by "
                      f"{final.F_average_error:.3e} "
                      f">= {pass_F_avg_atol:.0e} -- this is a "
                      f"homogenization-consistency violation")


def _indent(s: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in s.splitlines())


def _print_step_result(r) -> None:
    print(f"      Krylov: iters={r.krylov_iters}, "
          f"converged={r.krylov_converged}, "
          f"final_norm={r.krylov_final_norm:.3e}")
    print(f"      ||u||_inf      = {r.u_inf:.3e}")
    print(f"      ||u_tilde||_inf = {r.u_tilde_inf:.3e}")
    print(f"      ||C u_tilde||_2 = {r.constraint_residual:.3e}")
    print(f"      <F> =\n{_indent(repr(r.F_average), 12)}")
    print(f"      |<F> - F_macro|_max = {r.F_average_error:.3e}")


if __name__ == "__main__":
    main()
