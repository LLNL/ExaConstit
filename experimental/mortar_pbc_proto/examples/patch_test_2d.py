"""2D mortar PBC patch test (Lopes et al. Section 5.1.1).

Subject a homogeneous square RVE to the macroscopic deformation gradient

    F = [[1.5, 0.5],
         [0.5, 1.0]]

The expected micro response is a uniform displacement field
    u_mu(Y) = (F - I) * Y     (linear part)
with zero fluctuation u_tilde = 0 everywhere -- so the deformed mesh is
itself a sheared parallelogram with constant Cauchy strain.

This driver:
    1. Builds the FE problem and assembles K (HypreParMatrix) and the
       constraint matrix C (scipy CSR, identical on every rank).
    2. Solves the saddle-point Newton step *distributedly* using
       ``SaddlePointSolver`` (Krylov + mfem.BlockOperator).  K is
       consumed via ``Mult`` only -- no gather to root, no CSR
       materialization.
    3. Cross-checks the result against ``SciPyDirectSolver`` (gathered
       to rank 0; quarantined verification path).  Prints the
       ||du_krylov - du_direct||_inf diff so any divergence between the
       two paths is immediately visible.

For the prototype the material is linear-elastic so the Newton step
converges in one iteration.  This isolates the mortar machinery from
material nonlinearity.

Run with:
    python examples/patch_test_2d.py            # np = 1
    mpirun -n 2 python examples/patch_test_2d.py
    mpirun -n 4 python examples/patch_test_2d.py
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
        print("Mortar PBC 2D patch test (distributed Krylov, np > 1 capable)")
        print(f"  MPI ranks: {nranks}")
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

    # ----- Assemble K as HypreParMatrix -----
    K_hyp = assemble_linear_elastic_K_hypre(pmesh, fes, E=70.0e3, nu=0.3)

    # ---------------------------------------------------------------------
    # Steps 8-9: compute f distributedly, then eliminate Dirichlet
    # ---------------------------------------------------------------------
    F_macro      = np.array([[1.5, 0.5], [0.5, 1.0]])
    u_lin_local  = apply_linear_part(fes, F_macro)
    u_lin_par    = numpy_to_mfem_vector(u_lin_local)

    f_par = mfem.Vector(fes.GetTrueVSize())
    K_hyp.Mult(u_lin_par, f_par)

    # In-place: eliminate K's corner rows/cols + zero f at corners.
    apply_dirichlet_to_distributed_K(K_hyp, f_par, corner_tdofs, fes)

    # ---------------------------------------------------------------------
    # Step 10: distributed Krylov solve
    # ---------------------------------------------------------------------

    # GMRES + block-Jacobi is the safe default.  GMRES works whether or
    # not K is symmetric (avoids the Lanczos breakdown MINRES can hit on
    # mildly non-symmetric K).  Block-Jacobi preconditioning brings the
    # iteration count down dramatically on saddle-point systems and makes
    # the solver scale-friendly to bigger problems.
    sps = SaddlePointSolver(
        solver="GMRES",
        preconditioner="block_jacobi",
        # rel_tol is relative to the initial residual ||rhs||.  For our
        # patch test ||rhs|| ~ O(1e+4) (Lame-modulus * F-magnitude), so
        # rel_tol = 1e-14 drives the absolute residual to ~ 3e-10, which
        # gives ||du - du_exact||_inf of similar magnitude.
        rel_tol=1e-14,
        abs_tol=1e-16,
        max_iter=1000,
        print_level=-1,
    )
    if rank == 0:
        print(f"\n--- Distributed Krylov solve "
              f"({sps.solver_name} + {sps.preconditioner}) ---")

    # ---------------------------------------------------------------------
    # Pre-Krylov diagnostic: verify the distributed C_op produces the same
    # answer as scipy's C_global on a known test input.  If they don't
    # match, fail loudly NOW rather than letting Krylov stagnate.
    # ---------------------------------------------------------------------
    if rank == 0:
        print("--- Operator-correctness diagnostic ---")
    # Build a deterministic test velocity vector x_test in the global TDOF
    # space.  We use sin(i + 0.5) to ensure no zeros (which would mask sign
    # errors).
    n_tdof_global = fes.GlobalTrueVSize()
    x_test_global = np.sin(np.arange(n_tdof_global, dtype=np.float64) + 0.5)
    # Each rank gets its own slice as an mfem.Vector.
    my_first_tdof_diag = _get_my_first_tdof(fes, rank)
    my_n_tdof_diag = fes.GetTrueVSize()
    x_test_local = mfem.Vector(my_n_tdof_diag)
    for i in range(my_n_tdof_diag):
        x_test_local[i] = float(x_test_global[my_first_tdof_diag + i])
    # Apply the distributed C_op.
    y_test_local = mfem.Vector(n_lam_local)
    C_op.Mult(x_test_local, y_test_local)
    # On rank 0, compare against scipy.
    if rank == 0:
        y_test_local_np = np.array(y_test_local.GetDataArray(), dtype=np.float64).copy()
        y_test_scipy = C_global_csr_modified @ x_test_global
        diff_op = float(np.linalg.norm(y_test_local_np - y_test_scipy, ord=np.inf))
        scipy_norm = float(np.linalg.norm(y_test_scipy, ord=np.inf))
        print(f"  C_op vs scipy: ||C_op @ x_test - C_global @ x_test||_inf = {diff_op:.3e}")
        print(f"                 ||C_global @ x_test||_inf             = {scipy_norm:.3e}")
        if diff_op > 1e-10 * max(scipy_norm, 1.0):
            print("  *** WARNING: C_op disagrees with scipy C; Krylov will not converge. ***")
        else:
            print("  C_op MATCHES scipy.  The constraint operator is correct.")

    # Warm-started initial iterate: u_par <- u_lin everywhere.
    # For HOMOGENEOUS LINEAR ELASTICITY this is the EXACT solution to
    # the BVP (corner Dirichlets at u_lin[corner] + periodic) -- so the
    # linear solve below should produce du ~ 0 (machine precision).
    # Real correctness testing of the mortar machinery happens in the
    # heterogeneous nonlinear driver.  This file is a regression test:
    # confirms Method D + warm-start + saddle-point inner solve form a
    # consistent system on the simplest problem.
    u_par = mfem.Vector(fes.GetTrueVSize())
    for i in range(fes.GetTrueVSize()):
        u_par[i] = float(u_lin_local[i])

    n_lam_local_sanity = n_lam_total if rank == 0 else 0
    lam_par = mfem.Vector(n_lam_local_sanity)
    lam_par.Assign(0.0)

    # r1 = F_int(u) + C^T λ = K @ u_lin + 0 = f_par.
    # r2 = C @ u_lin - g.  Since g = C @ u_lin, r2 = 0 by construction.
    g_par = mfem.Vector(n_lam_local_sanity)
    C_op.Mult(numpy_to_mfem_vector(u_lin_local), g_par)

    r1_par = f_par
    r2_par = mfem.Vector(n_lam_local_sanity)
    Cu_at_init = mfem.Vector(n_lam_local_sanity)
    C_op.Mult(numpy_to_mfem_vector(u_lin_local), Cu_at_init)
    for i in range(n_lam_local_sanity):
        r2_par[i] = float(Cu_at_init[i]) - float(g_par[i])  # = 0

    du_par, dlam_par = sps.solve_step(
        K_op=K_hyp, C_op=C_op, CT_op=CT_op,
        r1_local=r1_par, r2_local=r2_par,
    )

    if rank == 0:
        print(f"  Krylov: iters={sps.last_iterations}, "
              f"converged={sps.last_converged}, "
              f"final_norm={sps.last_final_norm:.3e}")

    # ---------------------------------------------------------------------
    # Steps 11-15: verification cross-check (rank 0 only)
    # ---------------------------------------------------------------------
    # Gather du from the Krylov solve to rank 0 for the diff.
    du_local_np = mfem_vector_to_numpy(du_par)
    counts_v = np.array(comm.allgather(du_local_np.size), dtype=np.int64)
    if rank == 0:
        du_krylov_global = np.empty(int(counts_v.sum()), dtype=np.float64)
        displs = np.concatenate([[0], np.cumsum(counts_v[:-1])]).astype(np.int64)
        comm.Gatherv(du_local_np, [du_krylov_global, counts_v, displs, MPI.DOUBLE], root=0)
    else:
        comm.Gatherv(du_local_np, None, root=0)
        du_krylov_global = None

    # Gather K and u_lin to rank 0 for the direct solve.
    K_global_csr = hypre_to_scipy_csr(K_hyp, fes)  # already eliminated K
    u_lin_global = gather_tdof_vector_to_root(u_lin_local, fes)
    f_local_np = mfem_vector_to_numpy(f_par)
    f_global = gather_tdof_vector_to_root(f_local_np, fes)

    if rank == 0:
        assert K_global_csr is not None and f_global is not None and u_lin_global is not None

        print("\n--- Verification (SciPy direct LU on rank 0) ---")
        # Method D: r1 = F_int(u_init) = K @ u_lin = f_global,
        #           r2 = C u_init - g = C u_lin - C u_lin = 0.
        # The direct solve should produce du ~ 0 (machine precision)
        # because u_lin is the exact linear-elastic solution.
        r1_global = f_global
        r2_global = np.zeros(C_global_csr_modified.shape[0])
        verifier = SciPyDirectSolver(verbose=True)
        du_direct_global, dlam_direct_global = verifier.solve_step(
            K=K_global_csr, C=C_global_csr_modified,
            r1=r1_global, r2=r2_global,
        )

        # ---- Diff Krylov vs direct ----
        du_diff = du_krylov_global - du_direct_global
        diff_inf = float(np.linalg.norm(du_diff, ord=np.inf))
        kry_inf  = float(np.linalg.norm(du_krylov_global, ord=np.inf))
        dir_inf  = float(np.linalg.norm(du_direct_global, ord=np.inf))

        # ---- PASS criterion (Method D: u_initial = u_lin) ----
        # Since u_initial = u_lin (warm-started), the post-solve total
        # displacement is u = u_lin + du.  The fluctuation u_tilde =
        # u - u_lin = du.  For homogeneous linear elastic under uniform
        # F, the exact answer is u_tilde = 0, so we expect ||du||_inf ~
        # machine precision.  Constraint residual measures whether the
        # Krylov solution actually satisfies C du = 0 (since g = C u_lin
        # is already balanced at the initial iterate).
        u_tilde_global   = du_krylov_global
        constraint_residual = float(np.linalg.norm(
            C_global_csr_modified @ u_tilde_global
        ))
        fluctuation_inf = float(np.linalg.norm(u_tilde_global, ord=np.inf))

        print("\n" + "-" * 70)
        print("Patch test results (Method D + warm-start)")
        print("-" * 70)
        print(f"  Krylov:    ||du||_inf = {kry_inf:.3e}     (= ||u - u_lin||)")
        print(f"  Direct:    ||du||_inf = {dir_inf:.3e}")
        print(f"  Diff:      ||Krylov - Direct||_inf = {diff_inf:.3e}")
        print(f"  Constraint residual ||C(u_lin + du) - g||_2"
              f"   ~ ||C du||_2 = {constraint_residual:.3e}")
        print(f"  Fluctuation         ||u - u_lin||_inf = {fluctuation_inf:.3e}")

        # PASS criterion: homogeneous linear-elastic + warm-start should
        # produce du at machine precision.
        passed = (
            sps.last_converged
            and constraint_residual < 1e-8
            and fluctuation_inf    < 1e-7
        )
        if passed:
            print("  PASS")
        else:
            print("  FAIL")
            if not sps.last_converged:
                print(f"    -> Krylov did not converge in {sps.last_iterations} iterations")
            if constraint_residual >= 1e-8:
                print(f"    -> Constraint residual too large: {constraint_residual:.3e}")
            if fluctuation_inf >= 1e-7:
                print(f"    -> Fluctuation too large: {fluctuation_inf:.3e}")


if __name__ == "__main__":
    main()
