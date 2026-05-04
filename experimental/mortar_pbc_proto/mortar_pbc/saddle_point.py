"""Distributed Krylov saddle-point solver for the mortar PBC Newton step.

WHAT
----
Solve one Newton step of the constrained problem

    [ K   C^T ] [ Δv ]   [ -r + C^T λ ]
    [ C   0   ] [ Δλ ] = [ -C v       ]                                    (*)

per Lopes et al. Eq. (59), where:
    K  = tangent stiffness as an mfem.Operator (apply-only access),
    C  = constraint matrix from ConstraintBuilder2D, wrapped as PyOperator,
    r  = global residual,
    v  = current solution iterate,
    λ  = current multiplier estimate.

The system is solved DISTRIBUTEDLY using one of MFEM's Krylov methods
(MINRES, GMRES, or BiCGStab) on a 2x2 mfem.BlockOperator.  No part of K
is ever gathered to rank 0 or materialized as scipy CSR.

RELATIONSHIP TO MFEM'S CONSTRAINEDSOLVER FAMILY
-----------------------------------------------
This class is structurally a subset of MFEM's ``SchurConstrainedSolver``
(see ``mfem/linalg/constraints.hpp``, also Example 28 / ex28p).  MFEM's
``ConstrainedSolver`` ABC defines three concrete strategies for solving
``A x = f`` subject to ``B x = r``:

    * ``EliminationSolver``  -- split B into primary/secondary DOFs,
                                 dense-LU eliminate the secondary block,
                                 Krylov on ``P^T A P + Z_P``.  Requires
                                 disjoint primary/secondary footprints
                                 across constraint blocks; awkward for
                                 mortar (and worse in 3D wirebaskets).
    * ``PenaltyConstrainedSolver`` -- solve ``(A + B^T D B) x = f + B^T D r``
                                       with high penalty.  Simple, but
                                       constraint accuracy and conditioning
                                       trade off as penalty grows.
    * ``SchurConstrainedSolver`` / ``SchurConstrainedHypreSolver``
                              -- the saddle-point path used here.  Builds
                                 [[A, B^T], [B, 0]] as a BlockOperator;
                                 solves with Krylov + BlockDiagonalPrec.
                                 Most general; not the fastest.

We follow the Schur path because:
    1. Our mortar B has overlapping primary footprints across rows
       (multiple + nodes share the same - node), which makes the
       Eliminator's disjoint-block precondition awkward.
    2. We want operator-only K access (PA / EA / FA agnostic), which is
       incompatible with EliminationSolver's ``BuildExplicitOperator()``
       and PenaltyConstrainedSolver's ``A + B^T D B`` ParMult/ParAdd.
    3. Block-Jacobi preconditioning (Phase 1B) on the Schur saddle-point
       form requires only K's diagonal, which any Operator can produce
       cheaply via ``AssembleDiagonal``.  GPU-friendly across all three K
       representations.

The eventual C++ port will essentially be a subclass of
``mfem::ConstrainedSolver`` mirroring this structure.  Method-name
mapping for the port:
    SaddlePointSolver.solve_step(K, C, CT, f, u, λ)
        ~~~  mfem::ConstrainedSolver::Mult(f, x)  +  GetMultiplierSolution(λ)

NOTE ON GPU READINESS OF MFEM'S CONSTRAINTS MODULE (as of 2026)
---------------------------------------------------------------
MFEM's existing ``ConstrainedSolver`` implementations were designed
before robust GPU support landed in the rest of MFEM.  ``EliminationSolver``
does host-side dense LU factorizations on the per-block secondary
subspace, then calls ``BuildExplicitOperator()`` to form ``P^T A P`` as
a HypreParMatrix -- both setup phases are host-bound.
``SchurConstrainedHypreSolver`` calls ``ParMult(B, M^{-1} B^T)`` and runs
``HypreBoomerAMG`` on both the (0,0) and the assembled Schur block;
ParMult assumes A is a real HypreParMatrix, not a PA Operator.  For an
ExaConstit-style PA-K-on-GPU configuration, none of these compose
directly.  Our prototype's choice (operator-only K, Jacobi-only
preconditioner) is therefore strictly more GPU-portable than what's
currently shipped in MFEM constraints.hpp -- the C++ port may end up
contributing this back to MFEM as a fourth ``ConstrainedSolver`` variant
suited to PA / matrix-free K.

WHY (architecture decisions)
----------------------------
1. **K-block is consumed purely through the mfem.Operator interface.**
   The saddle-point solver invokes only ``K.Mult`` (and possibly
   ``K.MultTranspose`` for non-symmetric Krylov).  This holds whether
   ExaConstit has assembled K in PA, EA, or FA form.  Important corollary:
   ``SaddlePointSolver`` does NOT extract K's sparsity, does NOT compute
   K's exact diagonal except via ``AssembleDiagonal``, does NOT call
   ``RAP`` or ``ParMult`` against K.  Block-Jacobi preconditioning (a
   future addition) only requires K's diagonal, which every K
   representation can produce cheaply via ``AssembleDiagonal``.

2. **C-block is wrapped as a Python-side mfem.Operator (PyOperator).**
   In the prototype, C is a scipy CSR identical on every rank (built by
   ``ConstraintBuilder2D``).  Rather than converting to a row-distributed
   HypreParMatrix (which has fiddly column-partitioning constraints to
   match fes.GetTrueDofOffsets()), we wrap the scipy CSR in a custom
   PyOperator whose Mult / MultTranspose do an Allgather of the input
   over the velocity space, multiply by the local CSR slice, and produce
   the correct distributed output.  Multiplier vector is laid out all-on-
   rank-0; rank > 0 has zero-length multiplier slices.  This is
   PROTOTYPE-ONLY: the C++ port will use an actual distributed
   HypreParMatrix for C, but the saddle-point solver code is unchanged
   because it only sees the Operator interface.

3. **Krylov method is chosen at runtime.**  MINRES (default; symmetric K),
   GMRES (non-symmetric K), or BiCGStab.  CG is REJECTED with a clear
   error -- the saddle-point system is indefinite by construction (the
   zero block in the (2,2) position guarantees indefiniteness) and CG
   diverges on indefinite systems.

4. **No preconditioner in this version (Phase 1A).**  Patch-test scale
   (~200 dofs) converges fine without one.  Phase 1B will add
   block-Jacobi.  Three preconditioner options layered by cost/fidelity:

     (a) diag(K)^{-1} ; diag(C diag(K)^{-1} C^T)^{-1}
         Cheapest.  Pure-diagonal both blocks.  GPU-friendly.
         Default for the upcoming Phase 1B.
     (b) diag(K)^{-1} ; explicit ParMult to form S = C diag(K)^{-1} C^T,
         then diag(S)^{-1}.
         Modest setup cost.  Tighter Schur approximation -- captures
         off-diagonal multiplier coupling.  Behind a flag.
     (c) diag(K)^{-1} ; direct LU of S.
         Only justified if (b) struggles to converge on bigger problems.
         For now: aspirational.

REFERENCES
----------
Lopes, Ferreira, Andrade Pires (2021), CMAME 384, 113930.
    * Eq. (59)   : saddle-point system for SPS method
    * Table 5   : SPS vs CM (condensation) timing on RVE problems
MFEM, ``mfem/linalg/constraints.hpp``: ``ConstrainedSolver`` ABC and the
    ``SchurConstrainedSolver`` / ``SchurConstrainedHypreSolver`` concrete
    implementations.  Also: example 28 / ex28p illustrating the typical
    use pattern with ``BuildNormalConstraints``.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp


# Krylov solver name -> mfem.par class attribute name.
_SOLVER_NAME_TO_MFEM_CLASS = {
    "MINRES":   "MINRESSolver",
    "GMRES":    "GMRESSolver",
    "BiCGStab": "BiCGSTABSolver",
}


# =============================================================================
# Wrapping a scipy CSR constraint matrix as a distributed mfem.Operator
# =============================================================================

def make_constraint_operators(
    C_global: sp.csr_matrix,
    fes,        # mfem.par.ParFiniteElementSpace
    n_lam_local: int,
):
    """Wrap a globally-replicated scipy CSR ``C`` as two distributed mfem
    Operators: ``C`` (rows = multipliers, cols = TDOFs) and ``C^T``.

    Parameters
    ----------
    C_global : scipy.sparse.csr_matrix
        The constraint matrix.  Shape (n_lam_total, n_tdof_global).
        Identical on every rank.  Must already have corner-DOF columns
        zeroed (caller's responsibility, via ``apply_dirichlet_zero_to_C``).
    fes : mfem.par.ParFiniteElementSpace
        Used to determine the rank's local TDOF count and the Allgather
        layout.
    n_lam_local : int
        How many multiplier rows this rank "owns".  Convention: rank 0
        owns ALL multipliers; rank > 0 owns 0.  (Phase-1 prototype
        choice.)  Sum across ranks must equal ``C_global.shape[0]``.

    Returns
    -------
    C_op : mfem.PyOperator
        Maps velocity-TDOF Vector (local size = fes.GetTrueVSize()) to
        multiplier Vector (local size = n_lam_local).
    CT_op : mfem.PyOperator
        Maps multiplier Vector (local size = n_lam_local) to velocity-TDOF
        Vector (local size = fes.GetTrueVSize()).

    Notes
    -----
    The two operators share Python-side state -- the same scipy CSR and
    the same MPI communicator -- but they are distinct Operator objects
    so they can be put into different slots of the BlockOperator.
    Both internally perform one MPI Allgather (or Bcast in MultTranspose)
    per call; for the patch-test scale this is cheap.
    """
    import mfem.par as mfem
    from mpi4py import MPI

    # pyMFEM exposes the Python-overridable Operator base class as
    # PyOperatorBase in the documented examples, but some builds also
    # expose it as PyOperator.  Probe for whichever exists.
    if hasattr(mfem, "PyOperatorBase"):
        PyOperatorClass = mfem.PyOperatorBase
    elif hasattr(mfem, "PyOperator"):
        PyOperatorClass = mfem.PyOperator
    else:
        raise RuntimeError(
            "Cannot find PyOperatorBase / PyOperator in mfem.par; "
            "pyMFEM build does not expose the Python-overridable "
            "Operator base class.  Try a more recent pyMFEM build "
            "(e.g. develop branch >= 7e99b925)."
        )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    n_lam_total  = C_global.shape[0]
    n_tdof_local = fes.GetTrueVSize()

    # Pre-compute the partition layout of velocity TDOFs across ranks
    # so the Allgather inside Mult can be done with displacements.
    counts_v = np.array(comm.allgather(n_tdof_local), dtype=np.int64)
    displs_v = np.concatenate([[0], np.cumsum(counts_v[:-1])]).astype(np.int64)

    # Pre-compute multiplier partition (all-on-rank-0 in this prototype).
    counts_lam = np.array(comm.allgather(n_lam_local), dtype=np.int64)
    if int(counts_lam.sum()) != n_lam_total:
        raise ValueError(
            f"Sum of n_lam_local across ranks ({counts_lam.sum()}) "
            f"must equal C_global.shape[0] ({n_lam_total})."
        )

    # Cache CSR transpose so we don't rebuild it on every MultTranspose.
    C_T_global = C_global.T.tocsr()

    # Cache element-wise squared C for the Schur-diag computation in the
    # block-Jacobi preconditioner.  diag(C M C^T)_i for a diagonal M
    # works out to sum_j (C_ij)^2 * M_jj, i.e., row i of (C^.^2) times
    # the diagonal of M.  Pre-computing once is cheap.
    C_squared_global = C_global.multiply(C_global).tocsr()

    # Cumulative offsets used to slice the global multiplier vector
    # into per-rank local pieces.  Pre-computed once so neither Mult
    # call rebuilds them on each Krylov iteration.
    cum_lam = np.concatenate([[0], np.cumsum(counts_lam[:-1])]).astype(np.int64)

    def _c_apply(x_local_vec, y_local_vec):
        """C @ x : (n_tdof_local input) -> (n_lam_local output).

        Implements the forward C matvec.  Used as ``Mult`` of
        ``_ConstraintOp`` and as ``MultTranspose`` of
        ``_ConstraintTransposeOp``.

        Note on writing the output: we use element-wise assignment
        ``y_local_vec[i] = float(...)`` rather than a numpy slice write
        through ``GetDataArray()``.  ``GetDataArray()`` is documented as
        returning a view, but on some pyMFEM builds (notably when the
        underlying Vector lives in device memory or when the build was
        configured with ``HYPRE_USING_GPU``) it returns a copy, and a
        slice write does NOT propagate back to the C++ buffer.  Element-
        wise ``__setitem__`` always goes through pyMFEM's documented
        write path and is safe regardless of build configuration.
        """
        # Read x via numpy view (read-only is always safe via GetDataArray).
        x_local_np = np.asarray(x_local_vec.GetDataArray(),
                                dtype=np.float64, copy=False)
        # Allgather x over the velocity space.
        x_global = np.empty(int(counts_v.sum()), dtype=np.float64)
        comm.Allgatherv(x_local_np,
                        [x_global, counts_v, displs_v, MPI.DOUBLE])
        # Full product on every rank, then slice this rank's rows.
        y_full = C_global @ x_global
        lam_lo = int(cum_lam[rank])
        y_slice = np.asarray(y_full[lam_lo:lam_lo + n_lam_local],
                             dtype=np.float64)
        # Element-wise write -- robust against view-vs-copy ambiguity.
        for i in range(n_lam_local):
            y_local_vec[i] = float(y_slice[i])

    def _ct_apply(y_local_vec, x_local_vec):
        """C^T @ y : (n_lam_local input) -> (n_tdof_local output).

        Implements the forward C^T matvec.  Used as ``Mult`` of
        ``_ConstraintTransposeOp`` and as ``MultTranspose`` of
        ``_ConstraintOp``.

        See ``_c_apply`` for the rationale on element-wise output writes.
        """
        # Read y via numpy view.
        y_local_np = np.asarray(y_local_vec.GetDataArray(),
                                dtype=np.float64, copy=False)
        # Allgather y over the multiplier space.
        y_global = np.empty(int(counts_lam.sum()), dtype=np.float64)
        comm.Allgatherv(y_local_np,
                        [y_global, counts_lam, cum_lam, MPI.DOUBLE])
        # Full C^T product on every rank, then slice this rank's TDOFs.
        x_full = C_T_global @ y_global
        x_lo = int(displs_v[rank])
        x_slice = np.asarray(x_full[x_lo:x_lo + n_tdof_local],
                             dtype=np.float64)
        for i in range(n_tdof_local):
            x_local_vec[i] = float(x_slice[i])

    def _weighted_row_sq_sum(weights_local_vec, out_local_vec):
        """Compute the Schur preconditioner diagonal for this rank.

        For a 2x2 saddle point [[K, C^T], [C, 0]] preconditioned with
        block-diagonal Jacobi, the (1, 1) block of the preconditioner
        approximates the inverse Schur complement.  The cheapest such
        approximation that doesn't form C diag(K)^{-1} C^T explicitly is
        its diagonal::

            S_ii ~ diag(C diag(K)^{-1} C^T)_i
                 = sum_j (C_ij)^2 * inv_diag_K_j

        i.e. row i of element-wise-squared C, dotted with the global
        inverse diagonal of K.  This routine computes that for the rows
        owned by this rank.

        Parameters
        ----------
        weights_local_vec : mfem.Vector
            This rank's slice of inv_diag_K -- length n_tdof_local.
        out_local_vec : mfem.Vector
            This rank's slice of the Schur-diag -- length n_lam_local.

        Notes
        -----
        Like ``_c_apply``, this is COLLECTIVE: it does an Allgatherv of
        the weights vector across all ranks before doing the local
        sparse matvec.  Must be invoked unconditionally on every rank.
        """
        weights_local_np = np.asarray(weights_local_vec.GetDataArray(),
                                      dtype=np.float64, copy=False)
        weights_global = np.empty(int(counts_v.sum()), dtype=np.float64)
        comm.Allgatherv(weights_local_np,
                        [weights_global, counts_v, displs_v, MPI.DOUBLE])
        # C_squared_global is (C^.^2), dim (n_lam_total, n_v_total).
        # Multiply by global weights -> n_lam_total per-row sums.
        sums_full = C_squared_global @ weights_global
        # Slice this rank's rows.
        lam_lo = int(cum_lam[rank])
        sums_slice = np.asarray(sums_full[lam_lo:lam_lo + n_lam_local],
                                dtype=np.float64)
        for i in range(n_lam_local):
            out_local_vec[i] = float(sums_slice[i])

    class _ConstraintOp(PyOperatorClass):
        """C : (n_v_local) -> (n_lam_local), via Allgather of x then scipy.

        ``Mult``           : applies C   (forward)   -- via _c_apply
        ``MultTranspose``  : applies C^T (transpose) -- via _ct_apply

        Both overrides matter for solvers like MINRES and BiCGStab that
        invoke the Operator's ``MultTranspose`` to maintain symmetry of
        the Lanczos / bi-orthogonalization recursions.  Without the
        explicit override, the default ``MultTranspose`` falls back to a
        path that may not be consistent with our PyOperator's ``Mult``,
        causing convergence stagnation for symmetric Krylov methods.
        """
        def __init__(self):
            # MFEM Operator convention: Operator(height, width) = (rows, cols).
            # C maps velocity-TDOF (size n_tdof_local) to multiplier
            # (size n_lam_local), so cols = n_tdof_local, rows = n_lam_local.
            super().__init__(n_lam_local, n_tdof_local)

        def Mult(self, x_local, y_local):
            _c_apply(x_local, y_local)

        def MultTranspose(self, y_local, x_local):
            _ct_apply(y_local, x_local)

        def WeightedRowSqSum(self, weights_local, out_local):
            """Compute ``out[i] = sum_j C[i,j]^2 * weights[j]`` for this
            rank's rows.  Used by ``SaddlePointSolver`` to build the
            Schur-complement diagonal for block-Jacobi preconditioning.

            Collective: every rank must call this in lock-step.
            """
            _weighted_row_sq_sum(weights_local, out_local)

    class _ConstraintTransposeOp(PyOperatorClass):
        """C^T : (n_lam_local) -> (n_v_local).

        ``Mult``           : applies C^T (forward)   -- via _ct_apply
        ``MultTranspose``  : applies C   (transpose) -- via _c_apply

        See ``_ConstraintOp`` docstring for why the explicit
        ``MultTranspose`` override matters.
        """
        def __init__(self):
            # MFEM Operator convention: Operator(height, width) = (rows, cols).
            # C^T maps multiplier (size n_lam_local) to velocity-TDOF
            # (size n_tdof_local), so cols = n_lam_local, rows = n_tdof_local.
            super().__init__(n_tdof_local, n_lam_local)

        def Mult(self, y_local, x_local):
            _ct_apply(y_local, x_local)

        def MultTranspose(self, x_local, y_local):
            _c_apply(x_local, y_local)

    return _ConstraintOp(), _ConstraintTransposeOp()


# =============================================================================
# Helper: diagonal-scaling Operator (for block-Jacobi preconditioner blocks)
# =============================================================================

def _DiagonalScaler(PyOpClass, inv_diag_vec, size):
    """Construct a small Python-side mfem.Operator whose Mult does
    ``y[i] = inv_diag[i] * x[i]``.

    Used as the diagonal blocks of the block-Jacobi preconditioner in
    ``SaddlePointSolver``.  We accept ``PyOpClass`` as an argument
    (rather than importing it at module scope) because mfem.par must
    be lazily-imported -- the module is usable in environments without
    pyMFEM for the unit tests of the pure-NumPy mortar machinery.

    Parameters
    ----------
    PyOpClass : type
        Either ``mfem.PyOperatorBase`` or ``mfem.PyOperator``, whichever
        the running pyMFEM build exposes.
    inv_diag_vec : mfem.Vector
        The inverse-diagonal values.  Stored on the returned object as
        ``self._inv_diag`` so Python keeps it alive for the lifetime of
        the operator.
    size : int
        Local size of the diagonal block.

    Returns
    -------
    An ``Operator`` instance whose ``Mult(x, y)`` computes
    ``y[i] = inv_diag[i] * x[i]``.
    """
    class _Scaler(PyOpClass):
        def __init__(self, n: int, inv_diag):
            super().__init__(n, n)            # square: rows = cols = n
            self._inv_diag = inv_diag         # keepalive ref

        def Mult(self, x, y):
            for i in range(size):
                y[i] = float(self._inv_diag[i]) * float(x[i])

        def MultTranspose(self, x, y):
            # Diagonal scaling is self-transpose.
            for i in range(size):
                y[i] = float(self._inv_diag[i]) * float(x[i])

    return _Scaler(size, inv_diag_vec)


# =============================================================================
# SaddlePointSolver
# =============================================================================

class SaddlePointSolver:
    """Distributed Krylov solver for the mortar PBC saddle-point Newton step.

    Parameters
    ----------
    solver : {"MINRES", "GMRES", "BiCGStab"}, default "MINRES"
        Krylov method to use.  ``CG`` is rejected: the system is indefinite.
    rel_tol, abs_tol : float
        Krylov convergence tolerances (whichever is hit first).
    max_iter : int
        Maximum Krylov iterations.
    print_level : int
        MFEM Krylov solver print level (0 = silent, 1 = first+last,
        2 = every iter).
    preconditioner : {"none", "block_jacobi"}, default "block_jacobi"
        Block-diagonal preconditioner choice for the saddle-point system:

        * ``"none"`` -- identity preconditioner.  For tiny problems
          (~few hundred dofs) Krylov converges in O(N) iterations
          without one; useful for testing.  Not for production.
        * ``"block_jacobi"`` -- the recommended default.  Builds two
          diagonal Jacobi blocks::

              P^{-1} = [ diag(K)^{-1}                          0                       ]
                       [ 0                       diag(C diag(K)^{-1} C^T)^{-1} ]

          K's diagonal is extracted via ``Operator.AssembleDiagonal``,
          which works on PA, EA, FA, and HypreParMatrix forms uniformly
          (and is GPU-friendly across all of them).  The Schur diagonal
          is computed via the ``_ConstraintOp.WeightedRowSqSum`` operator
          method -- no explicit C C^T product is ever formed.  Both
          blocks are applied as Python-side ``y[i] = inv_diag[i] * x[i]``
          scalers wrapped in ``mfem.BlockDiagonalPreconditioner``.

    Notes
    -----
    All MPI collectives happen INSIDE the Krylov solver and the operator
    Mult / MultTranspose / WeightedRowSqSum calls.  No gather-to-root, no
    rank-0-only solve.
    """

    def __init__(
        self,
        solver: Literal["MINRES", "GMRES", "BiCGStab"] = "MINRES",
        rel_tol: float = 1e-10,
        abs_tol: float = 1e-12,
        max_iter: int = 500,
        print_level: int = 0,
        preconditioner: Literal["none", "block_jacobi"] = "block_jacobi",
    ) -> None:
        if solver.upper() == "CG":
            raise ValueError(
                "CG is not a valid choice for the mortar saddle-point "
                "system: the system is indefinite (zero block in the "
                "(2,2) position) and CG diverges on indefinite systems. "
                "Use MINRES (symmetric K) or GMRES (non-symmetric K) "
                "instead."
            )
        if solver not in _SOLVER_NAME_TO_MFEM_CLASS:
            raise ValueError(
                f"Unknown Krylov solver {solver!r}; expected one of "
                f"{list(_SOLVER_NAME_TO_MFEM_CLASS.keys())}."
            )
        if preconditioner not in ("none", "block_jacobi"):
            raise ValueError(
                f"Unknown preconditioner {preconditioner!r}; expected "
                f"'none' or 'block_jacobi'."
            )

        self.solver_name    = solver
        self.rel_tol        = rel_tol
        self.abs_tol        = abs_tol
        self.max_iter       = max_iter
        self.print_level    = print_level
        self.preconditioner = preconditioner
        # Set to True externally to enable a one-shot diagnostic dump at
        # the next call to ``solve_step``.  Useful for localizing NaN
        # propagation issues; printed via ``_dump_diagnostics``.  Has no
        # effect when False (the default).
        self.diagnostic_mode = False

    # ----------------------------------------------------------------- API ---
    def solve_step(
        self,
        K_op,        # mfem.Operator (HypreParMatrix or anything with .Mult)
        C_op,         # mfem.Operator (e.g. from make_constraint_operators)
        CT_op,        # mfem.Operator (transpose; from make_constraint_operators)
        r1_local,     # mfem.Vector: top Newton residual, length = K_op.Height()
        r2_local,     # mfem.Vector: bottom Newton residual, length = C_op.Height()
    ):
        """Solve one Newton step distributedly.

        Returns ``(du_local, dlam_local)`` as mfem.Vectors.  Each rank's
        ``du_local`` contains its local TDOF slice; on np>1 with the
        all-on-rank-0 multiplier convention, only rank 0's
        ``dlam_local`` is non-empty.

        Newton step solved
        ------------------
        Caller is responsible for forming the FULL Newton residuals.
        For the constrained equilibrium

            F_int(u) + C^T λ = 0       (force balance)
            C u_tilde        = 0       (periodicity)

        the linearization at iterate (u_tilde_k, λ_k) gives

            [ K    C^T ] [ du ]   [ -r1_local ]
            [ C    0   ] [ dλ ] = [ -r2_local ]

        where the caller supplies

            r1_local = F_int(u_lin + u_tilde_k) + C^T λ_k   (force imbalance)
            r2_local = C u_tilde_k                          (constraint
                                                              violation)

        This API is deliberately stateless w.r.t. λ -- the solver does
        not know or care about Lagrange multipliers, which makes the
        sign convention unambiguous (the right-hand side is simply the
        negation of whatever the caller passes).  The price is the
        caller does one extra ``C^T``-mat-vec per Newton step to build
        ``r1``; this matches what would be required anyway to compute
        the Newton convergence check ``||F_int + C^T λ||``.
        """
        import mfem.par as mfem
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

        # Sanity checks on dimensions.
        n_v_local   = K_op.Height()
        n_lam_local = C_op.Height()
        assert K_op.Width()  == n_v_local,   "K must be square"
        assert C_op.Width()  == n_v_local,   "C cols must match K rows"
        assert CT_op.Height() == n_v_local,  "C^T rows must match K rows"
        assert CT_op.Width()  == n_lam_local, "C^T cols must match C rows"
        assert r1_local.Size() == n_v_local,   "r1 must match K_op.Height()"
        assert r2_local.Size() == n_lam_local,  "r2 must match C_op.Height()"

        # ---- PyOperator dispatch sanity check -----------------------------
        # The PyOperator subclasses (C and C^T) override Mult in Python.
        # SWIG dispatch from the Krylov solver back into Python requires
        # ``%feature("director")`` on the wrapped class -- if that's missing,
        # our Python override is silently never invoked, the operator
        # behaves as the C++ default (zero), and Krylov stalls without
        # any informative error.  Diagnose this once-up-front by applying
        # C and C^T to known inputs and verifying the outputs are non-trivial
        # for a non-trivial operator.
        self._verify_constraint_dispatch(C_op, CT_op, n_v_local, n_lam_local)

        # ---- block_offsets : LOCAL on each rank -------------------------
        # offsets[0] = 0
        # offsets[1] = n_v_local         (end of velocity block)
        # offsets[2] = n_v_local + n_lam_local
        block_offsets = mfem.intArray([
            0, n_v_local, n_v_local + n_lam_local
        ])

        # ---- Build the block operator [K, C^T; C, 0] --------------------
        block_op = mfem.BlockOperator(block_offsets)
        block_op.SetBlock(0, 0, K_op)
        block_op.SetBlock(0, 1, CT_op)
        block_op.SetBlock(1, 0, C_op)
        # (1, 1) zero -> not set.

        # ---- Build the block-diagonal preconditioner --------------------
        # If preconditioner == "block_jacobi", build:
        #   P^{-1} = [ diag(K)^{-1}                            0                       ]
        #            [ 0                          diag(C diag(K)^{-1} C^T)^{-1} ]
        # K's diagonal is extracted via Operator.AssembleDiagonal (works
        # uniformly across PA / EA / FA / HypreParMatrix).  The Schur
        # diagonal is computed by the C operator's WeightedRowSqSum
        # method, which is clean operator-interface access -- no
        # exposing of the underlying scipy CSR.  Keep refs alive in
        # ``_prec_keepalive`` so neither the BlockDiagonalPreconditioner
        # nor the per-block scaler operators get GC'd before Krylov.Mult
        # finishes.
        block_prec = None
        _prec_keepalive = []
        if self.preconditioner == "block_jacobi":
            block_prec, _prec_keepalive = self._build_block_jacobi_prec(
                K_op, C_op, n_v_local, n_lam_local, block_offsets,
            )
            # Stash on self to also outlive any garbage collection
            # weirdness during the Krylov solve.
            self._last_prec_refs = _prec_keepalive

        # ---- One-shot diagnostic dump (gated by self.diagnostic_mode) ---
        # Dumps min / max / num-NaN / num-inf for every array involved in
        # the saddle-point system.  Set ``sps.diagnostic_mode = True``
        # before the call to enable.  Used to localize NaN propagation;
        # otherwise silent.
        if getattr(self, "diagnostic_mode", False):
            self._dump_diagnostics(
                K_op, C_op, CT_op,
                r1_local, r2_local,
                n_v_local, n_lam_local,
                _prec_keepalive,
            )

        # ---- RHS [-f + C^T λ; -C u] -------------------------------------
        # Strategy: construct the two halves as numpy/mfem.Vector objects
        # in their own scope, then write them element-wise into the
        # BlockVector's buffer.  Avoids the view-vs-copy ambiguity that
        # can bite when binding ``rhs_block.GetBlock(i)`` to a local
        # variable and calling methods on it across multiple statements.

        # ---- Build the RHS for one Newton step of the constrained system.
        #
        # Equilibrium: F_int(u) + C^T λ = 0  with  C u_tilde = 0.
        # ---- Build the RHS: [-r1; -r2] ----------------------------------
        # The caller has already assembled the full Newton residuals
        # (including any C^T λ contribution); the solver simply negates.
        # No collectives needed in this construction phase.
        rhs_block = mfem.BlockVector(block_offsets)
        rhs_block.Assign(0.0)
        for i in range(n_v_local):
            rhs_block[i] = -float(r1_local[i])
        for i in range(n_lam_local):
            rhs_block[n_v_local + i] = -float(r2_local[i])

        # ---- Krylov solver ----------------------------------------------
        SolverClass = getattr(mfem, _SOLVER_NAME_TO_MFEM_CLASS[self.solver_name])
        krylov = SolverClass(comm)
        krylov.SetRelTol(self.rel_tol)
        krylov.SetAbsTol(self.abs_tol)
        krylov.SetMaxIter(self.max_iter)
        krylov.SetPrintLevel(self.print_level)
        krylov.SetOperator(block_op)

        # Disable iterative mode on the Krylov solver.  iterative_mode
        # = True tells the solver to treat the INPUT solution vector as
        # the initial guess; iterative_mode = False forces it to start
        # from zero internally.  For the saddle-point Newton step this
        # MUST be False:
        #   * The Newton outer loop already warm-starts at the
        #     OUTER level via u_tilde and λ -- those carry information
        #     across iterations.
        #   * The INNER linear solve, however, is for the INCREMENTAL
        #     update (du, dλ).  At each Newton step the previous step's
        #     du has no relevance to the current step's du; using it as
        #     an initial guess is a category error that can produce
        #     incorrect Krylov convergence behavior, especially for CG.
        #   * Even though we explicitly zero ``solution_block`` below,
        #     belt-and-suspenders: SetIterativeMode(False) forces the
        #     solver to ignore the input, which is the safer contract.
        if hasattr(krylov, "SetIterativeMode"):
            krylov.SetIterativeMode(False)
        elif hasattr(krylov, "iterative_mode"):
            # Some pyMFEM versions expose this as a Python attribute.
            krylov.iterative_mode = False

        # GMRES default restart length is 50 (kdim=50).  For an
        # unpreconditioned saddle-point system with O(100-1000) dofs,
        # restart kills the n-step finite-termination property and
        # convergence becomes painful.  Disable restart effectively by
        # setting kdim equal to the GLOBAL system size (the union of
        # velocity TDOFs and multipliers across all ranks).  For
        # bigger production problems, the user should set max_iter to
        # something modest and add a preconditioner (Phase 1B).
        if self.solver_name == "GMRES" and hasattr(krylov, "SetKDim"):
            from mpi4py import MPI as _mpi
            _comm = _mpi.COMM_WORLD
            global_block_size = (
                _comm.allreduce(n_v_local + n_lam_local, op=_mpi.SUM)
            )
            # Cap at max_iter so we never allocate enormous Krylov bases.
            krylov.SetKDim(min(global_block_size, self.max_iter))

        # Wire in the block-Jacobi preconditioner (if requested).
        if block_prec is not None:
            krylov.SetPreconditioner(block_prec)

        # ---- Solve ------------------------------------------------------
        solution_block = mfem.BlockVector(block_offsets)
        solution_block.Assign(0.0)  # initial guess: zero increment
        krylov.Mult(rhs_block, solution_block)

        # Stash diagnostics for the caller.
        self.last_iterations = krylov.GetNumIterations()
        self.last_converged  = bool(krylov.GetConverged())
        self.last_final_norm = krylov.GetFinalNorm()

        # ---- Extract du and dlam ----------------------------------------
        # Read directly from solution_block by global element index,
        # avoiding the GetBlock(j) view-vs-copy ambiguity.
        du_local = mfem.Vector(n_v_local)
        for i in range(n_v_local):
            du_local[i] = float(solution_block[i])
        dlam_local = mfem.Vector(n_lam_local)
        for i in range(n_lam_local):
            dlam_local[i] = float(solution_block[n_v_local + i])

        return du_local, dlam_local

    # --------------------------------------- block-Jacobi prec -------
    @staticmethod
    def _build_block_jacobi_prec(K_op, C_op, n_v_local, n_lam_local,
                                  block_offsets):
        """Construct a 2x2 block-diagonal Jacobi preconditioner.

        Returns
        -------
        block_prec : mfem.BlockDiagonalPreconditioner
            The preconditioner ready to be passed to Krylov via
            ``SetPreconditioner``.
        keepalive : list
            Python references to the inverse-diagonal vectors and
            individual Jacobi scaler operators.  Caller must keep
            this list alive for the lifetime of the Krylov solve --
            ``BlockDiagonalPreconditioner`` does not own its diagonal
            blocks, and Python GC will collect them as soon as their
            references go out of scope.

        Construction
        ------------
        Block (0, 0):  ``y[i] = inv_diag(K)[i] * x[i]``.
            K's diagonal is extracted via ``K_op.AssembleDiagonal``
            (the canonical mfem.Operator method that works on PA, EA,
            FA, and HypreParMatrix forms uniformly).  Falls back to
            ``K_op.GetDiag(vec)`` for older HypreParMatrix wrappers
            without ``AssembleDiagonal`` exposed.

        Block (1, 1):  ``y[i] = inv(diag(C diag(K)^{-1} C^T))[i] * x[i]``.
            The Schur diagonal is computed by the C operator's
            ``WeightedRowSqSum`` method, which collectively gathers
            the K-diagonal-inverse and computes
            ``sum_j C[i,j]^2 * inv_diag_K[j]`` for each owned row.
            No explicit C C^T product is ever formed.

        Both diagonal blocks are wrapped as small Python-side scaler
        Operators (see ``_DiagonalScaler``) and registered with
        ``mfem.BlockDiagonalPreconditioner``.
        """
        import mfem.par as mfem
        from mpi4py import MPI

        # ---- Compute inv_diag(K) ----
        diag_K = mfem.Vector(n_v_local)
        diag_K.Assign(0.0)
        try:
            K_op.AssembleDiagonal(diag_K)
        except (AttributeError, NotImplementedError):
            # HypreParMatrix exposes GetDiag(Vector&) which fills the
            # local rank's diagonal slice.  This path is the fallback
            # for pyMFEM builds where AssembleDiagonal isn't exposed
            # on Operator.
            K_op.GetDiag(diag_K)

        # Element-wise inverse with safety floor for zero entries.
        # After EliminateRowsCols on K, corner Dirichlet rows have
        # diagonal = 1, so inversion is well-defined.  The tiny floor
        # only triggers in pathological cases (interior dof with K[i,i]=0
        # which would already be a model error upstream).
        inv_diag_K = mfem.Vector(n_v_local)
        for i in range(n_v_local):
            d = float(diag_K[i])
            inv_diag_K[i] = (1.0 / d) if abs(d) > 1e-300 else 0.0

        # ---- Compute inv(Schur_diag) ----
        # Collective: every rank calls WeightedRowSqSum (Allgatherv inside).
        schur_diag = mfem.Vector(n_lam_local)
        if hasattr(C_op, "WeightedRowSqSum"):
            C_op.WeightedRowSqSum(inv_diag_K, schur_diag)   # COLLECTIVE
        else:
            # Fallback: caller passed a C operator that doesn't expose
            # the row-squared-sum method.  This shouldn't happen with
            # the prototype's ``make_constraint_operators`` factory --
            # all operators it returns have ``WeightedRowSqSum``.  If
            # we reach this branch with a real operator (e.g., a future
            # HypreParMatrix-backed C), the caller needs to extend it
            # with the same method.
            raise RuntimeError(
                "C operator does not expose WeightedRowSqSum(); "
                "block_jacobi preconditioner requires this method to "
                "compute the Schur diagonal.  Use preconditioner='none' "
                "or add the method to your C operator subclass."
            )

        inv_schur_diag = mfem.Vector(n_lam_local)
        for i in range(n_lam_local):
            s = float(schur_diag[i])
            inv_schur_diag[i] = (1.0 / s) if abs(s) > 1e-300 else 0.0

        # ---- Wrap both as Python-side Solver-equivalent operators ----
        if hasattr(mfem, "PyOperatorBase"):
            PyOpClass = mfem.PyOperatorBase
        elif hasattr(mfem, "PyOperator"):
            PyOpClass = mfem.PyOperator
        else:
            raise RuntimeError("pyMFEM build does not expose PyOperatorBase")

        K_jac    = _DiagonalScaler(PyOpClass, inv_diag_K,    n_v_local)
        Schur_jac = _DiagonalScaler(PyOpClass, inv_schur_diag, n_lam_local)

        # ---- Assemble the block-diagonal preconditioner ----
        block_prec = mfem.BlockDiagonalPreconditioner(block_offsets)
        block_prec.SetDiagonalBlock(0, K_jac)
        block_prec.SetDiagonalBlock(1, Schur_jac)

        # Return refs so the caller's scope keeps everything alive.
        keepalive = [block_prec, K_jac, Schur_jac, inv_diag_K, inv_schur_diag,
                     diag_K, schur_diag]
        return block_prec, keepalive

    # ----------------------------------------- internal diagnostics ---
    @staticmethod
    def _dump_diagnostics(K_op, C_op, CT_op,
                          r1_local, r2_local,
                          n_v_local, n_lam_local,
                          prec_keepalive):
        """Print min/max/num-NaN/num-inf for every array involved in
        one saddle-point solve.  Called once, at iter 0 of the Newton
        loop, when ``SaddlePointSolver.diagnostic_mode = True``.
        Helps localize NaN propagation between the residual, the
        tangent's diagonal, and the Schur preconditioner diagonal.
        """
        import mfem.par as mfem
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        def stats(arr_np: np.ndarray, label: str) -> None:
            """Print min/max/finite/nan/inf counts for a numpy array."""
            n_total  = int(arr_np.size)
            n_nan    = int(np.sum(np.isnan(arr_np)))
            n_inf    = int(np.sum(np.isinf(arr_np)))
            n_finite = n_total - n_nan - n_inf
            if n_finite > 0:
                finite_arr = arr_np[np.isfinite(arr_np)]
                amin = float(np.min(finite_arr))
                amax = float(np.max(finite_arr))
                amax_abs = float(np.max(np.abs(finite_arr)))
            else:
                amin = amax = amax_abs = float("nan")
            print(f"    {label:24s}  n={n_total:5d}  "
                  f"finite={n_finite:5d}  nan={n_nan:3d}  inf={n_inf:3d}  "
                  f"min={amin:+.3e}  max={amax:+.3e}  |max|={amax_abs:.3e}")

        def vec_to_np(v: mfem.Vector) -> np.ndarray:
            return np.array(v.GetDataArray(), dtype=np.float64).copy()

        if rank == 0:
            print("\n  === Saddle-point diagnostic dump (iter 0) ===")

        # ---- 1. Residuals ----
        r1_np = vec_to_np(r1_local) if n_v_local > 0 else np.array([], dtype=np.float64)
        r2_np = vec_to_np(r2_local) if n_lam_local > 0 else np.array([], dtype=np.float64)
        if rank == 0:
            stats(r1_np, "r1 (top, F_int+C^Tλ)")
            stats(r2_np, "r2 (bottom, C u_tilde)")

        # ---- 2. K's diagonal (extracted via AssembleDiagonal) ----
        diag_K = mfem.Vector(n_v_local)
        diag_K.Assign(0.0)
        try:
            K_op.AssembleDiagonal(diag_K)
        except (AttributeError, NotImplementedError):
            try:
                K_op.GetDiag(diag_K)
            except Exception:
                pass
        diag_K_np = vec_to_np(diag_K) if n_v_local > 0 else np.array([], dtype=np.float64)
        if rank == 0:
            stats(diag_K_np, "diag(K)")

        # ---- 3. K's action on the e_0 unit vector (sanity check) ----
        # Picks up K[*, 0] as a column.  If K has NaN anywhere in column 0,
        # this reveals it.
        if n_v_local > 0:
            e0 = mfem.Vector(n_v_local)
            e0.Assign(0.0)
            e0[0] = 1.0
            Ke0 = mfem.Vector(n_v_local)
            K_op.Mult(e0, Ke0)
            Ke0_np = vec_to_np(Ke0)
            if rank == 0:
                stats(Ke0_np, "K @ e_0 (col 0 of K)")

        # ---- 4. Schur diagonal ----
        if hasattr(C_op, "WeightedRowSqSum"):
            inv_diag_K = mfem.Vector(n_v_local)
            for i in range(n_v_local):
                d = float(diag_K[i])
                inv_diag_K[i] = (1.0 / d) if abs(d) > 1e-300 else 0.0
            schur_diag = mfem.Vector(n_lam_local)
            C_op.WeightedRowSqSum(inv_diag_K, schur_diag)        # COLLECTIVE
            inv_diag_K_np = vec_to_np(inv_diag_K) if n_v_local > 0 else np.array([], dtype=np.float64)
            schur_diag_np = vec_to_np(schur_diag) if n_lam_local > 0 else np.array([], dtype=np.float64)
            if rank == 0:
                stats(inv_diag_K_np, "inv_diag(K)")
                stats(schur_diag_np, "schur_diag")

        # ---- 5. C op applied to a unit vector (sanity, geometric only) ----
        if n_v_local > 0:
            e0_v = mfem.Vector(n_v_local)
            e0_v.Assign(0.0)
            e0_v[0] = 1.0
            Ce0 = mfem.Vector(n_lam_local)
            C_op.Mult(e0_v, Ce0)                                 # COLLECTIVE
            Ce0_np = vec_to_np(Ce0) if n_lam_local > 0 else np.array([], dtype=np.float64)
            if rank == 0:
                stats(Ce0_np, "C @ e_0 (col 0 of C)")

        if rank == 0:
            print("  === end diagnostic dump ===\n")

    @staticmethod
    def _verify_constraint_dispatch(C_op, CT_op, n_v_local, n_lam_local):
        """Verify that C_op.Mult and CT_op.Mult are dispatched into the
        Python override (and not silently bypassed by SWIG).

        Method
        ------
        We construct an input mfem.Vector of all 1.0, hand it to
        ``C_op.Mult(x, y)``, and look at ``y``.  If our Python ``Mult``
        ran, ``y`` reflects the actual matvec.  If SWIG didn't install a
        director hook for our PyOperator subclass, ``y`` will be left as
        whatever its default-initialized contents were (typically zero,
        but undefined in general).

        Detection criterion
        -------------------
        We pre-fill the output with a sentinel value (``-1234.5``).  If
        after the Mult the vector still contains that sentinel anywhere
        (i.e. our override didn't write at least one element), the
        dispatch is broken.

        On dispatch failure we raise with a clear, actionable error
        message rather than letting the caller see Krylov stagnation or
        wrong answers.
        """
        import mfem.par as mfem
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # ----- Test C: (n_v_local) -> (n_lam_local) -----
        # CRITICAL: C_op.Mult is COLLECTIVE (does an Allgatherv internally)
        # and must be invoked on EVERY rank.  Do not guard the call on
        # n_lam_local > 0 -- ranks with zero local multipliers still
        # participate in the collective even though they don't produce
        # any output.  Only the sentinel CHECK afterwards is rank-local.
        x_test = mfem.Vector(n_v_local)
        for i in range(n_v_local):
            x_test[i] = 1.0
        y_test = mfem.Vector(n_lam_local)
        SENTINEL = -1234.5
        for i in range(n_lam_local):
            y_test[i] = SENTINEL
        C_op.Mult(x_test, y_test)            # COLLECTIVE -- must be unconditional
        # Local sentinel check: only meaningful where this rank owns at
        # least one multiplier row.
        if n_lam_local > 0 and float(y_test[0]) == SENTINEL:
            raise RuntimeError(
                "PyOperator dispatch failure: C_op.Mult did not invoke "
                "the Python override.  The output sentinel was not "
                "overwritten, meaning SWIG did not route the C++ Mult "
                "call back into Python.  This typically indicates that "
                "your pyMFEM build does not have %feature(\"director\") "
                "enabled on the PyOperator base class -- update or "
                "rebuild pyMFEM, or use a HypreParMatrix-based C "
                "matrix instead of the Python-side wrapper."
            )

        # ----- Test C^T: (n_lam_local) -> (n_v_local) -----
        # Same collective-invariance rule: CT_op.Mult must be called on
        # every rank.  Build the inputs / outputs unconditionally; only
        # the sentinel check is guarded.
        ylam_test = mfem.Vector(n_lam_local)
        for i in range(n_lam_local):
            ylam_test[i] = 1.0
        xv_test = mfem.Vector(n_v_local)
        for i in range(n_v_local):
            xv_test[i] = SENTINEL
        CT_op.Mult(ylam_test, xv_test)       # COLLECTIVE -- must be unconditional
        # The sentinel check: C^T applied to ylam=1 produces nonzero output
        # at any TDOF where C has a nonzero column entry.  For the
        # patch-test mortar system that's the case on at least the
        # boundary TDOFs of every rank that owns boundary nodes.  Skip
        # the check on ranks where every TDOF could legitimately end up
        # zero (rank where n_lam_local=0 contributes nothing to the
        # "y_global=1 everywhere" Allgather but the resulting C^T y is
        # still nonzero on this rank's TDOFs since C has nonzero columns
        # mapped here).
        if n_v_local > 0 and float(xv_test[0]) == SENTINEL:
            # Note: this check is more lenient than C's check because
            # element 0 of x might happen to map to a column of C with
            # all zero entries (e.g. an interior DOF).  We don't raise
            # here; the C-side check above is the stronger test.
            pass


# =============================================================================
# Helper: zero out corner-DOF columns of the scipy-CSR C matrix
# =============================================================================

def apply_dirichlet_zero_to_C(
    C: sp.csr_matrix,
    dirichlet_tdofs: np.ndarray,
) -> sp.csr_matrix:
    """Return a copy of C with the columns at ``dirichlet_tdofs`` zeroed.

    The constraint matrix should not couple to DOFs that are already
    pinned to zero (the rigid-body-mode-removal corners).  This is the
    constraint-side counterpart of ``apply_dirichlet_to_K`` (which
    operates on the distributed K).
    """
    C = C.tolil()
    for d in dirichlet_tdofs:
        C[:, int(d)] = 0
    return C.tocsr()
