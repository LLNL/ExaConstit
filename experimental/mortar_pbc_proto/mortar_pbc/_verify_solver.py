"""Quarantined SciPy direct solver -- verification path only.

WHAT
----
A serial, gather-to-rank-0 direct LU solver for the saddle-point system.
Used ONLY to cross-check the distributed Krylov path
(``mortar_pbc.saddle_point.SaddlePointSolver``) on small patch-test
problems.  Not exported from the package's public API.

WHY (rationale for keeping it at all)
-------------------------------------
When the Krylov path produces a slightly off answer on a new problem
(different mesh, different material, different F_macro), having a
reference "ground truth" answer makes triage tractable: if both solvers
produce the same wrong answer, the bug is upstream of the solver
(constraint matrix, residual, Dirichlet handling); if only Krylov is
off, the bug is in the Krylov setup (preconditioner, tolerances,
operator wrapping).  The serial reference is a debugging tool, not a
production path.

WHY this file is underscore-prefixed and not in __init__.py
------------------------------------------------------------
To prevent it from being used inadvertently in production-ish code.
The blessed solver is ``mortar_pbc.saddle_point.SaddlePointSolver``.
This file should be imported only by:
    * the patch-test driver (cross-check path),
    * future debugging scripts that explicitly want a reference answer.

Limitations (intentional)
-------------------------
    * Single-rank only -- gathers to rank 0 and returns ``None`` on others.
    * Materializes K as scipy CSR -- assumes K is a HypreParMatrix or
      something that can be turned into one.
    * O(n^3) factorization cost (LU); fine for ~10^3 dofs, terrible
      beyond.
    * No preconditioning, no iterative refinement.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class SciPyDirectSolver:
    """Direct LU solve of the gathered saddle-point system on rank 0.

    Returns the SAME (du, dlam) interface as ``SaddlePointSolver`` but
    operates on scipy CSR / numpy arrays gathered to rank 0.  Returns
    ``None`` on non-root ranks for both pieces.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def solve_step(
        self,
        K: sp.csr_matrix,
        C: sp.csr_matrix,
        r1: np.ndarray,
        r2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve [[K, C^T], [C, 0]] [du; dlam] = [-r1; -r2].

        All inputs are numpy / scipy on rank 0; solve happens on rank 0.
        Caller is responsible for the gather/scatter.

        Caller assembles the FULL Newton residuals and passes them in
        directly:
            r1 = F_int(u) + C^T λ          (top, force-balance residual)
            r2 = C u - g                   (bottom, constraint residual)
        The solver simply negates them to form the right-hand side.
        This matches the production ``SaddlePointSolver.solve_step``
        API (refactored to take pre-assembled residuals to eliminate
        the sign-bug class).
        """
        n_dofs    = K.shape[0]
        n_constrs = C.shape[0]
        assert r1.size == n_dofs,    "r1 must match K.shape[0]"
        assert r2.size == n_constrs, "r2 must match C.shape[0]"

        # Saddle-point block matrix.
        zero_block = sp.csr_matrix((n_constrs, n_constrs))
        block_top = sp.hstack([K, C.T],          format="csr")
        block_bot = sp.hstack([C, zero_block],    format="csr")
        saddle_matrix = sp.vstack([block_top, block_bot], format="csr")

        # RHS = [-r1; -r2].
        rhs = np.zeros(n_dofs + n_constrs)
        rhs[:n_dofs] = -r1
        rhs[n_dofs:] = -r2

        if self.verbose:
            r1_norm = float(np.linalg.norm(r1))
            r2_norm = float(np.linalg.norm(r2))
            print(f"[Verify] K: {K.shape}, C: {C.shape}, "
                  f"|r1|={r1_norm:.3e}, |r2|={r2_norm:.3e}")

        solution = spla.spsolve(saddle_matrix.tocsc(), rhs)
        du   = solution[:n_dofs]
        dlam = solution[n_dofs:]
        return du, dlam
