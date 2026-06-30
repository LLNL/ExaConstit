"""Multi-step mortar-PBC driver with ExaConstit-style warm-start.

Provides a thin wrapper around the saddle-point solve that:

  * tracks state across load increments (``u``, ``lambda``, ``F_macro``);
  * builds a warm-start initial iterate when going from step n to step
    n+1, using ExaConstit's ``SystemDriver::SolveInit`` recipe adapted
    to the saddle-point structure;
  * records solve statistics for downstream reporting.

ExaConstit's recipe (verbatim, translated to displacement primal +
saddle-point):

    Step 1 (warm-start projection, before the actual solve):
      1a. K_n   := tangent stiffness at the previously converged state.
                   For linear elasticity this is a constant K
                   (independent of u); for nonlinear materials it
                   comes from ``nlf.GetGradient(u_n)``.
      1b. Build ``deltaF`` of size n_tdof, zeroed everywhere except at
          essential DOFs (the 4 corners), where
              deltaF[corner] = u_macro_{n+1}[corner] - u_macro_n[corner]
          i.e. the change in prescribed corner displacement.
      1c. Compute  K_full @ deltaF  (action of the FULL tangent, before
          essential-DOF elimination, on the deltaF vector).  This is
          the change in residual at FREE DOFs caused by the change in
          essential-DOF prescribed values.  Call this "b".
      1d. Compute the residual at the previous-converged state
          (``R^n = F_int(u_n) + C^T lambda_n - f_ext``).  At
          convergence of step n this is zero on free DOFs and zero on
          essential DOFs (the latter because the BC was satisfied
          exactly).  We add it back in case step n didn't fully
          converge -- this picks up any leftover imbalance.
      1e. Solve the ELIMINATED system
              K_eliminated @ delta_u_solve  +  C^T @ delta_lam = -b
              C @ delta_u_solve                                = -(C @ deltaF)
          for delta_u_solve.  Note the saddle-point structure: this is
          the same linear system shape as the actual nonlinear step.
      1f. Initial guess for the next solve:
              u_initial   = u_n + deltaF + delta_u_solve
              lam_initial = lambda_n + delta_lam

    Step 2 (the main solve, as normal):
      2a. Apply u_macro_{n+1}[corner] EXACTLY at the essential corners.
      2b. Run the saddle-point solve from u_initial.

For linear elasticity, where K is constant and the problem is linear,
the warm-start completely solves the next step in one shot
(delta_u_solve at step 2 lands at machine precision if step 1 was
exact).  The benefit shows up most when the integrator is nonlinear:
the warm-start starts Newton inside the basin of convergence.

Volume-averaged deformation gradient diagnostic
-----------------------------------------------
``compute_volume_averaged_F(pmesh, fes, u)`` returns the volume-
averaged total deformation gradient

    <F> = (1/V) ∫_Ω F dΩ = I + (1/V) ∫_Ω ∇u dΩ

via Gauss quadrature on each element.  By the homogenization average
theorem, on a periodic RVE under macroscopic F_macro,

    <F> = F_macro

to machine precision -- regardless of internal heterogeneity.  This
is THE consistency check for any computational homogenization driver:
if ``<F>`` differs from the prescribed F_macro by more than a few
ulps, something is wrong with the mortar constraint, the corner
Dirichlet, or the post-processing of the displacement field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mfem.par as mfem
from mpi4py import MPI


# ---------------------------------------------------------------------------
# Volume-averaged deformation gradient
# ---------------------------------------------------------------------------

def compute_volume_averaged_F(
    pmesh: mfem.ParMesh,
    fes: mfem.ParFiniteElementSpace,
    u_par: mfem.Vector,
) -> np.ndarray:
    """Compute <F> = (1/V) ∫_Ω F dΩ over the parallel mesh.

    Uses element-level Gauss quadrature with the rule appropriate for
    the FE order (``2*order + 1``).  Returns a (dim, dim) numpy array
    valid on every rank (Allreduce).

    Notes
    -----
    For an H1 vector grid function representing displacement u(X),
    the deformation gradient is F(X) = I + ∇u(X), and the average is

        <F> = I + (1/V) ∫_Ω ∇u dΩ

    By the homogenization average theorem (Hill-Mandel), for a periodic
    RVE under macroscopic F_macro applied via the additive
    decomposition u = (F_macro - I) X + ũ, ``<F>`` should equal
    ``F_macro`` exactly (because ∫ ∇ũ dΩ = ∮ ũ ⊗ n dΓ = 0 by
    periodicity of ũ and antisymmetric outward normals on opposite
    faces).  Hence this is a clean consistency check for the PBC
    implementation.
    """
    comm = pmesh.GetComm() if hasattr(pmesh, "GetComm") else MPI.COMM_WORLD
    dim = pmesh.Dimension()

    # Build a ParGridFunction holding u so we can call GetVectorGradient.
    gf_u = mfem.ParGridFunction(fes)
    gf_u.SetFromTrueDofs(u_par)

    # Accumulate ∫ ∇u dΩ and ∫ 1 dΩ over local elements.
    grad_u_acc = np.zeros((dim, dim), dtype=np.float64)
    vol_acc    = 0.0

    grad_u_at_qp = mfem.DenseMatrix(dim, dim)

    for e in range(pmesh.GetNE()):
        fe = fes.GetFE(e)
        eltrans = fes.GetElementTransformation(e)
        order = 2 * fe.GetOrder() + 1
        ir = mfem.IntRules.Get(fe.GetGeomType(), order)

        for q in range(ir.GetNPoints()):
            ip = ir.IntPoint(q)
            eltrans.SetIntPoint(ip)
            w = ip.weight * eltrans.Weight()        # quadrature weight * |J|
            # GetVectorGradient writes ∂u_i/∂x_j into grad_u_at_qp[i, j]
            gf_u.GetVectorGradient(eltrans, grad_u_at_qp)
            for i in range(dim):
                for j in range(dim):
                    grad_u_acc[i, j] += w * float(grad_u_at_qp[i, j])
            vol_acc += w

    # Allreduce: sum local contributions across ranks.
    grad_u_global_flat = np.zeros(dim * dim, dtype=np.float64)
    comm.Allreduce(grad_u_acc.flatten(), grad_u_global_flat, op=MPI.SUM)
    vol_global = comm.allreduce(vol_acc, op=MPI.SUM)

    grad_u_global = grad_u_global_flat.reshape((dim, dim))
    F_avg = np.eye(dim, dtype=np.float64) + grad_u_global / vol_global
    return F_avg


# ---------------------------------------------------------------------------
# Multi-step mortar-PBC driver
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Per-step record of solver statistics."""
    step: int
    F_macro: np.ndarray
    krylov_iters: int
    krylov_converged: bool
    krylov_final_norm: float
    u_inf: float
    u_tilde_inf: float
    constraint_residual: float
    F_average: np.ndarray
    F_average_error: float        # ||F_average - F_macro||_max


class MortarPbcDriver2D:
    """Multi-step mortar-PBC driver for linear-elastic RVEs.

    Owns the persistent state needed for ExaConstit-style warm-start:

      * ``self.u_par``       : the converged total displacement u_n.
      * ``self.lam_par``     : the converged Lagrange multipliers λ_n.
      * ``self.F_prev``      : the macroscopic F at step n.
      * ``self.history``     : list of ``StepResult`` records.

    The driver does NOT own the FE space or mesh -- those are passed in
    once at construction and held by reference.  The driver does own the
    pre-eliminated K (since step-to-step K is unchanged for linear
    elasticity, we can assemble it once); for nonlinear materials this
    will need to be re-assembled per step.

    Workflow
    --------
    Construction
        driver = MortarPbcDriver2D(
            pmesh=..., fes=..., K_op=..., C_op=..., CT_op=...,
            corner_tdofs=..., apply_dirichlet_to_K=..., sps=...,
            apply_linear_part=..., n_lam_local=...,
        )

    Step 1 (first call)
        result = driver.solve_first_step(F_macro_1)

    Step 2+  (subsequent calls)
        result = driver.solve_next_step(F_macro_2)

    Each call returns a ``StepResult`` and updates ``driver.history``.

    Implementation notes
    --------------------
    The signatures are intentionally pyMFEM-style (passing operators and
    helper callables, not abstract interfaces) so the driver can be
    transplanted into the eventual ExaConstit C++ port with minimal
    re-architecture.  Functions like ``apply_dirichlet_to_K`` and
    ``apply_linear_part`` are passed as callables to keep the driver
    decoupled from the example-driver scaffolding (those helpers live
    in the patch-test scripts because they're MFEM-version-specific).
    """

    def __init__(
        self,
        *,
        pmesh: mfem.ParMesh,
        fes: mfem.ParFiniteElementSpace,
        K_op,                              # mfem.HypreParMatrix (eliminated)
        K_op_full,                         # mfem.HypreParMatrix (NOT eliminated)
        C_op,
        CT_op,
        corner_tdofs: np.ndarray,
        apply_linear_part_fn,              # callable: (fes, F_macro) -> np.ndarray
        numpy_to_mfem_vector_fn,           # callable: (np.ndarray) -> mfem.Vector
        sps,                               # SaddlePointSolver
        n_lam_local: int,
        local_corner_tdofs: list,          # local indices into per-rank vectors
    ) -> None:
        self.pmesh = pmesh
        self.fes   = fes
        self.K_op       = K_op
        self.K_op_full  = K_op_full
        self.C_op       = C_op
        self.CT_op      = CT_op
        self.corner_tdofs       = np.asarray(corner_tdofs, dtype=np.int64)
        self.apply_linear_part  = apply_linear_part_fn
        self.numpy_to_mfem_vec  = numpy_to_mfem_vector_fn
        self.sps = sps
        self.n_lam_local = n_lam_local
        self.local_corner_tdofs = list(local_corner_tdofs)

        # Persistent state across steps.
        self.u_par:     Optional[mfem.Vector] = None
        self.lam_par:   Optional[mfem.Vector] = None
        self.F_prev:    Optional[np.ndarray]  = None
        self.history:   list[StepResult]     = []

        self._comm = pmesh.GetComm() if hasattr(pmesh, "GetComm") else MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._my_n_tdof = fes.GetTrueVSize()

    # ------------------------------------------------------------------ API

    def solve_first_step(self, F_macro: np.ndarray) -> StepResult:
        """Solve the first load step.

        Method-D + linear-elastic Lopes 2021 Remark 1: the linear
        displacement part is applied to the entire RVE domain in the
        first stage as an initial guess.  We solve the saddle-point
        system

            [K_e   C^T] [du ]   [-K_full @ u_lin]   (corner entries
            [C      0 ] [dlam] = [    0          ]    of top zeroed)

        for ``du = u_tilde``, then form ``u = u_lin + du``.  ``K_full``
        (un-eliminated) is used on the RHS so the K_uc block
        contribution from the corners is retained; ``K_e``
        (eliminated) is used as the saddle-point top block so the
        corner BC is enforced via diagonal-1 rows.

        For homogeneous material under uniform F, du is identically
        zero (machine precision); for heterogeneous material it is
        the non-trivial fluctuation.
        """
        result = self._solve_independently(F_macro)
        result.step = 1
        self.history.append(result)
        return result

    def solve_next_step(self, F_macro_next: np.ndarray) -> StepResult:
        """Solve the next load step.

        For LINEAR ELASTICITY -- which is what this prototype validates
        until pyMFEM's NeoHookean integrator is fixed -- each step is
        completely independent of the prior state.  The "warm-start
        projection" loop from ExaConstit's ``SystemDriver::SolveInit``
        becomes degenerate: the projection itself solves the linear
        system exactly, so there is nothing left for Newton to do.
        We therefore implement ``solve_next_step`` as a re-invocation
        of ``solve_first_step`` with the new F_macro.  The driver
        still:
            * tracks the converged ``u``, ``lambda``, ``F_macro``
              across calls (visible via ``self.u_par`` etc.);
            * records each step in ``self.history`` for downstream
              reporting;
            * computes the volume-averaged-F homogenization
              consistency check at every step.

        For NONLINEAR materials (when the integrator is fixed), this
        method must be re-implemented to:
            1. Build deltaF = (u_lin_next - u_par_prev) at corners,
               zero elsewhere.
            2. Compute b = K_n @ deltaF using the previous-state
               tangent.
            3. Add R^n (residual at u_par_prev), normally zero at
               step-n convergence.
            4. Solve [K, C^T; C, 0] [Δv; Δλ] = [-b; -C deltaF] for
               Δv, Δλ.
            5. Set u_initial = u_par_prev + deltaF + Δv as Newton's
               initial iterate.
            6. Run Newton to convergence from u_initial.

        See ExaConstit's ``SystemDriver::SolveInit`` and
        ``NonlinearMechOperator::GetUpdateBCsAction`` for the
        canonical implementation.  The architectural skeleton in
        :class:`MortarPbcDriver2D` is set up to make the nonlinear
        extension a focused change to this method only.
        """
        if self.u_par is None or self.F_prev is None:
            raise RuntimeError(
                "solve_next_step called before solve_first_step; "
                "the driver has no previous state to warm-start from."
            )

        # Linear-elastic placeholder: solve fresh, then advance state.
        # Save current step number (history.append in solve_first_step
        # would otherwise re-tag this as step 1).
        result = self._solve_independently(F_macro_next)
        result.step = len(self.history) + 1
        self.history.append(result)
        return result

    def _solve_independently(self, F_macro: np.ndarray) -> StepResult:
        """Same solve as ``solve_first_step`` but doesn't touch
        ``self.history`` -- caller is responsible for appending.

        RHS construction
        ----------------
        The Newton residual for "u = u_lin satisfies equilibrium with
        corner BC" is

            r1 = F_int(u_lin) = K_full @ u_lin   (linear elastic)

        evaluated with the FULL (un-eliminated) tangent.  This includes
        the K_uc @ u_lin[corner] coupling at free rows -- crucial for
        correctness, because for homogeneous material under affine BC
        the affine field IS the equilibrium, so K_full @ u_lin = 0 at
        free rows (K_uu @ u_lin[free] + K_uc @ u_lin[corner] = 0).

        Using ``K_eliminated @ u_lin`` instead would give
        K_uu @ u_lin[free] only (K_uc column zeroed by elimination),
        which is NOT zero even for homogeneous material -- the solver
        would then compute a spurious ``du`` to "correct" a residual
        that physically isn't there, giving the WRONG sign of
        free-DOF displacement.  The prior single-step working code
        avoided this by computing K @ u_lin BEFORE applying the
        elimination to K; in the multi-step driver K arrives already
        eliminated, so we must use K_full for the RHS computation.
        """
        u_lin_local = self.apply_linear_part(self.fes, F_macro)
        u_lin_par   = self.numpy_to_mfem_vec(u_lin_local)

        # f = K_full @ u_lin  (NOT K_eliminated -- see docstring).
        # Then zero corner entries: the saddle-point top block uses the
        # ELIMINATED K which has identity rows at corners, so a zero
        # corner RHS produces du[corner] = 0 (the essential BC).
        f_par = mfem.Vector(self._my_n_tdof)
        self.K_op_full.Mult(u_lin_par, f_par)
        for local_idx in self.local_corner_tdofs:
            f_par[local_idx] = 0.0

        # Constraint RHS r2 = 0 (Method-C reading: solving for the
        # fluctuation u_tilde = du with C @ u_tilde = 0).
        r2_par = mfem.Vector(self.n_lam_local)
        r2_par.Assign(0.0)

        du_par, dlam_par = self.sps.solve_step(
            K_op=self.K_op, C_op=self.C_op, CT_op=self.CT_op,
            r1_local=f_par, r2_local=r2_par,
        )

        u_par = mfem.Vector(self._my_n_tdof)
        for i in range(self._my_n_tdof):
            u_par[i] = float(u_lin_par[i]) + float(du_par[i])
        lam_par = mfem.Vector(self.n_lam_local)
        for i in range(self.n_lam_local):
            lam_par[i] = float(dlam_par[i])

        result = self._make_step_result(
            step=0, F_macro=F_macro,             # caller will set step
            u_par=u_par, du_par=du_par, u_lin_par=u_lin_par,
        )
        self._update_state(u_par=u_par, lam_par=lam_par, F_macro=F_macro)
        return result

    # --------------------------------------------------------------- private

    def _update_state(self, u_par: mfem.Vector, lam_par: mfem.Vector,
                       F_macro: np.ndarray) -> None:
        # Replace persistent state (clone vectors so the caller can't
        # mutate driver state from outside).
        self.u_par = mfem.Vector(self._my_n_tdof)
        for i in range(self._my_n_tdof):
            self.u_par[i] = float(u_par[i])
        self.lam_par = mfem.Vector(self.n_lam_local)
        for i in range(self.n_lam_local):
            self.lam_par[i] = float(lam_par[i])
        self.F_prev = np.array(F_macro, dtype=np.float64, copy=True)

    def _make_step_result(self, *, step: int, F_macro: np.ndarray,
                           u_par: mfem.Vector, du_par: mfem.Vector,
                           u_lin_par: mfem.Vector) -> StepResult:
        comm = self._comm

        # Norms (Allreduce-summed across ranks).
        local_u_sq        = sum(float(u_par[i])**2 for i in range(self._my_n_tdof))
        local_du_sq       = sum(float(du_par[i])**2 for i in range(self._my_n_tdof))
        local_u_inf       = max((abs(float(u_par[i])) for i in range(self._my_n_tdof)),
                                 default=0.0)
        local_du_inf      = max((abs(float(du_par[i])) for i in range(self._my_n_tdof)),
                                 default=0.0)
        u_inf       = comm.allreduce(local_u_inf, op=MPI.MAX)
        u_tilde_inf = comm.allreduce(local_du_inf, op=MPI.MAX)

        # Constraint residual ||C u_tilde||_2 = ||C du||_2.  The C_op
        # delivers all rows on rank 0 in our current parallel layout.
        Cu_par = mfem.Vector(self.n_lam_local)
        self.C_op.Mult(du_par, Cu_par)
        local_Cu_sq = sum(float(Cu_par[i])**2 for i in range(self.n_lam_local))
        global_Cu_sq = comm.allreduce(local_Cu_sq, op=MPI.SUM)
        constraint_residual = float(np.sqrt(global_Cu_sq))

        # Volume-averaged F and its error vs F_macro.
        F_average = compute_volume_averaged_F(self.pmesh, self.fes, u_par)
        F_average_error = float(np.max(np.abs(F_average - F_macro)))

        return StepResult(
            step=step,
            F_macro=np.array(F_macro, dtype=np.float64, copy=True),
            krylov_iters=int(self.sps.last_iterations),
            krylov_converged=bool(self.sps.last_converged),
            krylov_final_norm=float(self.sps.last_final_norm),
            u_inf=float(u_inf),
            u_tilde_inf=float(u_tilde_inf),
            constraint_residual=constraint_residual,
            F_average=F_average,
            F_average_error=F_average_error,
        )
