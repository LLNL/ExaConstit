// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other ExaConstit Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: MIT

#include "solvers/trust_region_solver.hpp"

#include "utilities/mechanics_log.hpp"
#include "utilities/unified_logger.hpp"

#include "mfem.hpp"
#include "mfem/general/globals.hpp"
#include "mfem/linalg/linalg.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

/**
 * @brief Compute the Powell dogleg step inside the trust region.
 *
 * @details Step-by-step algorithm:
 *
 * 1. **Full Newton step inside trust region**:
 *    If ||s_N|| <= delta, take the full Newton step. The predicted residual
 *    is zero (the linear model F + J*s_N = 0 is exactly satisfied).
 *
 * 2. **Cauchy point outside trust region**:
 *    Compute the Cauchy point parameters:
 *       - alpha = ||g||^2 / ||J*g||^2   (optimal scaling along steepest descent)
 *       - ||s_sd_opt|| = alpha * ||g||  (norm of the optimal Cauchy step)
 *    If ||s_sd_opt|| >= delta, the optimal Cauchy point is outside the trust
 *    region. Step along the steepest descent direction to the boundary:
 *       delx = -delta * g / ||g||
 *    The predicted residual norm is computed from the linear model evaluated
 *    at this truncated Cauchy step.
 *
 * 3. **Dogleg interpolation (second leg)**:
 *    Otherwise, interpolate along the line segment from the Cauchy point to
 *    the Newton point, finding the parameter beta in [0, 1] such that the
 *    interpolated step lies on the trust-region boundary. The intersection
 *    is found by solving a quadratic:
 *       delx(beta) = beta * s_N - (1 - beta) * alpha * g
 *       ||delx(beta)||^2 = delta^2
 *    yielding qa*beta^2 - 2*qb*beta + qc = 0 where:
 *       qa = ||p||^2,   qb = alpha * (p . g),   qc = ||s_sd_opt||^2 - delta^2
 *       and p = s_N + alpha * g.
 *    Beta is taken from the larger root and clamped to [0, 1] for safety.
 */
void ExaTrustRegionSolver::Dogleg(double delta, double res_0, double nr_norm,
                                  double Jg_2, const mfem::Vector &grad,
                                  const mfem::Vector &nrStep, mfem::Vector &delx,
                                  double &pred_resid, bool &use_nr) const
{
   use_nr = false;

   // --- Case 1: Full Newton step fits inside the trust region ---
   if (nr_norm <= delta) {
      use_nr = true;
      delx = nrStep;
      pred_resid = 0.0;

      if (print_level > 0) {
         mfem::out << "TR dogleg: taking full Newton step (||s_N|| = "
                   << nr_norm << " <= delta = " << delta << ")\n";
      }
      return;
   }

   // Cauchy point parameters using MPI-aware dot products
   const double norm2_grad = Dot(grad, grad);
   const double norm_grad = std::sqrt(norm2_grad);

   const double alpha = (Jg_2 > 0.0) ? (norm2_grad / Jg_2) : 1.0;
   const double norm_grad_inv = (norm_grad > 0.0) ? (1.0 / norm_grad) : 1.0;
   const double norm_s_sd_opt = alpha * norm_grad;

   // --- Case 2: Cauchy point is outside the trust region ---
   // Take a step along the steepest descent direction to the trust-region boundary
   if (norm_s_sd_opt >= delta) {
      // delx = -delta * (grad / ||grad||)
      const double factor = -delta * norm_grad_inv;
      delx = grad;
      delx *= factor;

      // Predicted residual from linear model at the truncated Cauchy step
      const double val = -(delta * norm_grad) +
                         0.5 * delta * delta * Jg_2 *
                         (norm_grad_inv * norm_grad_inv);
      pred_resid = std::sqrt(std::max(2.0 * val + res_0 * res_0, 0.0));

      if (print_level > 0) {
         mfem::out << "TR dogleg: stepping along first leg (steepest descent)\n";
      }
   }
   // --- Case 3: Cauchy inside, Newton outside; interpolate along the second leg ---
   else {
      // Reuse delx as workspace for p = nrStep + alpha * grad
      mfem::Vector &p = delx;
      add(nrStep, alpha, grad, p);

      // Quadratic coefficients for the trust-region boundary intersection
      double qa = Dot(p, p);
      double qb = Dot(p, grad) * alpha;
      double qc = norm_s_sd_opt * norm_s_sd_opt - delta * delta;

      double discriminant = qb * qb - qa * qc;
      double beta = (qa > 0.0)
         ? (qb + std::sqrt(std::max(discriminant, 0.0))) / qa
         : 0.0;

      // Clamp beta to [0, 1] to handle any roundoff at the boundary
      beta = std::max(0.0, std::min(1.0, beta));

      // delx = beta * nrStep - (1 - beta) * alpha * grad
      const double omb = 1.0 - beta;
      const double omba = omb * alpha;
      add(beta, nrStep, -omba, grad, delx);

      // Predicted residual from linear model at the dogleg step
      const double res_cauchy = (Jg_2 > 0.0)
         ? std::sqrt(std::max(res_0 * res_0 - alpha * norm2_grad, 0.0))
         : res_0;
      pred_resid = omb * res_cauchy;

      if (print_level > 0) {
         mfem::out << "TR dogleg: stepping along second leg (beta = "
                   << beta << ")\n";
      }
   }
}

/**
 * @brief Trust-region dogleg Newton iteration implementation.
 *
 * @details Step-by-step algorithm for solving F(x) = b:
 *
 * **Initial setup**:
 *   1. Validate that operator (oper_mech), preconditioner (prec_mech), and
 *      delta_ctrl are properly configured
 *   2. Allocate all device-aware working vectors (nrStep, grad, delx, Jg_temp,
 *      x_prev) once before the iteration loop
 *   3. Evaluate initial residual r = F(x) - b and compute its norm
 *   4. Set the convergence threshold norm_max = max(rel_tol * res, abs_tol)
 *   5. Initialize trust-region radius delta from delta_ctrl.deltaInit
 *
 * **Main iteration loop** (until convergence or max_iter):
 *   1. If the previous step was *not* rejected, recompute Newton machinery:
 *      a. Get Jacobian J = oper_mech->GetGradient(x). The material state is
 *         consistent with x because Mult(x, r) was just evaluated.
 *      b. Compute steepest descent: grad = J^T * r (gradient of f = 0.5 ||F||^2)
 *      c. Compute Jg_2 = ||J * grad||^2 for the optimal Cauchy step length
 *      d. Solve the Newton system J*c = r via the Krylov solver (prec_mech),
 *         then negate: nrStep = -c. The negation matches SNLS convention where
 *         the Newton update is x += nrStep (whereas ExaNewtonSolver uses x -= c).
 *      e. Compute nr_norm = ||nrStep||
 *      If the previous step *was* rejected, all of this data is still valid
 *      from the last accepted iteration and we just recompute the dogleg with
 *      the smaller delta.
 *   2. Save x_prev = x for potential rollback on rejection
 *   3. Compute the dogleg step delx via Dogleg() helper
 *   4. Apply the trial step: x = x_prev + delx
 *   5. Evaluate residual at the trial point: r = F(x) - b
 *   6. Check convergence: if ||r|| <= norm_max, accept and exit
 *   7. Update delta via delta_ctrl.UpdateDelta() based on actual vs predicted
 *      reduction. This may also flag the step for rejection.
 *   8. If rejected: restore x = x_prev, restore residual norm, set reject_prev.
 *      The material state inside the model handles itself analogously to the
 *      ExaNewtonLSSolver line-search behavior — when Mult() is called again at
 *      the next trial point, the model recomputes from the beginning-step state.
 *
 * **Performance Profiling**:
 *   - "TR_dogleg_solver" scope for overall trust-region solver performance
 *   - "TR_newton_setup" scope for J^T*r and J*g computations
 *   - "TR_gradient_transpose" scope for the J^T*r call specifically
 *   - "TR_newton_solve" scope for the Krylov inner solve
 *   - "TR_trial_eval" scope for residual evaluations at trial points
 *   - "krylov_solver" scope for the actual Krylov solver call
 *
 * @note All scalar quantities (norms, dot products) use MFEM's MPI-aware
 *       Norm() and Dot() functions through the IterativeSolver base class
 */
void ExaTrustRegionSolver::Mult(const mfem::Vector &b, mfem::Vector &x) const
{
   CALI_CXX_MARK_SCOPE("TR_dogleg_solver");
   MFEM_ASSERT_0(oper_mech, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT_0(prec_mech, "the Solver is not set (use SetSolver).");
   MFEM_ASSERT(delta_ctrl.Validate(),
               "TrDeltaControl parameters are invalid.");

   const bool have_b = (b.Size() == Height());

   // Phase 5.11.G — cache the scaler-enabled flag once per Mult so
   // the per-iter scaling branches don't keep dereferencing the
   // shared_ptr. The IsEnabled() check is cheap but the indirection
   // is unnecessary inside the inner loop.
   const bool scaler_active = (m_scaler && m_scaler->IsEnabled());

   // --- Allocate working vectors once, reused across iterations ---
   mfem::Vector nrStep(width, mfem::Device::GetMemoryType());
   mfem::Vector grad(width, mfem::Device::GetMemoryType());
   mfem::Vector delx(width, mfem::Device::GetMemoryType());
   mfem::Vector Jg_temp(width, mfem::Device::GetMemoryType());
   mfem::Vector x_prev(width, mfem::Device::GetMemoryType());

   nrStep.UseDevice(true);
   grad.UseDevice(true);
   delx.UseDevice(true);
   Jg_temp.UseDevice(true);
   x_prev.UseDevice(true);

   // Match ExaNewtonSolver / ExaNewtonLSSolver semantics: in
   // non-iterative mode the caller is asking for a fresh solve, so
   // ignore any incoming iterate and start from zero.
   if (!iterative_mode) {
      x = 0.0;
   }

   // --- Initial residual evaluation: r = F(x) - b ---
   // When scaler_active, oper_mech is the 5.11.D ScaledSaddleOperator
   // wrapper, so r holds r_solver (scaled) from this point onward.
   oper_mech->Mult(x, r);
   if (have_b) { r -= b; }

   // Phase 5.11.G — capture the initial residual for the relative
   // convergence test. Stays constant through the loop; distinct
   // from res_0 (which tracks the previous-iter residual for
   // rejection rollback).
   const double res_initial = Norm(r);
   double res = res_initial;
   double res_0 = res;

   // Phase 5.11.G — derived legacy threshold kept only for the
   // diagnostic sink and the existing logging output. The actual
   // convergence test below evaluates the two conditions
   // independently (SNLS-style).
   const double norm_max = std::max(rel_tol * res_initial, abs_tol);

   if (print_level >= 0) {
      mfem::out << "TR dogleg: initial ||r|| = " << res << "\n";
   }

   // Phase 5.11.G — SNLS-style two-condition convergence test at
   // iter 0 (pre-loop). Equivalent to the legacy
   //   `if (res <= max(rel_tol*res_initial, abs_tol)) ...`
   // but evaluates each condition separately so the diagnostic
   // sink and 5.11.I post-processor can label which fired.
   {
      const bool conv_abs = (res <= abs_tol);
      const bool conv_rel = (res <= rel_tol * res_initial);
      const bool converged_now = conv_abs || conv_rel;

      // Phase 5.11.F — diagnostic sink, iter 0.
      if (m_diagnostic_sink) {
         m_diagnostic_sink(NewtonIterDiagnostic{
            0, res, res_initial, norm_max, converged_now, &r, &x});
      }

      if (converged_now) {
         converged = true;
         final_iter = 0;
         final_norm = res;
         return;
      }
   }

   // --- Initialize trust-region state ---
   double delta = delta_ctrl.deltaInit;
   double rho = 0.0;
   bool reject_prev = false;

   // Persisted across iterations when a step is not rejected
   double Jg_2 = 0.0;
   double nr_norm = 0.0;

   int it = 0;
   converged = false;

   // --- Main iteration loop ---
   while (it < max_iter) {
      it++;

      // If the previous step was not rejected, recompute Newton
      // direction and steepest descent at the current x. Material
      // state is current because oper_mech->Mult(x, r) was just
      // called (either pre-loop on iter 0 or at the end of the
      // previous accepted iter).
      if (!reject_prev) {
         CALI_CXX_MARK_SCOPE("TR_newton_setup");

         mfem::Operator &J = oper_mech->GetGradient(x);

         // Steepest descent direction: grad = J^T * r. When
         // scaler_active, J is the 5.11.D ScaledJacobianOperator
         // and grad ends up in scaled coords by virtue of the
         // wrapper's MultTranspose convention.
         {
            CALI_CXX_MARK_SCOPE("TR_gradient_transpose");
            J.MultTranspose(r, grad);
         }

         // Compute ||J * grad||^2 for the optimal Cauchy step length
         //    alpha_cauchy = ||grad||^2 / ||J*grad||^2
         {
            J.Mult(grad, Jg_temp);
            Jg_2 = Dot(Jg_temp, Jg_temp);
         }

         // Solve Newton system: J * c = r, then nrStep = -c.
         // CGSolver follows the same convention as ExaNewtonSolver
         // where the Krylov solve produces c such that the Newton
         // update would be x -= c. For the dogleg we want
         // nrStep = -J^{-1} r, so we negate after the solve.
         {
            CALI_CXX_MARK_SCOPE("TR_newton_solve");
            c = 0.0;
            this->CGSolver(J, r, c);

            // Phase 5.11.G — when scaler_active, prec_mech is the
            // 5.11.D ScaledSaddleSolver wrapper, which returns c
            // in physical coords (the wrapper multiplies the inner
            // Krylov's dx_solver output by D for the Newton
            // u_phys-update protocol). The dogleg needs c in
            // SCALED coords because it interpolates with grad
            // (above) which is in scaled coords. Apply the scaler
            // to recover dx_solver before negating.
            if (scaler_active) {
               mfem::BlockVector c_view;
               c_view.Update(c, m_scaler_block_offsets);
               m_scaler->ApplyToIncrement(c_view);
            }

            nrStep = c;
            nrStep.Neg();
         }

         nr_norm = Norm(nrStep);
      }

      // Save state for potential step rejection
      x_prev = x;

      // Compute the dogleg step. All inputs and outputs are in
      // whatever coordinate system grad/nrStep are in — scaled
      // when scaler_active, physical otherwise. The math inside
      // Dogleg(...) is coord-agnostic; it uses MFEM's MPI-aware
      // Dot()/Norm() on whatever vectors arrive.
      double pred_resid = 0.0;
      bool use_nr = false;
      Dogleg(delta, res_0, nr_norm, Jg_2, grad, nrStep,
             delx, pred_resid, use_nr);

      // Phase 5.11.G — when scaler_active, delx is in scaled
      // coords. Convert to physical before applying to x (which
      // is in physical throughout). With the scaler disabled this
      // branch is skipped and delx stays in physical.
      if (scaler_active) {
         mfem::BlockVector delx_view;
         delx_view.Update(delx, m_scaler_block_offsets);
         m_scaler->UnapplyToIncrement(delx_view);
      }

      // Apply the trial step: x = x_prev + delx
      x = x_prev;
      x += delx;

      // Evaluate residual at the trial point
      reject_prev = false;
      {
         CALI_CXX_MARK_SCOPE("TR_trial_eval");
         oper_mech->Mult(x, r);
         if (have_b) { r -= b; }
      }

      res = Norm(r);

      if (print_level >= 0) {
         mfem::out << "TR dogleg: iter " << it
                   << ", ||r|| = " << res
                   << ", delta = " << delta
                   << (use_nr ? " [NR]" : " [DL]")
                   << "\n";
      }

      // Phase 5.11.G — SNLS-style two-condition convergence test.
      // Same OR-of-thresholds as the pre-loop block above; kept
      // explicit (not lumped into a max() threshold) so the
      // diagnostic sink can carry the two flags through 5.11.I.
      const bool conv_abs = (res <= abs_tol);
      const bool conv_rel = (res <= rel_tol * res_initial);
      const bool converged_now = conv_abs || conv_rel;

      // Phase 5.11.F — diagnostic sink invocation (per loop iter).
      // Fires AFTER res has been updated at the trial point and
      // BEFORE the convergence-check break, mirroring NR/NRLS.
      // For TRDOG `norm_max` is the legacy lumped threshold,
      // emitted for 5.11.I's diagnostic logging only — the actual
      // convergence decision is the OR of conv_abs / conv_rel
      // captured in converged_now.
      if (m_diagnostic_sink) {
         m_diagnostic_sink(NewtonIterDiagnostic{
            it, res, res_initial, norm_max, converged_now, &r, &x});
      }

      if (converged_now) {
         converged = true;
         break;
      }

      // Update delta from actual vs predicted reduction. May flag
      // for rejection. With scaler_active, both `res` (current
      // scaled norm), `res_0` (previous-iter scaled norm), and
      // `pred_resid` (output of Dogleg, in scaled coords) are in
      // the same scaled-merit space, so rho is consistent without
      // further work.
      bool delta_ok = delta_ctrl.UpdateDelta(
         delta, res, res_0, pred_resid, reject_prev,
         use_nr, nr_norm, rho, print_level);

      if (!delta_ok) {
         if (print_level >= 0) {
            mfem::out << "TR dogleg: delta control failure at iter "
                      << it << "\n";
         }
         converged = false;
         break;
      }

      // If the step is rejected, revert x and residual.
      // On the next iteration, reject_prev == true so we skip the
      // Newton solve and recompute the dogleg with the updated
      // (smaller) delta. The Jacobian, grad, nrStep, and Jg_2
      // remain valid from the last accepted state.
      if (reject_prev) {
         if (print_level > 0) {
            mfem::out << "TR dogleg: rejecting step, reverting to "
                         "previous state\n";
         }
         x = x_prev;
         res = res_0;
      }

      res_0 = res;
   }

   final_iter = it;
   final_norm = res;

   if (!converged && print_level >= 0) {
      mfem::out << "TR dogleg: failed to converge in " << it
                << " iterations, final ||r|| = " << res << "\n";
   }
}