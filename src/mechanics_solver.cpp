
#include "mfem.hpp"
#include "mechanics_solver.hpp"
#include "mfem/linalg/linalg.hpp"
#include "mfem/general/globals.hpp"
#include "mechanics_log.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>


using namespace std;
using namespace mfem;

void ExaNewtonSolver::SetOperator(const Operator &op)
{
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_ASSERT(height == width, "square Operator is required.");

   r.SetSize(width, Device::GetMemoryType()); r.UseDevice(true);
   c.SetSize(width, Device::GetMemoryType()); c.UseDevice(true);
}

void ExaNewtonSolver::SetOperator(const NonlinearForm &op)
{
   oper_mech = &op;
   oper = &op;
   height = op.Height();
   width = op.Width();
   MFEM_ASSERT(height == width, "square NonlinearForm is required.");

   r.SetSize(width, Device::GetMemoryType()); r.UseDevice(true);
   c.SetSize(width, Device::GetMemoryType()); c.UseDevice(true);
}

void ExaNewtonSolver::Mult(const Vector &b, Vector &x) const
{
   CALI_CXX_MARK_SCOPE("NR_solver");
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm, norm_max;
   double norm_prev, norm_ratio;
   const bool have_b = (b.Size() == Height());

   // Might want to use this to fix things later on for example when we have a
   // large residual. We might also want to eventually try and find a converged
   // relaxation factor which would mean resetting our solution vector a few times.
   Vector x_prev(x.Size());
   x_prev.UseDevice(true);

   if (!iterative_mode) {
      x = 0.0;
   }

   x_prev = x;

   oper_mech->Mult(x, r);
   if (have_b) {
      r -= b;
   }

   norm0 = norm = norm_prev = Norm(r);
   norm_ratio = 1.0;
   // Set the value for the norm that we'll exit on
   norm_max = std::max(rel_tol * norm, abs_tol);

   prec->iterative_mode = false;
   double scale = 1.0;

   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++) {
      // Make sure the norm is finite
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0) {
         mfem::out << "Newton iteration " << setw(2) << it
                   << " : ||r|| = " << norm;
         if (it > 0) {
            mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
         }
         mfem::out << '\n';
      }
      // See if our solution has converged and we can quit
      if (norm <= norm_max) {
         converged = 1;
         break;
      }
      // See if we've gone over the max number of desired iterations
      if (it >= max_iter) {
         converged = 0;
         break;
      }

      prec->SetOperator(oper_mech->GetGradient(x));
      CALI_MARK_BEGIN("krylov_solver");
      prec->Mult(r, c); // c = [DF(x_i)]^{-1} [F(x_i)-b]
                        // ExaConstit may use GMRES here

      CALI_MARK_END("krylov_solver");
      const double c_scale = scale;
      if (c_scale == 0.0) {
         converged = 0;
         break;
      }

      add(x, -c_scale, c, x); // full update to the current config
                              // ExaConstit (srw)

      // We now get our new residual
      oper_mech->Mult(x, r);
      if (have_b) {
         r -= b;
      }

      // Find our new norm and save our previous time step value.
      norm_prev = norm;
      norm = Norm(r);
      // We're going to more or less use a heuristic method here for now if
      // our ratio is greater than 1e-1 then we'll set our scaling factor for
      // the next iteration to 0.5.
      // We want to do this since it's not uncommon for us to run into the case
      // where our solution is oscillating over the one we actually want.
      // Eventually, we'll fix this in our scaling factor function.
      norm_ratio = norm / norm_prev;

      if (norm_ratio > 5.0e-1) {
         scale = 0.5;
         if (print_level >= 0) {
            mfem::out << "The relaxation factor for the next iteration has been reduced to " << scale << "\n";
         }
      }
      else {
         scale = 1.0;
      }
   }

   final_iter = it;
   final_norm = norm;
}

void ExaNewtonLSSolver::Mult(const Vector &b, Vector &x) const
{
   CALI_CXX_MARK_SCOPE("NRLS_solver");
   MFEM_ASSERT(oper != NULL, "the Operator is not set (use SetOperator).");
   MFEM_ASSERT(prec != NULL, "the Solver is not set (use SetSolver).");

   int it;
   double norm0, norm, norm_max;
   double norm_prev;
   const bool have_b = (b.Size() == Height());

   // Might want to use this to fix things later on for example when we have a
   // large residual. We might also want to eventually try and find a converged
   // relaxation factor which would mean resetting our solution vector a few times.
   Vector x_prev(x.Size());
   Vector Jr(x.Size());
   Jr.UseDevice(true);
   x_prev.UseDevice(true);

   if (!iterative_mode) {
      x = 0.0;
   }

   x_prev = x;

   oper_mech->Mult(x, r);
   if (have_b) {
      r -= b;
   }

   norm0 = norm = norm_prev = Norm(r);
   // Set the value for the norm that we'll exit on
   norm_max = std::max(rel_tol * norm, abs_tol);

   prec->iterative_mode = false;
   double scale = 1.0;

   // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
   for (it = 0; true; it++) {
      // Make sure the norm is finite
      MFEM_ASSERT(IsFinite(norm), "norm = " << norm);
      if (print_level >= 0) {
         mfem::out << "Newton iteration " << setw(2) << it
                   << " : ||r|| = " << norm;
         if (it > 0) {
            mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
         }
         mfem::out << '\n';
      }
      // See if our solution has converged and we can quit
      if (norm <= norm_max) {
         converged = 1;
         break;
      }
      // See if we've gone over the max number of desired iterations
      if (it >= max_iter) {
         converged = 0;
         break;
      }

      prec->SetOperator(oper_mech->GetGradient(x));
      CALI_MARK_BEGIN("krylov_solver");
      prec->Mult(r, c); // c = [DF(x_i)]^{-1} [F(x_i)-b]
                        // ExaConstit may use GMRES here
      CALI_MARK_END("krylov_solver");
      // This line search method is based on the quadratic variation of the norm
      // of the residual line search described in this conference paper:
      // https://doi.org/10.1007/978-3-642-01970-8_46 . We can probably do better
      // than this one.
      {
         CALI_CXX_MARK_SCOPE("Line Search");
         x_prev = x;
         add(x, -1.0, c, x);
         oper_mech->Mult(x, r);
         if(have_b) {
            r -= b;
         }
         double q1 = norm;
         double q3 = Norm(r);
         x = x_prev;
         add(x, -0.5, c, x);
         oper_mech->Mult(x, r);
         if(have_b) {
            r -= b;
         }
         double q2 = Norm(r);

         double eps = (3.0 * q1 - 4.0 * q2 + q3) / (4.0 * (q1 - 2.0 * q2 + q3));

         if ((q1 - 2.0 * q2 + q3) > 0 && eps > 0 && eps < 1) {
            scale = eps;
         } else if (q3 < q1) {
            scale = 1.0;
         } else {
            // We should probably just quit if this is the case...
            scale = 0.05;
         }

         if (print_level >= 0) {
            mfem::out << "The relaxation factor for this iteration is " << scale << std::endl;
         }

         x = x_prev;
      }

      const double c_scale = scale;
      if (c_scale == 0.0) {
         converged = 0;
         break;
      }

      add(x, -c_scale, c, x); // full update to the current config
                              // ExaConstit (srw)

      // We now get our new residual
      oper_mech->Mult(x, r);
      if (have_b) {
         r -= b;
      }

      // Find our new norm and save our previous time step value.
      norm_prev = norm;
      norm = Norm(r);

   }

   final_iter = it;
   final_norm = norm;
}