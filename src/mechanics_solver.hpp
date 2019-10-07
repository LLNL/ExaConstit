
#ifndef MECHANICS_SOLVER
#define MECHANICS_SOLVER

#include "../../linalg/solvers.hpp"

namespace mfem
{
  
class IterativeSolver;

/// Newton's method for solving F(x)=b for a given operator F.
/** The method GetGradient() must be implemented for the operator F.
    The preconditioner is used (in non-iterative mode) to evaluate
    the action of the inverse gradient of the operator. */
class ExaNewtonSolver : public IterativeSolver
{
protected:
   mutable Vector r, c;

public:
   ExaNewtonSolver() { }

#ifdef MFEM_USE_MPI
   ExaNewtonSolver(MPI_Comm _comm) : IterativeSolver(_comm) { }
#endif
   virtual void SetOperator(const Operator &op);

   /// Set the linear solver for inverting the Jacobian.
   /** This method is equivalent to calling SetPreconditioner(). */
   virtual void SetSolver(Solver &solver) { prec = &solver; }

   /// Solve the nonlinear system with right-hand side @a b.
   /** If `b.Size() != Height()`, then @a b is assumed to be zero. */
   virtual void Mult(const Vector &b, Vector &x) const;
  
   /** @brief This method can be overloaded in derived classes to implement line
       search algorithms. */
   /** The base class implementation (NewtonSolver) simply returns 1. A return
       value of 0 indicates a failure, interrupting the Newton iteration. */
   virtual double ComputeScalingFactor(const Vector &x, const Vector &b) const
   { return 1.0; }
  
};

}

#endif
