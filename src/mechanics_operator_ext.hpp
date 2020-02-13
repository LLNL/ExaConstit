#ifndef mechanics_operator_ext_hpp
#define mechanics_operator_ext_hpp

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"

// The NonlinearMechOperatorExt class contains all of the relevant info related to our
// partial assembly class.
class NonlinearMechOperatorExt : public mfem::Operator
{
   protected:
      mfem::NonlinearForm *oper_mech; // Not owned
   public:

      NonlinearMechOperatorExt(mfem::NonlinearForm *_mech_operator);

      virtual mfem::MemoryClass GetMemoryClass() const
      { return mfem::Device::GetMemoryClass(); }

      // Any assembly operation we would need to use before we might need to use
      // the Mult operator.
      virtual void Assemble() = 0;

      // Here we would assemble the diagonal of any matrix-like operation we might be
      // performing.
      virtual void AssembleDiagonal(mfem::Vector &diag) = 0;
};

// We'll pass this on through the GetGradient method which can be used
// within our Iterative solver.
class PANonlinearMechOperatorGradExt : public NonlinearMechOperatorExt
{
   protected:
      const mfem::FiniteElementSpace *fes; // Not owned
      mutable mfem::Vector localX, localY, ones;
      const mfem::Operator *elem_restrict_lex; // Not owned

   public:
      PANonlinearMechOperatorGradExt(mfem::NonlinearForm *_mech_operator);

      void Assemble();
      void AssembleDiagonal(mfem::Vector &diag);
      void Mult(const mfem::Vector &x, mfem::Vector &y) const;
};

/// Jacobi smoothing for a given bilinear form (no matrix necessary).
/// We're going to be using a l1-jacobi here.
/** Useful with tensorized, partially assembled operators. Can also be defined
    by given diagonal vector. This is basic Jacobi iteration; for tolerances,
    iteration control, etc. wrap with SLISolver. */
class MechOperatorJacobiSmoother  : public mfem::Solver
{
   public:

      /** Application is by the *inverse* of the given vector. It is assumed that
          the underlying operator acts as the identity on entries in ess_tdof_list,
          corresponding to (assembled) DIAG_ONE policy or ConstratinedOperator in
          the matrix-free setting. */
      MechOperatorJacobiSmoother(const mfem::Vector &d,
                                 const mfem::Array<int> &ess_tdof_list,
                                 const double damping = 1.0);
      ~MechOperatorJacobiSmoother() {}

      void Mult(const mfem::Vector &x, mfem::Vector &y) const;

      void SetOperator(const mfem::Operator &op) { oper = &op; }

      void Setup(const mfem::Vector &diag);
      void UpdateEssBCs(const mfem::Array<int> &ess_tdof_list);

   private:
      const int N;
      mfem::Vector dinv;
      const double damping;
      const mfem::Array<int> &ess_tdof_list;
      mutable mfem::Vector residual;

      const mfem::Operator *oper;
};


#endif /* mechanics_operator_hpp */
