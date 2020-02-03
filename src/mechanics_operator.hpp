
#ifndef mechanics_operator_hpp
#define mechanics_operator_hpp

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_umat.hpp"
#include "mechanics_ecmech.hpp"
#include "option_parser.hpp"
#include "mechanics_operator_ext.hpp"

// The NonlinearMechOperator class is what really drives the entire system.
// It's responsible for calling the Newton Rhapson solver along with several of
// our post-processing steps. It also contains all of the relevant information
// related to our Krylov iterative solvers.
class NonlinearMechOperator : public mfem::NonlinearForm
{
   protected:

      mfem::ParFiniteElementSpace &fe_space;
      mfem::ParNonlinearForm *Hform;
      mutable mfem::Vector diag;
      mutable mfem::Operator *Jacobian;
      const mfem::Vector *x;
      mutable PANonlinearMechOperatorGradExt *pa_oper;
      mutable MechOperatorJacobiSmoother *prec_oper;
      bool partial_assembly;
      /// nonlinear model
      ExaModel *model;
      /// Variable telling us if we should use the UMAT specific
      /// stuff
      MechType mech_type;

   public:
      NonlinearMechOperator(mfem::ParFiniteElementSpace &fes,
                            mfem::Array<int> &ess_bdr,
                            ExaOptions &options,
                            mfem::QuadratureFunction &q_matVars0,
                            mfem::QuadratureFunction &q_matVars1,
                            mfem::QuadratureFunction &q_sigma0,
                            mfem::QuadratureFunction &q_sigma1,
                            mfem::QuadratureFunction &q_matGrad,
                            mfem::QuadratureFunction &q_kinVars0,
                            mfem::QuadratureFunction &q_vonMises,
                            mfem::ParGridFunction &beg_crds,
                            mfem::ParGridFunction &end_crds,
                            mfem::Vector &matProps,
                            int nStateVars);

      /// Computes our jacobian operator for the entire system to be used within
      /// the newton raphson solver.
      virtual mfem::Operator &GetGradient(const mfem::Vector &x) const override;

      /// Performs the action of our function / force vector
      virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const override;

      /// Sets all of the data up for the Mult and GetGradient method
      /// This is of significant interest to be able to do partial assembly operations.
      void Setup(const mfem::Vector &k) const;

      // We need the solver to update the end coords after each iteration has been complete
      // We'll also want to have a way to update the coords before we start running the simulations.
      // It might also allow us to set a velocity at every point, so we could test the models almost
      // as if we're doing a MPS.
      void UpdateEndCoords(const mfem::Vector& vel) const;

      /// Get essential true dof list, if required
      const mfem::Array<int> &GetEssTDofList();

      ExaModel *GetModel() const;

      MechOperatorJacobiSmoother *GetPAPreconditioner(){return prec_oper;}

      virtual ~NonlinearMechOperator();
};


#endif /* mechanics_operator_hpp */
