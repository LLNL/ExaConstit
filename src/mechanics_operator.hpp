
#ifndef mechanics_operator_hpp
#define mechanics_operator_hpp

#include "mfem.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_model.hpp"
#include "mechanics_umat.hpp"
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
      mutable mfem::Vector diag, qpts_dshape, el_x, px, el_jac;
      mutable mfem::Operator *Jacobian;
      const mfem::Vector *x;
      const mfem::ParGridFunction &x_ref;
      const mfem::ParGridFunction &x_cur;
      mutable PANonlinearMechOperatorGradExt *pa_oper;
      mutable MechOperatorJacobiSmoother *prec_oper;
      const mfem::Operator *elem_restrict_lex;
      Assembly assembly;
      /// nonlinear model
      ExaModel *model;
      /// Variable telling us if we should use the UMAT specific
      /// stuff
      MechType mech_type;

      const mfem::Array2D<bool> &ess_bdr_comps;

   public:
      NonlinearMechOperator(mfem::ParFiniteElementSpace &fes,
                            mfem::Array<int> &ess_bdr,
                            mfem::Array2D<bool> &ess_bdr_comp,
                            ExaOptions &options,
                            mfem::QuadratureFunction &q_matVars0,
                            mfem::QuadratureFunction &q_matVars1,
                            mfem::QuadratureFunction &q_sigma0,
                            mfem::QuadratureFunction &q_sigma1,
                            mfem::QuadratureFunction &q_matGrad,
                            mfem::QuadratureFunction &q_kinVars0,
                            mfem::QuadratureFunction &q_vonMises,
                            mfem::ParGridFunction &ref_crds,
                            mfem::ParGridFunction &beg_crds,
                            mfem::ParGridFunction &end_crds,
                            mfem::Vector &matProps,
                            int nStateVars);

      /// Computes our jacobian operator for the entire system to be used within
      /// the newton raphson solver.
      virtual mfem::Operator &GetGradient(const mfem::Vector &x) const override;

      /// This computes the necessary quantities needed for when the BCs have been
      /// updated. So, we need the old Jacobian operator and old residual term
      /// that now includes the additional force term from the change in BCs on
      /// the unconstrained nodes.
      virtual mfem::Operator& GetUpdateBCsAction(const mfem::Vector &k, 
                                      const mfem::Vector &x,
                                      mfem::Vector &y) const;

      /// Performs the action of our function / force vector
      virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const override;

      /// Sets all of the data up for the Mult and GetGradient method
      /// This is of significant interest to be able to do partial assembly operations.
      using mfem::NonlinearForm::Setup;

      template<bool upd_crds>
      void Setup(const mfem::Vector &k) const;

      void SetupJacobianTerms() const;
      void CalculateDeformationGradient(mfem::QuadratureFunction &def_grad) const;

      // We need the solver to update the end coords after each iteration has been complete
      // We'll also want to have a way to update the coords before we start running the simulations.
      // It might also allow us to set a velocity at every point, so we could test the models almost
      // as if we're doing a MPS.
      void UpdateEndCoords(const mfem::Vector& vel) const;

      // Update the essential boundary conditions
      void UpdateEssTDofs(const mfem::Array<int> &ess_bdr, bool mono_def_flag);

      /// Get essential true dof list, if required
      const mfem::Array<int> &GetEssTDofList();

      ExaModel *GetModel() const;

      MechOperatorJacobiSmoother *GetPAPreconditioner(){ return prec_oper; }

      virtual ~NonlinearMechOperator();
};


#endif /* mechanics_operator_hpp */
