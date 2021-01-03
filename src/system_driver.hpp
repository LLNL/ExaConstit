#ifndef mechanics_system_driver_hpp
#define mechanics_system_driver_hpp

#include "mfem.hpp"
#include "mechanics_model.hpp"
#include "mechanics_operator.hpp"
#include "mechanics_solver.hpp"
#include "option_parser.hpp"
#include <iostream>

class SimVars
{
   protected:
      double time;
      double dt;
   public:
      double GetTime() const { return time; }

      double GetDTime() const { return dt; }

      void SetTime(double t) { time = t; }

      void SetDt(double dtime) { dt = dtime; }
};

// The NonlinearMechOperator class is what really drives the entire system.
// It's responsible for calling the Newton Rhapson solver along with several of
// our post-processing steps. It also contains all of the relevant information
// related to our Krylov iterative solvers.
class SystemDriver
{
   public:
      SimVars solVars;
   private:
      mfem::ParFiniteElementSpace &fe_space;
      /// Newton solver for the operator
      ExaNewtonSolver* newton_solver;
      /// Solver for the Jacobian solve in the Newton method
      mfem::Solver *J_solver;
      /// Preconditioner for the Jacobian
      mfem::Solver *J_prec;
      /// nonlinear model
      ExaModel *model;
      int newton_iter;
      int myid;
      /// Variable telling us if we should use the UMAT specific
      /// stuff
      MechType mech_type;
      NonlinearMechOperator *mech_operator;
      RTModel class_device;
      bool postprocessing;
      mfem::QuadratureFunction *evec;

   public:
      SystemDriver(mfem::ParFiniteElementSpace &fes,
                   mfem::Array<int> &ess_bdr,
                   ExaOptions &options,
                   mfem::QuadratureFunction &q_matVars0,
                   mfem::QuadratureFunction &q_matVars1,
                   mfem::QuadratureFunction &q_sigma0,
                   mfem::QuadratureFunction &q_sigma1,
                   mfem::QuadratureFunction &q_matGrad,
                   mfem::QuadratureFunction &q_kinVars0,
                   mfem::QuadratureFunction &q_vonMises,
                   mfem::QuadratureFunction *q_evec,
                   mfem::ParGridFunction &beg_crds,
                   mfem::ParGridFunction &end_crds,
                   mfem::Vector &matProps,
                   int nStateVars);

      /// Get FE space
      const mfem::ParFiniteElementSpace *GetFESpace() { return &fe_space; }

      /// Get essential true dof list, if required
      const mfem::Array<int> &GetEssTDofList();

      /// Driver for the newton solver
      void Solve(mfem::Vector &x) const;

      /// Solve the Newton system for the 1st time step
      /// It was found that for large meshes a ramp up to our desired applied BC might
      /// be needed. It should be noted that this is no longer a const function since
      /// we modify several values/objects held by our class.
      void SolveInit(const mfem::Vector &xprev, mfem::Vector &x);
      void SolveInit(mfem::Vector &x);

      /// routine to update beginning step model variables with converged end
      /// step values
      void UpdateModel();

      void UpdateEssBdr(mfem::Array<int> &ess_bdr) const { mech_operator->UpdateEssTDofs(ess_bdr); }

      /// Computes a volume average tensor/vector of some quadrature function
      /// it returns the vol avg value.
      void ComputeVolAvgTensor(const mfem::ParFiniteElementSpace* fes,
                               const mfem::QuadratureFunction* qf,
                               mfem::Vector& tensor,
                               int size);

      void ProjectVolume(mfem::ParGridFunction &vol);
      void ProjectModelStress(mfem::ParGridFunction &s);
      void ProjectVonMisesStress(mfem::ParGridFunction &vm, const mfem::ParGridFunction &s);
      void ProjectHydroStress(mfem::ParGridFunction &hss, const mfem::ParGridFunction &s);

      // These next group of Project* functions are only available with ExaCMech type models
      void ProjectDpEff(mfem::ParGridFunction &dpeff);
      void ProjectEffPlasticStrain(mfem::ParGridFunction &pleff);
      void ProjectShearRate(mfem::ParGridFunction &gdot);

      // This one requires that the orientations be made unit normals afterwards
      void ProjectOrientation(mfem::ParGridFunction &quats);

      // Here this can be either the CRSS for a voce model or relative dislocation density
      // value for the MTS model.
      void ProjectH(mfem::ParGridFunction &h);

      void SetTime(const double t);
      void SetDt(const double dt);
      void SetModelDebugFlg(const bool dbg);

      // Computes the element average of a quadrature function and stores it in a
      // vector. This is meant to be a helper function for the Project* methods.
      void CalcElementAvg(mfem::Vector *elemVal, const mfem::QuadratureFunction *qf);

      virtual ~SystemDriver();

};
#endif