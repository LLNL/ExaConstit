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
      bool last_step = false;
   public:
      double GetTime() const { return time; }
      double GetDTime() const { return dt; }
      bool   GetLastStep() const { return last_step; }

      void SetTime(double t) { time = t; }
      void SetDt(double dtime) { dt = dtime; }
      void SetLastStep(bool last) { last_step = last; }
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
      bool postprocessing = false;
      bool additional_avgs = false;
      bool auto_time = false;
      double dt_class = 0.0;
      double dt_min = 0.0;
      double dt_scale = 1.0;
      mfem::QuadratureFunction &def_grad;
      std::string avg_stress_fname;
      std::string avg_pl_work_fname;
      std::string avg_def_grad_fname;
      std::string avg_dp_tensor_fname;
      std::string auto_dt_fname;

      mfem::QuadratureFunction *evec;

      // define a boundary attribute array and initialize to 0
      std::unordered_map<std::string, mfem::Array<int> > ess_bdr;
      mfem::Array2D<double> ess_bdr_scale;
      std::unordered_map<std::string, mfem::Array2D<int> > ess_bdr_component;
      mfem::Vector ess_velocity_gradient;
      // declare a VectorFunctionRestrictedCoefficient over the boundaries that have attributes
      // associated with a Dirichlet boundary condition (ids provided in input)
      mfem::VectorFunctionRestrictedCoefficient *ess_bdr_func;

      const bool vgrad_origin_flag = false;
      mfem::Vector vgrad_origin;

   public:
      SystemDriver(mfem::ParFiniteElementSpace &fes,
                   ExaOptions &options,
                   mfem::QuadratureFunction &q_matVars0,
                   mfem::QuadratureFunction &q_matVars1,
                   mfem::QuadratureFunction &q_sigma0,
                   mfem::QuadratureFunction &q_sigma1,
                   mfem::QuadratureFunction &q_matGrad,
                   mfem::QuadratureFunction &q_kinVars0,
                   mfem::QuadratureFunction &q_vonMises,
                   mfem::QuadratureFunction *q_evec,
                   mfem::ParGridFunction &ref_crds,
                   mfem::ParGridFunction &beg_crds,
                   mfem::ParGridFunction &end_crds,
                   mfem::Vector &matProps,
                   int nStateVars);

      /// Get FE space
      const mfem::ParFiniteElementSpace *GetFESpace() { return &fe_space; }

      /// Get essential true dof list, if required
      const mfem::Array<int> &GetEssTDofList();

      /// Driver for the newton solver
      void Solve(mfem::Vector &x);

      /// Solve the Newton system for the 1st time step
      /// It was found that for large meshes a ramp up to our desired applied BC might
      /// be needed. It should be noted that this is no longer a const function since
      /// we modify several values/objects held by our class.
      void SolveInit(const mfem::Vector &xprev, mfem::Vector &x) const;

      /// routine to update beginning step model variables with converged end
      /// step values
      void UpdateModel();
      void UpdateEssBdr();
      void UpdateVelocity(mfem::ParGridFunction &velocity, mfem::Vector &vel_tdofs);

      void ProjectCentroid(mfem::ParGridFunction &centroid);
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

      // This one requires that the deviatoric strain be converted from 5d rep to 6d
      // and have vol. contribution added.
      void ProjectElasticStrains(mfem::ParGridFunction &estrain);

      void SetTime(const double t);
      void SetDt(const double dt);
      double GetDt();
      void SetModelDebugFlg(const bool dbg);

      // Computes the element average of a quadrature function and stores it in a
      // vector. This is meant to be a helper function for the Project* methods.
      void CalcElementAvg(mfem::Vector *elemVal, const mfem::QuadratureFunction *qf);
      virtual ~SystemDriver();

};
#endif