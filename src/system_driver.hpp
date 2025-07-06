#ifndef mechanics_system_driver_hpp
#define mechanics_system_driver_hpp

#include "mfem.hpp"
#include "mechanics_model.hpp"
#include "mechanics_operator.hpp"
#include "mechanics_solver.hpp"
#include "options/option_parser_v2.hpp"
#include "sim_state/simulation_state.hpp"
#include <iostream>

class LatticeTypeCubic;
template<class LatticeType>
class LightUp;
using LightUpCubic = LightUp<LatticeTypeCubic>;

// The NonlinearMechOperator class is what really drives the entire system.
// It's responsible for calling the Newton Rhapson solver along with several of
// our post-processing steps. It also contains all of the relevant information
// related to our Krylov iterative solvers.
class SystemDriver
{
   private:
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
      NonlinearMechOperator *mech_operator;
      RTModel class_device;
      bool auto_time = false;
      // define a boundary attribute array and initialize to 0
      std::unordered_map<std::string, mfem::Array<int> > ess_bdr;
      mfem::Array2D<double> ess_bdr_scale;
      std::unordered_map<std::string, mfem::Array2D<bool> > ess_bdr_component;
      mfem::Vector ess_velocity_gradient;
      // declare a VectorFunctionRestrictedCoefficient over the boundaries that have attributes
      // associated with a Dirichlet boundary condition (ids provided in input)
      mfem::VectorFunctionRestrictedCoefficient *ess_bdr_func;

      const bool vgrad_origin_flag = false;
      mfem::Vector vgrad_origin;
      const bool mono_def_flag = false;

      LightUpCubic* light_up = nullptr;
      SimulationState& m_sim_state;

   public:
      SystemDriver(SimulationState& sim_state);

      /// Get essential true dof list, if required
      const mfem::Array<int> &GetEssTDofList();

      /// Driver for the newton solver
      void Solve();

      /// Solve the Newton system for the 1st time step
      /// It was found that for large meshes a ramp up to our desired applied BC might
      /// be needed. It should be noted that this is no longer a const function since
      /// we modify several values/objects held by our class.
      void SolveInit() const;

      /// routine to update beginning step model variables with converged end
      /// step values
      void UpdateModel();
      void UpdateEssBdr();
      void UpdateVelocity();
      virtual ~SystemDriver();

};
#endif