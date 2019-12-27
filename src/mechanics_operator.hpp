
#ifndef mechanics_operator_hpp
#define mechanics_operator_hpp

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_umat.hpp"
#include "mechanics_ecmech.hpp"
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

//The NonlinearMechOperator class is what really drives the entire system.
//It's responsible for calling the Newton Rhapson solver along with several of
//our post-processing steps. It also contains all of the relevant information
//related to our Krylov iterative solvers.
class NonlinearMechOperator : public mfem::TimeDependentOperator
{
public:
   SimVars solVars;
protected:
   mfem::ParFiniteElementSpace &fe_space;
   
   mfem::ParNonlinearForm *Hform;
   mutable mfem::Operator *Jacobian;
   const mfem::Vector *x;
   
   /// Newton solver for the operator
   ExaNewtonSolver newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   mfem::Solver *J_solver;
   /// Preconditioner for the Jacobian
   mfem::Solver *J_prec;
   /// nonlinear model
   ExaModel *model;
   /// Variable telling us if we should use the UMAT specific
   /// stuff
   MechType mech_type;
   int newton_iter;
   int myid;
   
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
   
   /// Required to use the native newton solver
   virtual mfem::Operator &GetGradient(const mfem::Vector &x) const override;
   virtual void Mult(const mfem::Vector &k, mfem::Vector &y) const override;
   //We need the solver to update the end coords after each iteration has been complete
   //We'll also want to have a way to update the coords before we start running the simulations.
   //It might also allow us to set a velocity at every point, so we could test the models almost
   //as if we're doing a MPS.
   void UpdateEndCoords(const mfem::Vector& vel) const;
   /// Driver for the newton solver
   void Solve(mfem::Vector &x) const;
   
   /// Solve the Newton system for the 1st time step
   /// It was found that for large meshes a ramp up to our desired applied BC might
   /// be needed. It should be noted that this is no longer a const function since
   /// we modify several values/objects held by our class.
   void SolveInit(mfem::Vector &x);
   
   /// Get essential true dof list, if required
   const mfem::Array<int> &GetEssTDofList();
   
   /// Get FE space
   const mfem::ParFiniteElementSpace *GetFESpace() { return &fe_space; }
   
   /// routine to update beginning step model variables with converged end
   /// step values
   void UpdateModel();
   /// Computes a volume average tensor/vector of some quadrature function
   /// it returns the vol avg value.
   void ComputeVolAvgTensor(const mfem::ParFiniteElementSpace* fes,
                            const mfem::QuadratureFunction* qf,
                            mfem::Vector& tensor,
                            int size);
   
   void ProjectModelStress(mfem::ParGridFunction &s);
   void ProjectVonMisesStress(mfem::ParGridFunction &vm);
   void ProjectHydroStress(mfem::ParGridFunction &hss);
   //These next group of Project* functions are only available with ExaCMech type models
   void ProjectDpEff(mfem::ParGridFunction &dpeff);
   void ProjectEffPlasticStrain(mfem::ParGridFunction &pleff);
   void ProjectShearRate(mfem::ParGridFunction &gdot);
   //This one requires that the orientations be made unit normals afterwards
   void ProjectOrientation(mfem::ParGridFunction &quats);
   //Here this can be either the CRSS for a voce model or relative dislocation density
   //value for the MTS model.
   void ProjectH(mfem::ParGridFunction &h);
   
   void SetTime(const double t);
   void SetDt(const double dt);
   void SetModelDebugFlg(const bool dbg);
   
   void DebugPrintModelVars(int procID, double time);
   
   virtual ~NonlinearMechOperator();
};


#endif /* mechanics_operator_hpp */
