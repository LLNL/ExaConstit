#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_operator.hpp"
#include "mechanics_solver.hpp"
#include "system_driver.hpp"
#include "option_parser.hpp"
#include <iostream>

using namespace std;
using namespace mfem;


SystemDriver::SystemDriver(ParFiniteElementSpace &fes,
                           Array<int> &ess_bdr,
                           ExaOptions &options,
                           QuadratureFunction &q_matVars0,
                           QuadratureFunction &q_matVars1,
                           QuadratureFunction &q_sigma0,
                           QuadratureFunction &q_sigma1,
                           QuadratureFunction &q_matGrad,
                           QuadratureFunction &q_kinVars0,
                           QuadratureFunction &q_vonMises,
                           ParGridFunction &beg_crds,
                           ParGridFunction &end_crds,
                           Vector &matProps,
                           int nStateVars)
   : fe_space(fes),
   newton_solver(fes.GetComm())
{
   mech_operator = new NonlinearMechOperator(fes, ess_bdr,
                                             options, q_matVars0, q_matVars1,
                                             q_sigma0, q_sigma1, q_matGrad,
                                             q_kinVars0, q_vonMises, beg_crds,
                                             end_crds, matProps, nStateVars);
   model = mech_operator->GetModel();

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   mech_type = options.mech_type;

   if (options.solver == KrylovSolver::GMRES) {
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      HYPRE_Solver h_amg = (HYPRE_Solver) * prec_amg;
      HYPRE_Real st_val = 0.90;
      HYPRE_Real rt_val = -10.0;
      // HYPRE_Real om_val = 1.0;

      //
      int ml = HYPRE_BoomerAMGSetMaxLevels(h_amg, 30);
      ml = HYPRE_BoomerAMGSetCoarsenType(h_amg, 0);
      ml = HYPRE_BoomerAMGSetMeasureType(h_amg, 0);
      ml = HYPRE_BoomerAMGSetStrongThreshold(h_amg, st_val);
      ml = HYPRE_BoomerAMGSetNumSweeps(h_amg, 3);
      ml = HYPRE_BoomerAMGSetRelaxType(h_amg, 8);
      // int rwt = HYPRE_BoomerAMGSetRelaxWt(h_amg, rt_val);
      // int ro = HYPRE_BoomerAMGSetOuterWt(h_amg, om_val);
      // Dimensionality of our problem
      ml = HYPRE_BoomerAMGSetNumFunctions(h_amg, 3);
      ml = HYPRE_BoomerAMGSetSmoothType(h_amg, 3);
      ml = HYPRE_BoomerAMGSetSmoothNumLevels(h_amg, 3);
      ml = HYPRE_BoomerAMGSetSmoothNumSweeps(h_amg, 3);
      ml = HYPRE_BoomerAMGSetVariant(h_amg, 0);
      ml = HYPRE_BoomerAMGSetOverlap(h_amg, 0);
      ml = HYPRE_BoomerAMGSetDomainType(h_amg, 1);
      ml = HYPRE_BoomerAMGSetSchwarzRlxWeight(h_amg, rt_val);
      // Just to quite the compiler warnings...
      ml++;

      prec_amg->SetPrintLevel(0);

      J_prec = prec_amg;

      GMRESSolver *J_gmres = new GMRESSolver(fe_space.GetComm());
      // These tolerances are currently hard coded while things are being debugged
      // but they should eventually be moved back to being set by the options
      // J_gmres->iterative_mode = false;
      // The relative tolerance should be at this point or smaller
      J_gmres->SetRelTol(options.krylov_rel_tol);
      // The absolute tolerance could probably get even smaller then this
      J_gmres->SetAbsTol(options.krylov_abs_tol);
      J_gmres->SetMaxIter(options.krylov_iter);
      J_gmres->SetPrintLevel(0);
      J_gmres->SetPreconditioner(*J_prec);
      J_solver = J_gmres;
   }
   else if (options.solver == KrylovSolver::PCG) {
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      HYPRE_Solver h_amg = (HYPRE_Solver) * prec_amg;
      HYPRE_Real st_val = 0.90;
      HYPRE_Real rt_val = -10.0;
      // HYPRE_Real om_val = 1.0;
      //
      int ml = HYPRE_BoomerAMGSetMaxLevels(h_amg, 30);
      ml = HYPRE_BoomerAMGSetCoarsenType(h_amg, 0);
      ml = HYPRE_BoomerAMGSetMeasureType(h_amg, 0);
      ml = HYPRE_BoomerAMGSetStrongThreshold(h_amg, st_val);
      ml = HYPRE_BoomerAMGSetNumSweeps(h_amg, 3);
      ml = HYPRE_BoomerAMGSetRelaxType(h_amg, 8);
      // int rwt = HYPRE_BoomerAMGSetRelaxWt(h_amg, rt_val);
      // int ro = HYPRE_BoomerAMGSetOuterWt(h_amg, om_val);
      // Dimensionality of our problem
      ml = HYPRE_BoomerAMGSetNumFunctions(h_amg, 3);
      ml = HYPRE_BoomerAMGSetSmoothType(h_amg, 3);
      ml = HYPRE_BoomerAMGSetSmoothNumLevels(h_amg, 3);
      ml = HYPRE_BoomerAMGSetSmoothNumSweeps(h_amg, 3);
      ml = HYPRE_BoomerAMGSetVariant(h_amg, 0);
      ml = HYPRE_BoomerAMGSetOverlap(h_amg, 0);
      ml = HYPRE_BoomerAMGSetDomainType(h_amg, 1);
      ml = HYPRE_BoomerAMGSetSchwarzRlxWeight(h_amg, rt_val);
      // Just to quite the compiler warnings...
      ml++;

      prec_amg->SetPrintLevel(0);
      J_prec = prec_amg;

      CGSolver *J_pcg = new CGSolver(fe_space.GetComm());
      // These tolerances are currently hard coded while things are being debugged
      // but they should eventually be moved back to being set by the options
      // The relative tolerance should be at this point or smaller
      J_pcg->SetRelTol(options.krylov_rel_tol);
      // The absolute tolerance could probably get even smaller then this
      J_pcg->SetAbsTol(options.krylov_abs_tol);
      J_pcg->SetMaxIter(options.krylov_iter);
      J_pcg->SetPrintLevel(0);
      J_pcg->iterative_mode = true;
      J_pcg->SetPreconditioner(*J_prec);
      J_solver = J_pcg;
   } // The SuperLU capabilities were gotten rid of due to the size of our systems
   // no longer making it a viable option to keep 1e6+ dof systems
   // Also, a well tuned PCG should be much faster than SuperLU for systems roughly
   // 5e5 and up.
   else {
      printf("using minres solver \n");
      HypreSmoother *J_hypreSmoother = new HypreSmoother;
      J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
      J_hypreSmoother->SetPositiveDiagonal(true);
      J_prec = J_hypreSmoother;

      MINRESSolver *J_minres = new MINRESSolver(fe_space.GetComm());
      J_minres->SetRelTol(options.krylov_rel_tol);
      J_minres->SetAbsTol(options.krylov_abs_tol);
      J_minres->SetMaxIter(options.krylov_iter);
      J_minres->SetPrintLevel(-1);
      J_minres->SetPreconditioner(*J_prec);
      J_solver = J_minres;
   }
   // We might want to change our # iterations used in the newton solver
   // for the 1st time step. We'll want to swap back to the old one after this
   // step.
   newton_iter = options.newton_iter;

   // Set the newton solve parameters
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(*mech_operator);
   newton_solver.SetPrintLevel(1);
   newton_solver.SetRelTol(options.newton_rel_tol);
   newton_solver.SetAbsTol(options.newton_abs_tol);
   newton_solver.SetMaxIter(options.newton_iter);
}

const Array<int> &SystemDriver::GetEssTDofList()
{
   return mech_operator->GetEssTDofList();
}

// Solve the Newton system
void SystemDriver::Solve(Vector &x) const
{
   Vector zero;
   // We provide an initial guess for what our current coordinates will look like
   // based on what our last time steps solution was for our velocity field.
   // The end nodes are updated before the 1st step of the solution here so we're good.
   newton_solver.Mult(zero, x);
   // Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
   // back to the current configuration...
   // Once the system has finished solving, our current coordinates configuration are based on what our
   // converged velocity field ended up being equal to.
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
}

// Solve the Newton system for the 1st time step
// It was found that for large meshes a ramp up to our desired applied BC might
// be needed.
void SystemDriver::SolveInit(Vector &x)
{
   Vector zero;
   Vector init_x(x);
   // We shouldn't need more than 5 NR to converge to a solution during our
   // initial step in our solution.
   // We'll change this back to the old value at the end of the function.
   newton_solver.SetMaxIter(5);
   // We provide an initial guess for what our current coordinates will look like
   // based on what our last time steps solution was for our velocity field.
   // The end nodes are updated before the 1st step of the solution here so we're good.
   model->init_step = true;
   newton_solver.Mult(zero, x);
   model->init_step = false;
   // Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
   // back to the current configuration...

   // If the step didn't converge we're going to do a ramp up to the applied
   // velocity that we want. The assumption being made here is that our 1st time
   // step should be in the linear elastic regime. Therefore, we should be able
   // to go from our reduced solution to the desired solution. This has been noted
   // to be a problem when really increasing the mesh size.
   if (!newton_solver.GetConverged()) {
      // We're going to reset our initial applied BCs to being 1/64 of the original
      if (myid == 0) {
         mfem::out << "Solution didn't converge. Reducing initial condition to 1/4 original value\n";
      }
      x = init_x;
      x *= 0.25;
      // We're going to keep track of how many cuts we need to make. Hopefully we
      // don't have to reduce it anymore then 3 times total.
      int i = 1;

      // We provide an initial guess for what our current coordinates will look like
      // based on what our last time steps solution was for our velocity field.
      // The end nodes are updated before the 1st step of the solution here so we're good.
      newton_solver.Mult(zero, x);
      // Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
      // back to the current configuration...

      if (!newton_solver.GetConverged()) {
         // We're going to reset our initial applied BCs to being 1/16 of the original
         if (myid == 0) {
            mfem::out << "Solution didn't converge. Reducing initial condition to 1/16 original value\n";
         }
         x = init_x;
         x *= 0.0625;
         // We're going to keep track of how many cuts we need to make. Hopefully we
         // don't have to reduce it anymore then 3 times total.
         i++;

         // We provide an initial guess for what our current coordinates will look like
         // based on what our last time steps solution was for our velocity field.
         // The end nodes are updated before the 1st step of the solution here so we're good.
         newton_solver.Mult(zero, x);
         // Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
         // back to the current configuration...

         if (!newton_solver.GetConverged()) {
            // We're going to reset our initial applied BCs to being 1/64 of the original
            if (myid == 0) {
               mfem::out << "Solution didn't converge. Reducing initial condition to 1/64 original value\n";
            }
            x = init_x;
            x *= 0.015625;
            // We're going to keep track of how many cuts we need to make. Hopefully we
            // don't have to reduce it anymore then 3 times total.
            i++;

            // We provide an initial guess for what our current coordinates will look like
            // based on what our last time steps solution was for our velocity field.
            // The end nodes are updated before the 1st step of the solution here so we're good.
            newton_solver.Mult(zero, x);

            MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge after 1/64 reduction of applied BCs.");
         } // end of 1/64 reduction case
      } // end of 1/16 reduction case

      // Here we're upscaling our previous converged solution to the next level.
      // The upscaling should be a good initial guess, since everything we're doing
      // is linear in this first step.
      // We then have the solution try and converge again with our better initial
      // guess of the solution.
      // It might be that this process only needs to occur once and we can directly
      // upscale from the lowest level to our top layer since we're dealing with
      // supposedly a linear elastic type problem here.
      for (int j = 0; j < i; j++) {
         if (myid == 0) {
            mfem::out << "Upscaling previous solution by factor of 4\n";
         }
         x *= 4.0;
         // We provide an initial guess for what our current coordinates will look like
         // based on what our last time steps solution was for our velocity field.
         // The end nodes are updated before the 1st step of the solution here so we're good.
         newton_solver.Mult(zero, x);

         // Once the system has finished solving, our current coordinates configuration are based on what our
         // converged velocity field ended up being equal to.
         // If the update fails we want to exit.
         MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
      } // end of upscaling process
   } // end of 1/4 reduction case

   // Reset our max number of iterations to our original desired value.
   newton_solver.SetMaxIter(newton_iter);
}

void SystemDriver::ComputeVolAvgTensor(const ParFiniteElementSpace* fes,
                                       const QuadratureFunction* qf,
                                       Vector& tensor, int size){
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpace* qspace = qf->GetSpace();

   double el_vol = 0.0;
   double temp_wts = 0.0;
   double incr = 0.0;

   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

   // loop over elements
   for (int i = 0; i < fes->GetNE(); ++i) {
      // get element transformation for the ith element
      ElementTransformation* Ttr = fes->GetElementTransformation(i);
      ir = &(qspace->GetElementIntRule(i));
      int elem_offset = qf_offset * ir->GetNPoints();
      // loop over element quadrature points
      for (int j = 0; j < ir->GetNPoints(); ++j) {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Ttr->SetIntPoint(&ip);
         // Here we're setting the integration for the average value
         temp_wts = ip.weight * Ttr->Weight();
         // This tells us the element volume
         el_vol += temp_wts;
         incr += 1.0;
         int k = 0;
         for (int m = 0; m < size; ++m) {
            tensor[m] += temp_wts * qf_data[i * elem_offset + j * qf_offset + k];
            ++k;
         }
      }
   }

   double data[size];

   for (int i = 0; i < size; i++) {
      data[i] = tensor[i];
   }

   MPI_Allreduce(&data, tensor.GetData(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   double temp = el_vol;

   // Here we find what el_vol should be equal to
   MPI_Allreduce(&temp, &el_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

   // We meed to multiple by 1/V by our tensor values to get the appropriate
   // average value for the tensor in the end.
   double inv_vol = 1.0 / el_vol;

   for (int m = 0; m < size; m++) {
      tensor[m] *= inv_vol;
   }
}

void SystemDriver::UpdateModel()
{
   const ParFiniteElementSpace *fes = GetFESpace();
   const FiniteElement *fe;
   const IntegrationRule *ir;

   model->UpdateModelVars();

   // internally these two Update methods swap the internal data of the end step
   // with the begginning step using a simple pointer swap.
   // update the beginning step stress variable
   model->UpdateStress();
   // update the beginning step state variables
   if (model->numStateVars > 0) {
      model->UpdateStateVars();
   }

   // update state variables on a ExaModel
   for (int i = 0; i < fes->GetNE(); ++i) {
      fe = fes->GetFE(i);
      ir = &(IntRules.Get(fe->GetGeomType(), 2 * fe->GetOrder() + 1));

      // loop over element quadrature points
      for (int j = 0; j < ir->GetNPoints(); ++j) {
         // compute von Mises stress
         model->ComputeVonMises(i, j);
      }
   }


   // Here we're getting the average stress value
   Vector stress;
   int size = 6;

   stress.SetSize(size);

   stress = 0.0;

   QuadratureVectorFunctionCoefficient* qstress = model->GetStress0();

   const QuadratureFunction* qf = qstress->GetQuadFunction();

   ComputeVolAvgTensor(fes, qf, stress, size);

   cout.setf(ios::fixed);
   cout.setf(ios::showpoint);
   cout.precision(8);

   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   // Now we're going to save off the average stress tensor to a file
   if (my_id == 0) {
      std::ofstream file;

      file.open("avg_stress.txt", std::ios_base::app);

      stress.Print(file, 6);
   }

   qstress = NULL;
   qf = NULL;

   // Here we're computing the average deformation gradient
   // Vector defgrad;
   // size = 9;

   // defgrad.SetSize(size);

   // defgrad = 0.0;

   // QuadratureVectorFunctionCoefficient* qdefgrad = model->GetDefGrad0();

   // const QuadratureFunction* qf1 = qdefgrad->GetQuadFunction();

   // ComputeVolAvgTensor(fes, qf1, defgrad, size);

   ////We're now saving the average def grad off to a file
   // if(my_id == 0){
   // std::ofstream file;

   // file.open("avg_dgrad.txt", std::ios_base::app);

   // defgrad.Print(file, 9);
   // }
}

// This is probably wrong and we need to make this more in line with what
// the ProjectVonMisesStress is doing
void SystemDriver::ProjectModelStress(ParGridFunction &s)
{
   QuadratureVectorFunctionCoefficient *stress;
   stress = model->GetStress0();
   s.ProjectDiscCoefficient(*stress, mfem::GridFunction::ARITHMETIC);

   stress = NULL;

   return;
}

void SystemDriver::ProjectVonMisesStress(ParGridFunction &vm)
{
   QuadratureFunctionCoefficient *vonMisesStress;
   vonMisesStress = model->GetVonMises();
   vm.ProjectDiscCoefficient(*vonMisesStress, mfem::GridFunction::ARITHMETIC);

   vonMisesStress = NULL;

   return;
}

void SystemDriver::ProjectHydroStress(ParGridFunction &hss)
{
   QuadratureVectorFunctionCoefficient *stress;
   stress = model->GetStress0();
   const QuadratureFunction* qf = stress->GetQuadFunction();
   const double* qf_data = qf->GetData();

   const int vdim = qf->GetVDim();
   const int pts = qf->Size() / vdim;
   const double one_third = 1.0 / 3.0;

   // One option if we want to save on memory would be to just use the
   // vonMises quadrature function and overwrite the data there. It's currently
   // not being used for anything other then visualization purposes.
   // QuadratureFunction q_hyrdro(qf->GetSpace, 1);
   // QuadratureFunctionCoefficient* hyrdroStress(&q_hyrdro);
   // Here we're just reusing the vonMises quadrature function already created.
   QuadratureFunctionCoefficient *hydroStress;
   hydroStress = model->GetVonMises();

   QuadratureFunction* hydro = hydroStress->GetQuadFunction();
   double* q_hydro = hydro->GetData();

   for (int i = 0; i < pts; i++) {
      const int ii = i * vdim;
      q_hydro[i] = one_third * (qf_data[ii] + qf_data[ii + 1] + qf_data[ii + 2]);
   }

   hss.ProjectDiscCoefficient(*hydroStress, mfem::GridFunction::ARITHMETIC);

   return;
}

// These next group of Project* functions are only available with ExaCMech type models
// Need to figure out a smart way to get all of the indices that I want for down below
// that go with ExaModel
void SystemDriver::ProjectDpEff(ParGridFunction &dpeff)
{
   if (mech_type == MechType::EXACMECH) {
      QuadratureVectorFunctionCoefficient* qfvc = model->GetMatVars0();
      qfvc->SetIndex(0);
      qfvc->SetLength(1);
      dpeff.ProjectDiscCoefficient(*qfvc, mfem::GridFunction::ARITHMETIC);
   }
   return;
}

void SystemDriver::ProjectEffPlasticStrain(ParGridFunction &pleff)
{
   if (mech_type == MechType::EXACMECH) {
      QuadratureVectorFunctionCoefficient* qfvc = model->GetMatVars0();
      qfvc->SetIndex(1);
      qfvc->SetLength(1);
      pleff.ProjectDiscCoefficient(*qfvc, mfem::GridFunction::ARITHMETIC);
   }
   return;
}

void SystemDriver::ProjectShearRate(ParGridFunction &gdot)
{
   if (mech_type == MechType::EXACMECH) {
      QuadratureVectorFunctionCoefficient* qfvc = model->GetMatVars0();
      qfvc->SetIndex(13);
      qfvc->SetLength(12);
      gdot.ProjectDiscCoefficient(*qfvc, mfem::GridFunction::ARITHMETIC);
   }
   return;
}

// This one requires that the orientations be made unit normals afterwards
void SystemDriver::ProjectOrientation(ParGridFunction &quats)
{
   if (mech_type == MechType::EXACMECH) {
      QuadratureVectorFunctionCoefficient* qfvc = model->GetMatVars0();
      qfvc->SetIndex(8);
      qfvc->SetLength(4);
      quats.ProjectDiscCoefficient(*qfvc, mfem::GridFunction::ARITHMETIC);

      // The below is normalizing the quaternion since it most likely was not
      // returned normalized
      int _size = quats.Size();
      int size = _size / 4;

      double norm = 0;
      double inv_norm = 0;
      int index = 0;

      for (int i = 0; i < size; i++) {
         index = i * 4;

         norm = quats(index + 0) * quats(index + 0);
         norm += quats(index + 1) * quats(index + 1);
         norm += quats(index + 2) * quats(index + 2);
         norm += quats(index + 3) * quats(index + 3);

         inv_norm = 1.0 / norm;

         for (int j = 0; j < 4; j++) {
            quats(index + j) *= inv_norm;
         }
      }
   }
   return;
}

// Here this can be either the CRSS for a voce model or relative dislocation density
// value for the MTS model.
void SystemDriver::ProjectH(ParGridFunction &h)
{
   if (mech_type == MechType::EXACMECH) {
      QuadratureVectorFunctionCoefficient* qfvc = model->GetMatVars0();
      qfvc->SetIndex(12);
      qfvc->SetLength(1);
      h.ProjectDiscCoefficient(*qfvc, mfem::GridFunction::ARITHMETIC);
   }
   return;
}

void SystemDriver::SetTime(const double t)
{
   solVars.SetTime(t);
   model->SetModelTime(t);
   return;
}

void SystemDriver::SetDt(const double dt)
{
   solVars.SetDt(dt);
   model->SetModelDt(dt);
   return;
}

void SystemDriver::DebugPrintModelVars(int procID, double time)
{
   // print material properties vector on the model
   Vector *props = model->GetMatProps();
   ostringstream props_name;
   props_name << "props." << setfill('0') << setw(6) << procID << "_" << time;
   ofstream props_ofs(props_name.str().c_str());
   props_ofs.precision(8);
   props->Print(props_ofs);

   // print the beginning step material state variables quadrature function
   QuadratureVectorFunctionCoefficient *mv0 = model->GetMatVars0();
   ostringstream mv_name;
   mv_name << "matVars." << setfill('0') << setw(6) << procID << "_" << time;
   ofstream mv_ofs(mv_name.str().c_str());
   mv_ofs.precision(8);

   QuadratureFunction *matVars0 = mv0->GetQuadFunction();
   matVars0->Print(mv_ofs);

   matVars0 = NULL;
   props = NULL;

   return;
}

SystemDriver::~SystemDriver()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete mech_operator;
}