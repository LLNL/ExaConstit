#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_log.hpp"
#include "mechanics_operator.hpp"
#include "mechanics_solver.hpp"
#include "system_driver.hpp"
#include "option_parser.hpp"
#include "RAJA/RAJA.hpp"
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
   CALI_CXX_MARK_SCOPE("system_driver_init");
   mech_operator = new NonlinearMechOperator(fes, ess_bdr,
                                             options, q_matVars0, q_matVars1,
                                             q_sigma0, q_sigma1, q_matGrad,
                                             q_kinVars0, q_vonMises, beg_crds,
                                             end_crds, matProps, nStateVars);
   model = mech_operator->GetModel();

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   mech_type = options.mech_type;
   class_device = options.rtmodel;
   // Partial assembly we need to use a matrix free option instead for our preconditioner
   // Everything else remains the same.
   if (options.assembly == Assembly::PA) {
      J_prec = mech_operator->GetPAPreconditioner();
   }
   else {
      if (options.solver == KrylovSolver::GMRES || options.solver == KrylovSolver::PCG) {
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
      }
      else {
         printf("using minres solver \n");
         HypreSmoother *J_hypreSmoother = new HypreSmoother;
         J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
         J_hypreSmoother->SetPositiveDiagonal(true);
         J_prec = J_hypreSmoother;
      }
   }
   if (options.solver == KrylovSolver::GMRES) {
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
      if (options.assembly != Assembly::PA) {
         J_gmres->SetPreconditioner(*J_prec);
      }
      J_solver = J_gmres;
   }
   else if (options.solver == KrylovSolver::PCG) {
      CGSolver *J_pcg = new CGSolver(fe_space.GetComm());
      // These tolerances are currently hard coded while things are being debugged
      // but they should eventually be moved back to being set by the options
      // The relative tolerance should be at this point or smaller
      J_pcg->SetRelTol(options.krylov_rel_tol);
      // The absolute tolerance could probably get even smaller then this
      J_pcg->SetAbsTol(options.krylov_abs_tol);
      J_pcg->SetMaxIter(options.krylov_iter);
      J_pcg->SetPrintLevel(0);
      if (options.assembly != Assembly::PA) {
         J_pcg->SetPreconditioner(*J_prec);
      }
      J_solver = J_pcg;
   }
   else {
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
   Vector init_x(x); init_x.UseDevice(true);
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
                                       Vector& tensor, int size)
{
   Mesh *mesh = fes->GetMesh();
   const FiniteElement &el = *fes->GetFE(0);
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));;

   const int nqpts = ir->GetNPoints();
   const int nelems = fes->GetNE();
   const int npts = nqpts * nelems;

   const double* W = ir->GetWeights().Read();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS);

   double el_vol = 0.0;
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   double data[size];

   const int DIM2 = 2;
   std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
   RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);

   Vector wts(geom->detJ);
   RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > wts_view(wts.ReadWrite(), layout_geom);
   RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);

   RAJA::RangeSegment default_range(0, npts);

   MFEM_FORALL(i, nelems, {
      for (int j = 0; j < nqpts; j++) {
         wts_view(j, i) = j_view(j, i) * W[j];
      }
   });

   if (class_device == RTModel::CPU) {
      const double* qf_data = qf->HostRead();
      const double* wts_data = wts.HostRead();
      for (int j = 0; j < size; j++) {
         RAJA::ReduceSum<RAJA::seq_reduce, double> seq_sum(0.0);
         RAJA::ReduceSum<RAJA::seq_reduce, double> vol_sum(0.0);
         RAJA::forall<RAJA::loop_exec>(default_range, [ = ] (int i_npts){
            const double* stress = &(qf_data[i_npts * size]);
            seq_sum += wts_data[i_npts] * stress[j];
            vol_sum += wts_data[i_npts];
         });
         data[j] = seq_sum.get();
         el_vol = vol_sum.get();
      }
   }
#if defined(RAJA_ENABLE_OPENMP)
   if (class_device == RTModel::OPENMP) {
      const double* qf_data = qf->HostRead();
      const double* wts_data = wts.HostRead();
      for (int j = 0; j < size; j++) {
         RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> omp_sum(0.0);
         RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> vol_sum(0.0);
         RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [ = ] (int i_npts){
            const double* stress = &(qf_data[i_npts * size]);
            omp_sum += wts_data[i_npts] * stress[j];
            vol_sum += wts_data[i_npts];
         });
         data[j] = omp_sum.get();
         el_vol = vol_sum.get();
      }
   }
#endif
#if defined(RAJA_ENABLE_CUDA)
   if (class_device == RTModel::CUDA) {
      const double* qf_data = qf->Read();
      const double* wts_data = wts.Read();
      for (int j = 0; j < size; j++) {
         RAJA::ReduceSum<RAJA::cuda_reduce, double> cuda_sum(0.0);
         RAJA::ReduceSum<RAJA::cuda_reduce, double> vol_sum(0.0);
         RAJA::forall<RAJA::cuda_exec<1024> >(default_range, [ = ] RAJA_DEVICE(int i_npts){
            const double* stress = &(qf_data[i_npts * size]);
            cuda_sum += wts_data[i_npts] * stress[j];
            vol_sum += wts_data[i_npts];
         });
         data[j] = cuda_sum.get();
         el_vol = vol_sum.get();
      }
   }
   #endif

   for (int i = 0; i < size; i++) {
      tensor[i] = data[i];
   }

   MPI_Allreduce(&data, tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
   CALI_CXX_MARK_SCOPE("avg_stress_computation");
   const ParFiniteElementSpace *fes = GetFESpace();

   model->UpdateModelVars();

   // internally these two Update methods swap the internal data of the end step
   // with the begginning step using a simple pointer swap.
   // update the beginning step stress variable
   model->UpdateStress();
   // update the beginning step state variables
   if (model->numStateVars > 0) {
      model->UpdateStateVars();
   }

   // Here we're getting the average stress value
   Vector stress(6);
   stress = 0.0;

   const QuadratureFunction *qstress = model->GetStress0();

   ComputeVolAvgTensor(fes, qstress, stress, 6);

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
}

// This is probably wrong and we need to make this more in line with what
// the ProjectVonMisesStress is doing
void SystemDriver::ProjectModelStress(ParGridFunction &s)
{
   VectorQuadratureFunctionCoefficient stress(*model->GetStress0());
   s.ProjectDiscCoefficient(stress, mfem::GridFunction::ARITHMETIC);
}

void SystemDriver::ProjectVonMisesStress(ParGridFunction &vm)
{
   QuadratureFunction *qvm = model->GetVonMises();
   {
      const QuadratureFunction *stress = model->GetStress0();

      double* vm_data = qvm->ReadWrite();

      const int vdim = stress->GetVDim();
      const int npts = stress->Size() / vdim;

      const int DIM2 = 2;
      std::array<RAJA::idx_t, DIM2> perm2{{ 1, 0 } };
      // von Mises
      RAJA::Layout<DIM2> layout_stress = RAJA::make_permuted_layout({{ vdim, npts } }, perm2);
      RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > stress_view(stress->Read(), layout_stress);

      MFEM_FORALL(i, npts, {
         double term1 = stress_view(0, i) - stress_view(1, i);
         double term2 = stress_view(1, i) - stress_view(2, i);
         double term3 = stress_view(2, i) - stress_view(0, i);
         double term4 = stress_view(3, i) * stress_view(3, i)
                        + stress_view(4, i) * stress_view(4, i)
                        + stress_view(5, i) * stress_view(5, i);

         term1 *= term1;
         term2 *= term2;
         term3 *= term3;
         term4 *= 6.0;

         vm_data[i] = sqrt(0.5 * (term1 + term2 + term3 + term4));
      });
      qvm->HostReadWrite();
   }
   QuadratureFunctionCoefficient vonMisesStress(*qvm);
   vm.ProjectDiscCoefficient(vonMisesStress, mfem::GridFunction::ARITHMETIC);

   return;
}

void SystemDriver::ProjectHydroStress(ParGridFunction &hss)
{
   QuadratureFunction *hydro = model->GetVonMises();
   {
      const QuadratureFunction *stress = model->GetStress0();

      const int vdim = stress->GetVDim();
      const int npts = stress->Size() / vdim;
      const double one_third = 1.0 / 3.0;

      double* q_hydro = hydro->ReadWrite();

      const int DIM2 = 2;

      std::array<RAJA::idx_t, DIM2> perm2{{ 1, 0 } };
      // von Mises
      RAJA::Layout<DIM2> layout_stress = RAJA::make_permuted_layout({{ vdim, npts } }, perm2);
      RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > stress_view(stress->Read(), layout_stress);

      MFEM_FORALL(i, npts, {
         q_hydro[i] = one_third * (stress_view(0, i) + stress_view(1, i) + stress_view(2, i));
      });
      hydro->HostReadWrite();
   }
   QuadratureFunctionCoefficient hydroStress(*hydro);
   hss.ProjectDiscCoefficient(hydroStress, mfem::GridFunction::ARITHMETIC);

   return;
}

// These next group of Project* functions are only available with ExaCMech type models
// Need to figure out a smart way to get all of the indices that I want for down below
// that go with ExaModel
void SystemDriver::ProjectDpEff(ParGridFunction &dpeff)
{
   if (mech_type == MechType::EXACMECH) {
      std::string s_shrateEff = "shrateEff";
      auto qf_mapping = model->GetQFMapping();
      auto pair = qf_mapping->find(s_shrateEff)->second;

      VectorQuadratureFunctionCoefficient qfvc(*model->GetMatVars0());
      qfvc.SetComponent(pair.first, pair.second);
      dpeff.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
   }
   return;
}

void SystemDriver::ProjectEffPlasticStrain(ParGridFunction &pleff)
{
   if (mech_type == MechType::EXACMECH) {
      std::string s_shrEff = "shrEff";
      auto qf_mapping = model->GetQFMapping();
      auto pair = qf_mapping->find(s_shrEff)->second;

      VectorQuadratureFunctionCoefficient qfvc(*model->GetMatVars0());
      qfvc.SetComponent(pair.first, pair.second);
      pleff.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
   }
   return;
}

void SystemDriver::ProjectShearRate(ParGridFunction &gdot)
{
   if (mech_type == MechType::EXACMECH) {
      std::string s_gdot = "gdot";
      auto qf_mapping = model->GetQFMapping();
      auto pair = qf_mapping->find(s_gdot)->second;

      VectorQuadratureFunctionCoefficient qfvc(*model->GetMatVars0());
      qfvc.SetComponent(pair.first, pair.second);
      gdot.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
   }
   return;
}

// This one requires that the orientations be made unit normals afterwards
void SystemDriver::ProjectOrientation(ParGridFunction &quats)
{
   if (mech_type == MechType::EXACMECH) {
      std::string s_quats = "quats";
      auto qf_mapping = model->GetQFMapping();
      auto pair = qf_mapping->find(s_quats)->second;

      VectorQuadratureFunctionCoefficient qfvc(*model->GetMatVars0());
      qfvc.SetComponent(pair.first, pair.second);
      quats.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);

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

         inv_norm = 1.0 / sqrt(norm);

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
      std::string s_hard = "hardness";
      auto qf_mapping = model->GetQFMapping();
      auto pair = qf_mapping->find(s_hard)->second;

      VectorQuadratureFunctionCoefficient qfvc(*model->GetMatVars0());
      qfvc.SetComponent(pair.first, pair.second);
      h.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
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

SystemDriver::~SystemDriver()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete mech_operator;
}