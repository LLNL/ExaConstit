#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_log.hpp"
#include "system_driver.hpp"
#include "RAJA/RAJA.hpp"
#include <iostream>
#include "mechanics_kernels.hpp"
#include "ECMech_const.h"

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
                           QuadratureFunction *q_evec,
                           ParGridFunction &ref_crds,
                           ParGridFunction &beg_crds,
                           ParGridFunction &end_crds,
                           Vector &matProps,
                           int nStateVars)
   : fe_space(fes), def_grad(q_kinVars0), evec(q_evec)
{
   CALI_CXX_MARK_SCOPE("system_driver_init");
   mech_operator = new NonlinearMechOperator(fes, ess_bdr,
                                             options, q_matVars0, q_matVars1,
                                             q_sigma0, q_sigma1, q_matGrad,
                                             q_kinVars0, q_vonMises, ref_crds,
                                             beg_crds, end_crds, matProps,
                                             nStateVars);
   model = mech_operator->GetModel();

   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   mech_type = options.mech_type;
   class_device = options.rtmodel;
   avg_stress_fname = options.avg_stress_fname;
   avg_pl_work_fname = options.avg_pl_work_fname;
   avg_def_grad_fname = options.avg_def_grad_fname;
   avg_dp_tensor_fname = options.avg_dp_tensor_fname;
   additional_avgs = options.additional_avgs;

   // Partial assembly we need to use a matrix free option instead for our preconditioner
   // Everything else remains the same.
   if (options.assembly != Assembly::FULL) {
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
      J_gmres->SetPreconditioner(*J_prec);
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
      J_pcg->SetPreconditioner(*J_prec);
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
   if (options.nl_solver == NLSolver::NR) {
      newton_solver = new ExaNewtonSolver(fes.GetComm());
   }
   else if (options.nl_solver == NLSolver::NRLS) {
      newton_solver = new ExaNewtonLSSolver(fes.GetComm());
   }

   // Set the newton solve parameters
   newton_solver->iterative_mode = true;
   newton_solver->SetSolver(*J_solver);
   newton_solver->SetOperator(*mech_operator);
   newton_solver->SetPrintLevel(1);
   newton_solver->SetRelTol(options.newton_rel_tol);
   newton_solver->SetAbsTol(options.newton_abs_tol);
   newton_solver->SetMaxIter(options.newton_iter);
   if (options.visit || options.conduit || options.paraview || options.adios2) {
      postprocessing = true;
      CalcElementAvg(evec, model->GetMatVars0());
   } else {
      postprocessing = false;
   }
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
   newton_solver->Mult(zero, x);
   // Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
   // back to the current configuration...
   // Once the system has finished solving, our current coordinates configuration are based on what our
   // converged velocity field ended up being equal to.
   MFEM_VERIFY(newton_solver->GetConverged(), "Newton Solver did not converge.");
}

// Solve the Newton system for the 1st time step
// It was found that for large meshes a ramp up to our desired applied BC might
// be needed.
void SystemDriver::SolveInit(const Vector &xprev, Vector &x) const
{
   Vector b(x); b.UseDevice(true);
   
   Vector deltaF(x); deltaF.UseDevice(true);
   b = 0.0;
   // Want our vector for everything not on the Ess BCs to be 0
   // This means when we do K * diffF = b we're actually do the following:
   // K_uc * (x - x_prev)_c = deltaF_u
   {
      deltaF = 0.0;
      auto I = mech_operator->GetEssentialTrueDofs().Read();
      auto size = mech_operator->GetEssentialTrueDofs().Size();
      auto Y = deltaF.Write();
      auto XPREV = xprev.Read();
      auto X = x.Read();
      MFEM_FORALL(i, size, Y[I[i]] = X[I[i]] - XPREV[I[i]]; );
   }
   mfem::Operator &oper = mech_operator->GetUpdateBCsAction(xprev, deltaF, b);
   x = 0.0;
   //This will give us our -change in velocity
   //So, we want to add the previous velocity terms to it
   newton_solver->CGSolver(oper, b, x);
   auto X = x.ReadWrite();
   auto XPREV = xprev.Read();
   MFEM_FORALL(i, x.Size(), X[i] = -X[i] + XPREV[i]; );
}

void SystemDriver::UpdateModel()
{
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

   {
      CALI_CXX_MARK_SCOPE("avg_stress_computation");
      // Here we're getting the average stress value
      Vector stress(6);
      stress = 0.0;

      const QuadratureFunction *qstress = model->GetStress0();

      exaconstit::kernel::ComputeVolAvgTensor<true>(fes, qstress, stress, 6, class_device);

      cout.setf(ios::fixed);
      cout.setf(ios::showpoint);
      cout.precision(8);

      int my_id;
      MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
      // Now we're going to save off the average stress tensor to a file
      if (my_id == 0) {
         std::ofstream file;

         file.open(avg_stress_fname, std::ios_base::app);

         stress.Print(file, 6);
      }
   }

   if (mech_type == MechType::EXACMECH && additional_avgs) {
      CALI_CXX_MARK_SCOPE("extra_avgs_computations");
      const QuadratureFunction *qstate_var = model->GetMatVars0();
      // Here we're getting the average stress value
      Vector state_var(qstate_var->GetVDim());
      state_var = 0.0;

      std::string s_pl_work = "pl_work";
      auto qf_mapping = model->GetQFMapping();
      auto pair = qf_mapping->find(s_pl_work)->second;

      exaconstit::kernel::ComputeVolAvgTensor<false>(fes, qstate_var, state_var, state_var.Size(), class_device);

      cout.setf(ios::fixed);
      cout.setf(ios::showpoint);
      cout.precision(8);

      int my_id;
      MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
      // Now we're going to save off the average stress tensor to a file
      if (my_id == 0) {
         std::ofstream file;
         file.open(avg_pl_work_fname, std::ios_base::app);
         file << state_var[pair.first] << std::endl;
      }
      mech_operator->CalculateDeformationGradient(def_grad);
   }

   if (additional_avgs)
   {
      CALI_CXX_MARK_SCOPE("extra_avgs_def_grad_computation");
      const QuadratureFunction *qstate_var = &def_grad;
      // Here we're getting the average stress value
      Vector dgrad(qstate_var->GetVDim());
      dgrad = 0.0;

      exaconstit::kernel::ComputeVolAvgTensor<true>(fes, qstate_var, dgrad, dgrad.Size(), class_device);

      cout.setf(ios::fixed);
      cout.setf(ios::showpoint);
      cout.precision(8);

      int my_id;
      MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
      // Now we're going to save off the average stress tensor to a file
      if (my_id == 0) {
         std::ofstream file;
         file.open(avg_def_grad_fname, std::ios_base::app);
         dgrad.Print(file, dgrad.Size());
      }
   }

   if (mech_type == MechType::EXACMECH && additional_avgs) {
      CALI_CXX_MARK_SCOPE("extra_avgs_dp_tensor_computation");

      model->calcDpMat(def_grad);
      const QuadratureFunction *qstate_var = &def_grad;
      // Here we're getting the average stress value
      Vector dgrad(qstate_var->GetVDim());
      dgrad = 0.0;

      exaconstit::kernel::ComputeVolAvgTensor<true>(fes, qstate_var, dgrad, dgrad.Size(), class_device);

      cout.setf(ios::fixed);
      cout.setf(ios::showpoint);
      cout.precision(8);

      int my_id;
      MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
      // Now we're going to save off the average dp tensor to a file
      if (my_id == 0) {
         std::ofstream file;
         file.open(avg_dp_tensor_fname, std::ios_base::app);
         dgrad.Print(file, dgrad.Size());
      }
   }

   if(postprocessing) {
      CalcElementAvg(evec, model->GetMatVars0());
   }
}

void SystemDriver::CalcElementAvg(mfem::Vector *elemVal, const mfem::QuadratureFunction *qf)
{

   Mesh *mesh = fe_space.GetMesh();
   const FiniteElement &el = *fe_space.GetFE(0);
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));;

   const int nqpts = ir->GetNPoints();
   const int nelems = fe_space.GetNE();
   const int vdim = qf->GetVDim();

   const double* W = ir->GetWeights().Read();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS);

   const int DIM2 = 2;
   const int DIM3 = 3;
   std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
   std::array<RAJA::idx_t, DIM3> perm3 {{2, 1, 0}};

   RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);
   RAJA::Layout<DIM2> layout_ev = RAJA::make_permuted_layout({{ vdim, nelems } }, perm2);
   RAJA::Layout<DIM3> layout_qf = RAJA::make_permuted_layout({{vdim, nqpts, nelems}}, perm3);

   (*elemVal) = 0.0;

   RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);
   RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > qf_view(qf->Read(), layout_qf);
   RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > ev_view(elemVal->ReadWrite(), layout_ev);

   MFEM_FORALL(i, nelems, {
      double vol = 0.0;
      for(int j = 0; j < nqpts; j++) {
         const double wts = j_view(j, i) * W[j];
         vol += wts;
         for(int k = 0; k < vdim; k++) {
            ev_view(k, i) += qf_view(k, j, i) * wts;
         }
      }
      const double ivol = 1.0 / vol;
      for(int k = 0; k < vdim; k++) {
         ev_view(k, i) *= ivol;
      }
   });
}

void SystemDriver::ProjectCentroid(ParGridFunction &centroid)
{

   Mesh *mesh = fe_space.GetMesh();
   const FiniteElement &el = *fe_space.GetFE(0);
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));;

   const int nqpts = ir->GetNPoints();
   const int nelems = fe_space.GetNE();
   const int vdim = mesh->SpaceDimension();

   const double* W = ir->GetWeights().Read();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS | GeometricFactors::COORDINATES);

   const int DIM2 = 2;
   const int DIM3 = 3;
   std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
   std::array<RAJA::idx_t, DIM3> perm3 {{2, 1, 0}};

   RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);
   RAJA::Layout<DIM2> layout_ev = RAJA::make_permuted_layout({{ vdim, nelems } }, perm2);
   RAJA::Layout<DIM3> layout_qf = RAJA::make_permuted_layout({{nqpts, vdim, nelems}}, perm3);

   centroid = 0.0;

   RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);
   RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > x_view(geom->X.Read(), layout_qf);
   RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > ev_view(centroid.ReadWrite(), layout_ev);

   MFEM_FORALL(i, nelems, {
      double vol = 0.0;
      for(int j = 0; j < nqpts; j++) {
         const double wts = j_view(j, i) * W[j];
         vol += wts;
         for(int k = 0; k < vdim; k++) {
            ev_view(k, i) += x_view(j, k, i) * wts;
         }
      }
      const double ivol = 1.0 / vol;
      for(int k = 0; k < vdim; k++) {
         ev_view(k, i) *= ivol;
      }
   });
}

void SystemDriver::ProjectVolume(ParGridFunction &vol)
{
   Mesh *mesh = fe_space.GetMesh();
   const FiniteElement &el = *fe_space.GetFE(0);
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));;

   const int nqpts = ir->GetNPoints();
   const int nelems = fe_space.GetNE();

   const double* W = ir->GetWeights().Read();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, GeometricFactors::DETERMINANTS);

   const int DIM2 = 2;
   std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
   RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);

   double *vol_data = vol.ReadWrite();
   RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);

   MFEM_FORALL(i, nelems, {
      vol_data[i] = 0.0;
      for(int j = 0; j < nqpts; j++) {
         vol_data[i] += j_view(j, i) * W[j];
      }
   });
}

void SystemDriver::ProjectModelStress(ParGridFunction &s)
{
   CalcElementAvg(&s, model->GetStress0());
}

void SystemDriver::ProjectVonMisesStress(ParGridFunction &vm, const ParGridFunction &s)
{
   const int npts = vm.Size();

   const int DIM2 = 2;
   std::array<RAJA::idx_t, DIM2> perm2{{ 1, 0 } };

   RAJA::Layout<DIM2> layout_stress = RAJA::make_permuted_layout({{ 6, npts } }, perm2);
   RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > stress_view(s.Read(), layout_stress);
   double *vm_data = vm.ReadWrite();

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

}

void SystemDriver::ProjectHydroStress(ParGridFunction &hss, const ParGridFunction &s)
{
   const int npts = hss.Size();

   const int DIM2 = 2;
   std::array<RAJA::idx_t, DIM2> perm2{{ 1, 0 } };

   RAJA::Layout<DIM2> layout_stress = RAJA::make_permuted_layout({{ 6, npts } }, perm2);
   RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > stress_view(s.Read(), layout_stress);
   double* hydro = hss.ReadWrite();

   const double one_third = 1.0 / 3.0;

   MFEM_FORALL(i, npts, {
      hydro[i] = one_third * (stress_view(0, i) + stress_view(1, i) + stress_view(2, i));
   });

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

      VectorQuadratureFunctionCoefficient qfvc(*evec);
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

      VectorQuadratureFunctionCoefficient qfvc(*evec);
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

      VectorQuadratureFunctionCoefficient qfvc(*evec);
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

      VectorQuadratureFunctionCoefficient qfvc(*evec);
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

      VectorQuadratureFunctionCoefficient qfvc(*evec);
      qfvc.SetComponent(pair.first, pair.second);
      h.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
   }
   return;
}

// This one requires that the deviatoric strain be converted from 5d rep to 6d
// and have vol. contribution added.
void SystemDriver::ProjectElasticStrains(ParGridFunction &estrain)
{
   if (mech_type == MechType::EXACMECH) {
      std::string s_estrain = "elas_strain";
      std::string s_rvol = "rel_vol";
      auto qf_mapping = model->GetQFMapping();
      auto espair = qf_mapping->find(s_estrain)->second;
      auto rvpair = qf_mapping->find(s_rvol)->second;

      const int e_offset = espair.first;
      const int rv_offset = rvpair.first;

      int _size = estrain.Size();
      int nelems = _size / 6;

      auto data_estrain = mfem::Reshape(estrain.HostReadWrite(), 6, nelems);
      auto data_evec = mfem::Reshape(evec->HostReadWrite(), evec->GetVDim(), nelems);
      // The below is outputting the full elastic strain in the crystal ref frame
      // We'd only stored the 5d deviatoric elastic strain, so we need to convert
      // it over to the 6d version and add in the volume elastic strain contribution.
      for (int i = 0; i < nelems; i++) {
         const double t1 = ecmech::sqr2i * data_evec(0 + e_offset, i);
         const double t2 = ecmech::sqr6i * data_evec(1 + e_offset, i);
         //
         // Volume strain is ln(V^e_mean) term aka ln(relative volume)
         // Our plastic deformation has a det(1) aka no change in volume change
         const double elas_vol_strain = log(data_evec(rv_offset, i));
         // We output elastic strain formulation such that the relationship
         // between V^e and \varepsilon is just V^e = I + \varepsilon
         data_estrain(0, i) = (t1 - t2) + elas_vol_strain; // 11
         data_estrain(1, i) = (-t1 - t2) + elas_vol_strain ; // 22
         data_estrain(2, i) = ecmech::sqr2b3 * data_evec(1 + e_offset, i) + elas_vol_strain; // 33
         data_estrain(3, i) = ecmech::sqr2i * data_evec(4 + e_offset, i); // 23
         data_estrain(4, i) = ecmech::sqr2i * data_evec(3 + e_offset, i); // 31
         data_estrain(5, i) = ecmech::sqr2i * data_evec(2 + e_offset, i); // 12
      }
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
   delete newton_solver;
   delete mech_operator;
}