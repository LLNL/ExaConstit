#include "models/mechanics_umat.hpp"
#include "boundary_conditions/BCManager.hpp"
#include "utilities/assembly_ops.hpp"
#include "utilities/strain_measures.hpp"

#include "RAJA/RAJA.hpp"
#include "mfem/fem/qfunction.hpp"

#include <math.h> // log
#include <algorithm>
#include <iostream> // cerr


using namespace mfem;
using namespace std;

// NEW CONSTRUCTOR IMPLEMENTATION: Much simpler parameter list
// The key insight is that instead of passing in all QuadratureFunctions and material properties,
// we only pass in the essential UMAT-specific parameters and use the region ID to access
// data through SimulationState when needed.
AbaqusUmatModel::AbaqusUmatModel(const int region, int nStateVars,
                                 SimulationState& sim_state,
                                 const std::string& umat_library_path,
                                 const DynamicUmatLoader::LoadStrategy& load_strategy) :
                                 ExaModel(region, nStateVars, sim_state),
                                 umat_library_path_(umat_library_path),
                                 umat_function_(nullptr),
                                 load_strategy_(load_strategy),
                                 use_dynamic_loading_(!umat_library_path.empty())
{
   // Initialize working space QuadratureFunctions
   init_loc_sf_grads(m_sim_state.GetMeshParFiniteElementSpace());
   init_incr_end_def_grad();

   // If using dynamic loading with PERSISTENT strategy, load immediately
   if (use_dynamic_loading_ && load_strategy_ == DynamicUmatLoader::LoadStrategy::PERSISTENT) {
      if (!LoadUmatLibrary()) {
         throw std::runtime_error("Failed to load UMAT library: " + umat_library_path_);
      }
   }
}

AbaqusUmatModel::~AbaqusUmatModel() {
   // Unload library if needed
   if (use_dynamic_loading_ && load_strategy_ != DynamicUmatLoader::LoadStrategy::PERSISTENT) {
      UnloadUmatLibrary();
   }
}

// NEW HELPER METHOD: Get defGrad0 from SimulationState instead of using member variable
// This enables dynamic access to the correct region-specific deformation gradient data
std::shared_ptr<mfem::expt::PartialQuadratureFunction> AbaqusUmatModel::GetDefGrad0() {
    return m_sim_state.GetQuadratureFunction("def_grad_beg", m_region);
}

// UPDATED: UpdateModelVars now gets defGrad0 from SimulationState instead of member variable
void AbaqusUmatModel::UpdateModelVars()
{
   // UPDATED: Get defGrad0 from SimulationState instead of using member variable
   auto defGrad = GetDefGrad0();
   
   // update the beginning step deformation gradient
   double* dgrad0 = defGrad->HostReadWrite();
   double* dgrad1 = end_def_grad.HostReadWrite();
   
   // We just need to update our beginning of time step def. grad. with our
   // end step def. grad. now that they are equal.
   for (int i = 0; i < defGrad->Size(); i++) {
      dgrad0[i] = dgrad1[i];
   }
}

// Work through the initialization of all of this...
// UNCHANGED: This method doesn't directly access QuadratureFunctions that moved to SimulationState
void AbaqusUmatModel::init_loc_sf_grads(std::shared_ptr<mfem::ParFiniteElementSpace> fes)
{
   const FiniteElement *fe;
   const IntegrationRule *ir;
   
   // UPDATED: Get defGrad0 from SimulationState to determine quadrature space
   auto defGrad0 = GetDefGrad0();
   auto qspace = defGrad0->GetSpaceShared();

   ir = &(qspace->GetIntRule(0));

   const int NE = fes->GetNE();
   const int NQPTS = ir->GetNPoints();

   // get element transformation for the 0th element
   // We just want to get some basic stuff for now
   fe = fes->GetFE(0);

   // declare data to store shape function gradients
   // and element Jacobians
   DenseMatrix Jrt, DSh, DS;
   int dof = fe->GetDof(), dim = fe->GetDim();
   const int VDIM = dof * dim;

   DSh.SetSize(dof, dim);
   // This should probably be linked to the underlying quadrature function
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);

   // We now have enough information to create our loc0_sf_grad

   loc0_sf_grad.SetSpace(qspace, VDIM);
   double* data = loc0_sf_grad.HostReadWrite();

   // loop over elements
   for (int i = 0; i < NE; ++i) {
      // get element transformation for the ith element
      ElementTransformation* Ttr = fes->GetElementTransformation(i);
      fe = fes->GetFE(i);

      // PMatI.UseExternalData(el_x.ReadWrite(), dof, dim);

      ir = &(qspace->GetIntRule(i));

      // loop over integration points where the quadrature function is
      // stored
      for (int j = 0; j < NQPTS; ++j) {
         // The offset is the current location of the data
         int offset = (i * NQPTS * VDIM) + (j * VDIM);
         double* data_offset = data + offset;

         DS.UseExternalData(data_offset, dof, dim);

         const IntegrationPoint &ip = ir->IntPoint(j);
         Ttr->SetIntPoint(&ip);
         CalcInverse(Ttr->Jacobian(), Jrt);

         fe->CalcDShape(ip, DSh);
         Mult(DSh, Jrt, DS);
      }
   }
}

// UPDATED: init_incr_end_def_grad now gets defGrad0 from SimulationState
void AbaqusUmatModel::init_incr_end_def_grad()
{
   const IntegrationRule *ir;
   
   // UPDATED: Get defGrad0 from SimulationState instead of using member variable
   auto defGrad0 = GetDefGrad0();
   auto qspace = defGrad0->GetSpaceShared();

   ir = &(qspace->GetIntRule(0));

   const int TOTQPTS = qspace->GetSize();
   const int NQPTS = ir->GetNPoints();
   // We've got the same elements everywhere so we can do this.
   // If this assumption is no longer true we need to update the code
   const int NE = TOTQPTS / NQPTS;
   const int VDIM = defGrad0->GetVDim();

   incr_def_grad.SetSpace(qspace, VDIM);
   incr_def_grad = 0.0;
   double* incr_data = incr_def_grad.HostReadWrite();

   end_def_grad.SetSpace(qspace, VDIM);
   end_def_grad = 0.0;
   double* end_data = end_def_grad.HostReadWrite();

   // loop over elements
   for (int i = 0; i < NE; ++i) {
      // loop over integration points where the quadrature function is
      // stored
      for (int j = 0; j < NQPTS; ++j) {
         // The offset is the current location of the data
         int offset = (i * NQPTS * VDIM) + (j * VDIM);
         double* incr_data_offset = incr_data + offset;
         double* end_data_offset = end_data + offset;

         // It's now just initialized to being the identity matrix
         incr_data_offset[0] = 1.0;
         incr_data_offset[4] = 1.0;
         incr_data_offset[8] = 1.0;

         // It's now just initialized to being the identity matrix
         end_data_offset[0] = 1.0;
         end_data_offset[4] = 1.0;
         end_data_offset[8] = 1.0;
      }
   }
}

// UPDATED: calc_incr_end_def_grad now gets defGrad0 from SimulationState
void AbaqusUmatModel::calc_incr_end_def_grad(const ParGridFunction &x0)
{
   auto loc_fes = m_sim_state.GetMeshParFiniteElementSpace();
   const IntegrationRule *ir;
   
   // UPDATED: Get defGrad0 from SimulationState instead of using member variable
   auto defGrad0 = GetDefGrad0();
   auto qspace = defGrad0->GetSpaceShared();

   ir = &(qspace->GetIntRule(0));

   const int tot_qpts = qspace->GetSize();
   const int nqpts = ir->GetNPoints();
   // We've got the same type of elements everywhere so we can do this.
   // If this assumption is no longer true we need to update the code
   const int ne = tot_qpts / nqpts;
   const int vdim = defGrad0->GetVDim();
   // We also assume we're only dealing with 3D type elements.
   // If we aren't then this needs to change...
   const int dim = 3;
   const int vdim2 = loc0_sf_grad.GetVDim();
   const int dof = vdim2 / dim;

   double* incr_data = incr_def_grad.HostReadWrite();
   double* end_data = end_def_grad.HostReadWrite();
   double* int_data = defGrad0->HostReadWrite();
   double* ds_data = loc0_sf_grad.HostReadWrite();

   ParGridFunction x_gf(x0);

   DenseMatrix f_incr(dim, dim);
   DenseMatrix f_end(dim, dim);
   DenseMatrix f_beg(dim, dim);
   DenseMatrix f_beg_invr(dim, dim);
   DenseMatrix DS(dof, dim);
   DenseMatrix PMatI(dof, dim);
   // The below are constant but will change between steps
   Array<int> vdofs(vdim2);
   Vector el_x(PMatI.Data(), vdim2);

   // loop over elements
   for (int i = 0; i < ne; ++i) {
      loc_fes->GetElementVDofs(i, vdofs);
      // Our PMatI is now updated to the correct elemental values
      x_gf.GetSubVector(vdofs, el_x);
      // loop over integration points where the quadrature function is
      // stored
      for (int j = 0; j < nqpts; ++j) {
         // The offset is the current location of the data
         int offset = (i * nqpts * vdim) + (j * vdim);
         int offset2 = (i * nqpts * vdim2) + (j * vdim2);
         double* incr_data_offset = incr_data + offset;
         double* end_data_offset = end_data + offset;
         double* int_data_offset = int_data + offset;
         double* ds_data_offset = ds_data + offset2;

         f_end.UseExternalData(end_data_offset, dim, dim);
         f_beg.UseExternalData(int_data_offset, dim, dim);
         f_incr.UseExternalData(incr_data_offset, dim, dim);
         DS.UseExternalData(ds_data_offset, dof, dim);

         // Get the inverse of the beginning time step def. grad
         f_beg_invr = f_beg;
         f_beg_invr.Invert();

         // Find the end time step def. grad
         MultAtB(PMatI, DS, f_end);

         // Our incremental def. grad is now
         Mult(f_end, f_beg_invr, f_incr);
      }
   }
}

// UNCHANGED: These strain calculation methods don't access QuadratureFunctions
void AbaqusUmatModel::CalcLogStrainIncrement(DenseMatrix& dE, const DenseMatrix &Jpt)
{
   // calculate incremental logorithmic strain (Hencky Strain)
   // which is taken to be E = ln(U_hat) = 1/2 ln(C_hat), where
   // C_hat = (F_hat_T)F_hat, where F_hat = Jpt1 on the model
   // (available from MFEM element transformation computations).
   // We can compute F_hat, so use a spectral decomposition on C_hat to
   // obtain a form where we only have to take the natural log of the
   // eigenvalues
   // UMAT uses the E = ln(V) approach instead

   DenseMatrix F_hat, B_hat;

   constexpr int dim = 3;

   F_hat.SetSize(dim);
   B_hat.SetSize(dim);

   F_hat = Jpt;

   MultABt(F_hat, F_hat, B_hat);

   // compute eigenvalue decomposition of B
   double lambda[dim];
   double vec[dim * dim];
   B_hat.CalcEigenvalues(&lambda[0], &vec[0]);

   // compute ln(B) using spectral representation
   dE = 0.0;
   for (int i = 0; i<dim; ++i) { // outer loop for every eigenvalue/vector
      for (int j = 0; j<dim; ++j) { // inner loops for diadic product of eigenvectors
         for (int k = 0; k<dim; ++k) {
            // Dense matrices are col. maj. representation, so the indices were
            // reversed for it to be more cache friendly.
            dE(k, j) += 0.5 * log(lambda[i]) * vec[i * dim + j] * vec[i * dim + k];
         }
      }
   }

   return;
}

// This method calculates the Eulerian strain which is given as:
// e = 1/2 (I - B^(-1)) = 1/2 (I - F(^-T)F^(-1))
// UNCHANGED: This method doesn't access QuadratureFunctions
void AbaqusUmatModel::CalcEulerianStrainIncr(DenseMatrix& dE, const DenseMatrix &Jpt)
{
   constexpr int dim = 3;
   DenseMatrix Fincr(Jpt, dim);
   DenseMatrix Finv(dim), Binv(dim);

   double half = 1.0 / 2.0;

   CalcInverse(Fincr, Finv);

   MultAtB(Finv, Finv, Binv);

   dE = 0.0;

   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         dE(i, j) -= half * Binv(i, j);
      }

      dE(j, j) += half;
   }
}

// This method calculates the Lagrangian strain which is given as:
// E = 1/2 (C - I) = 1/2 (F^(T)F - I)
// UNCHANGED: This method doesn't access QuadratureFunctions
void AbaqusUmatModel::CalcLagrangianStrainIncr(DenseMatrix& dE, const DenseMatrix &Jpt)
{
   DenseMatrix C;

   constexpr int dim = 3;

   double half = 1.0 / 2.0;

   C.SetSize(dim);

   MultAtB(Jpt, Jpt, C);

   dE = 0.0;

   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         dE(i, j) += half * C(i, j);
      }

      dE(j, j) -= half;
   }

   return;
}

// UPDATED: Further testing needs to be conducted to make sure this still does everything it used to
// but it should. Since, it is just copy and pasted from the old EvalModel function and now
// has loops added to it. Now uses accessor methods to get QuadratureFunctions from SimulationState.
void AbaqusUmatModel::ModelSetup(const int nqpts, const int nelems, const int space_dim,
                                 const int /*nnodes*/, const Vector &jacobian,
                                 const Vector & /*loc_grad*/, const Vector &/*vel*/)
{
    // Load UMAT library if using on-demand loading
   if (use_dynamic_loading_ && load_strategy_ == DynamicUmatLoader::LoadStrategy::LOAD_ON_SETUP) {
      if (!LoadUmatLibrary()) {
         throw std::runtime_error("Failed to load UMAT library during ModelSetup: " + umat_library_path_);
      }
   }

   // Get region-specific element information
   auto stress0 = m_sim_state.GetQuadratureFunction("cauchy_stress_beg", m_region);
   auto qspace = stress0->GetPartialSpaceShared();
   
   // Determine actual elements to process for this region
   const mfem::Array<int>* local2global_ptr = nullptr;
   int local_nelems = nelems;

   if (!qspace->isFullSpace()) {
      const auto& local2global = qspace->getLocal2Global();
      local2global_ptr = &local2global;
      local_nelems = local2global.Size();
   }

   // All of this should be scoped to limit at least some of our memory usage
   {
      const auto end_crds = m_sim_state.getCurrentCoords();
      calc_incr_end_def_grad(*end_crds);
   }

   // ======================================================
   // Set UMAT input arguments
   // ======================================================

   // initialize Umat variables
   int ndi = 3; // number of direct stress components
   int nshr = 3; // number of shear stress components
   int ntens = ndi + nshr;
   int noel = 0;
   int npt = 0;
   int layer = 0;
   int kspt = 0;
   int kstep = 0;
   int kinc = 0;

   // set properties and state variables length (hard code for now);
   int nprops = GetMaterialProperties().size();
   int nstatv = numStateVars;

   double pnewdt = 10.0; // revisit this
   mfem::Vector props(nprops); // populate from the mat props vector wrapped by matProps on the base class
   mfem::Vector statev(nstatv); // populate from the state variables associated with this element/ip

   double rpl = 0.0; // volumetric heat generation per unit time, not considered
   double drpldt = 0.0; // variation of rpl wrt temperature set to 0.0
   double tempk = 300.0; // no thermal considered at this point
   double dtemp = 0.0; // no increment in thermal considered at this point
   double predef = 0.0; // no interpolated values of predefined field variables at ip point
   double dpred = 0.0; // no array of increments of predefined field variables
   double sse = 0.0; // specific elastic strain energy, mainly for output
   double spd = 0.0; // specific plastic dissipation, mainly for output
   double scd = 0.0; // specific creep dissipation, mainly for output
   double cmname = 0.0; // user defined UMAT name
   double celent = 0.0; // set element length

   // integration point coordinates
   // a material model shouldn't need this ever
   double coords[3] = { 0, 0, 0 };

   // set the time step
   double deltaTime = m_sim_state.getDeltaTime(); // set on the ExaModel base class

   // set time. Abaqus has odd increment definition. time[1] is the value of total
   // time at the beginning of the current increment. Since we are iterating from
   // tn to tn+1, this is just tn. time[0] is value of step time at the beginning
   // of the current increment. What is step time if not tn? It seems as though
   // they sub-increment between tn->tn+1, where there is a Newton Raphson loop
   // advancing the sub-increment. For now, set time[0] is set to t - dt/
   double time[2];
   time[0] = m_sim_state.getTime() - deltaTime;
   time[1] = m_sim_state.getTime();

   double stress[6]; // Cauchy stress at ip
   double ddsdt[6]; // variation of the stress increments wrt to temperature, set to 0.0
   double drplde[6]; // variation of rpl wrt strain increments, set to 0.0
   double stran[6]; // array containing total strains at beginning of the increment
   double dstran[6]; // array of strain increments

   double *drot; // rotation matrix for finite deformations
   double dfgrd0[9]; // deformation gradient at beginning of increment
   double dfgrd1[9]; // defomration gradient at the end of the increment.
                     // set to zero if nonlinear geometric effects are not
                     // included in the step as is the case for ExaConstit

   // UPDATED: Get defGrad0 from SimulationState instead of using member variable
   auto defGrad0 = GetDefGrad0();

   double* defgrad0 = defGrad0->HostReadWrite();
   double* defgrad1 = end_def_grad.HostReadWrite();
   double* incr_defgrad = incr_def_grad.HostReadWrite();
   DenseMatrix incr_dgrad, dgrad0, dgrad1;

   const int vdim = end_def_grad.GetVDim();
   double ddsdde[36]; // output Jacobian matrix of the constitutive model.
                      // ddsdde(i,j) defines the change in the ith stress component
                      // due to an incremental perturbation in the jth strain increment

   const int DIM4 = 4;

   std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
   // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
   RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ space_dim, space_dim, nqpts, nelems } }, perm4);
   RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.HostRead(), layout_jacob);

    // Update the element/IP loops to use proper indexing:
    for (int local_elemID = 0; local_elemID < local_nelems; local_elemID++) {
      // Map to global element ID for accessing global data
      const int global_elemID = local2global_ptr ? (*local2global_ptr)[local_elemID] : local_elemID;

      for (int ipID = 0; ipID < nqpts; ipID++) {
         // compute characteristic element length
         const double J11 = J(0, 0, ipID, global_elemID); // 0,0
         const double J21 = J(1, 0, ipID, global_elemID); // 1,0
         const double J31 = J(2, 0, ipID, global_elemID); // 2,0
         const double J12 = J(0, 1, ipID, global_elemID); // 0,1
         const double J22 = J(1, 1, ipID, global_elemID); // 1,1
         const double J32 = J(2, 1, ipID, global_elemID); // 2,1
         const double J13 = J(0, 2, ipID, global_elemID); // 0,2
         const double J23 = J(1, 2, ipID, global_elemID); // 1,2
         const double J33 = J(2, 2, ipID, global_elemID); // 2,2
         const double detJ = J11 * (J22 * J33 - J32 * J23) -
                             /* */ J21 * (J12 * J33 - J32 * J13) +
                             /* */ J31 * (J12 * J23 - J22 * J13);
         CalcElemLength(detJ);
         celent = elemLength;

         const int offset = local_elemID * nqpts * vdim + ipID * vdim;

         noel = local_elemID; // element id
         npt = ipID; // integration point number

         // initialize 1d arrays
         for (int i = 0; i<6; ++i) {
            stress[i] = 0.0;
            ddsdt[i] = 0.0;
            drplde[i] = 0.0;
            stran[i] = 0.0;
            dstran[i] = 0.0;
         }

         // initialize 6x6 2d arrays
         for (int i = 0; i<6; ++i) {
            for (int j = 0; j<6; ++j) {
               ddsdde[(i * 6) + j] = 0.0;
            }
         }


         incr_dgrad.UseExternalData((incr_defgrad + offset), 3, 3);
         dgrad0.UseExternalData((defgrad0 + offset), 3, 3);
         dgrad1.UseExternalData((defgrad1 + offset), 3, 3);

         DenseMatrix Uincr(3), Vincr(3);
         DenseMatrix Rincr(incr_dgrad, 3);
         CalcPolarDecompDefGrad(Rincr, Uincr, Vincr);

         drot = Rincr.GetData();

         // populate the beginning step and end step (or best guess to end step
         // within the Newton iterations) of the deformation gradients
         for (int i = 0; i<ndi; ++i) {
            for (int j = 0; j<ndi; ++j) {
               // Dense matrices have column major layout so the below is fine.
               dfgrd0[(i * 3) + j] = dgrad0(j, i);
               dfgrd1[(i * 3) + j] = dgrad1(j, i);
            }
         }

         // get state variables and material properties
         // UPDATED: These methods now use accessor methods to get QuadratureFunctions from SimulationState
         
         GetQFData(local_elemID, ipID, statev.HostReadWrite(), m_sim_state.GetQuadratureFunction("state_var_beg", m_region));
         {
            const auto prop_data = GetMaterialProperties();
            size_t index = 0;
            for (const auto& prop : prop_data) {
               props(index++) = prop;
            }
         }

         // get element stress and make sure ordering is ok
         double stressTemp[6];
         double stressTemp2[6];
         GetQFData(local_elemID, ipID, stressTemp, m_sim_state.GetQuadratureFunction("cauchy_stress_beg", m_region));

         // ensure proper ordering of the stress array. ExaConstit uses
         // Voigt notation (11, 22, 33, 23, 13, 12), while
         // ------------------------------------------------------------------
         // We use Voigt notation: (11, 22, 33, 23, 13, 12)
         //
         // ABAQUS USES:
         // (11, 22, 33, 12, 13, 23)
         // ------------------------------------------------------------------
         stress[0] = stressTemp[0];
         stress[1] = stressTemp[1];
         stress[2] = stressTemp[2];
         stress[3] = stressTemp[5];
         stress[4] = stressTemp[4];
         stress[5] = stressTemp[3];

         // Abaqus does mention wanting to use a log strain for large strains
         // It's also based on an updated lagrangian formulation so as long as
         // we aren't generating any crazy strains do we really need to use the
         // log strain?
         DenseMatrix LogStrain;
         LogStrain.SetSize(ndi); // ndi x ndi
         CalcEulerianStrain(LogStrain, dgrad1);

         // populate STRAN (symmetric)
         // ------------------------------------------------------------------
         // We use Voigt notation: (11, 22, 33, 23, 13, 12)
         //
         // ABAQUS USES:
         // (11, 22, 33, 12, 13, 23)
         // ------------------------------------------------------------------
         stran[0] = LogStrain(0, 0);
         stran[1] = LogStrain(1, 1);
         stran[2] = LogStrain(2, 2);
         stran[3] = 2 * LogStrain(0, 1);
         stran[4] = 2 * LogStrain(0, 2);
         stran[5] = 2 * LogStrain(1, 2);

         // compute incremental strain, DSTRAN
         DenseMatrix dLogStrain;
         dLogStrain.SetSize(ndi);
         CalcEulerianStrainIncr(dLogStrain, incr_dgrad);

         // populate DSTRAN (symmetric)
         // ------------------------------------------------------------------
         // We use Voigt notation: (11, 22, 33, 23, 13, 12)
         //
         // ABAQUS USES:
         // (11, 22, 33, 12, 13, 23)
         // ------------------------------------------------------------------
         dstran[0] = dLogStrain(0, 0);
         dstran[1] = dLogStrain(1, 1);
         dstran[2] = dLogStrain(2, 2);
         dstran[3] = 2 * dLogStrain(0, 1);
         dstran[4] = 2 * dLogStrain(0, 2);
         dstran[5] = 2 * dLogStrain(1, 2);


         // call c++ wrapper of umat routine
         CallUmat(&stress[0], statev.HostReadWrite(), &ddsdde[0], &sse, &spd, &scd, &rpl,
                  ddsdt, drplde, &drpldt, &stran[0], &dstran[0], time,
                  &deltaTime, &tempk, &dtemp, &predef, &dpred, &cmname,
                  &ndi, &nshr, &ntens, &nstatv, props.HostReadWrite(), &nprops, &coords[0],
                  drot, &pnewdt, &celent, &dfgrd0[0], &dfgrd1[0], &noel, &npt,
                  &layer, &kspt, &kstep, &kinc);

         // Due to how Abaqus has things ordered we need to swap the 4th and 6th columns
         // and rows with one another for our C_stiffness matrix.
         int j = 3;
         // We could probably just replace this with a std::swap operation...
         for (int i = 0; i < 6; i++) {
            std::swap(ddsdde[(6 * i) + j], ddsdde[(6 * i) + 5]);
         }

         for (int i = 0; i < 6; i++) {
            std::swap(ddsdde[(6 * j) + i], ddsdde[(6 * 5) + i]);
         }

         for (int i = 0; i < 36; i++) {
            ddsdde[i] *= deltaTime;
         }

         // set the material stiffness on the model
         // UPDATED: This method now uses accessor methods to get QuadratureFunctions from SimulationState
         SetQFData(local_elemID, ipID, ddsdde, m_sim_state.GetQuadratureFunction("tangent_stiffness", m_region));

         // set the updated stress on the model. Have to convert from Abaqus
         // ordering to Voigt notation ordering
         // ------------------------------------------------------------------
         // We use Voigt notation: (11, 22, 33, 23, 13, 12)
         //
         // ABAQUS USES:
         // (11, 22, 33, 12, 13, 23)
         // ------------------------------------------------------------------
         stressTemp2[0] = stress[0];
         stressTemp2[1] = stress[1];
         stressTemp2[2] = stress[2];
         stressTemp2[3] = stress[5];
         stressTemp2[4] = stress[4];
         stressTemp2[5] = stress[3];

         // UPDATED: This method now uses accessor methods to get QuadratureFunctions from SimulationState
         SetQFData(local_elemID, ipID, stressTemp2, m_sim_state.GetQuadratureFunction("cauchy_stress_end", m_region));

         // set the updated statevars
         // UPDATED: This method now uses accessor methods to get QuadratureFunctions from SimulationState
         SetQFData(local_elemID, ipID, statev.HostReadWrite(), m_sim_state.GetQuadratureFunction("state_vars_end", m_region));
      }
   }

   auto global_stress = m_sim_state.GetQuadratureFunction("cauchy_stress_end");
   auto stress_final = m_sim_state.GetQuadratureFunction("cauchy_stress_end", m_region);
   stress_final->FillQuadratureFunction(*global_stress);

   auto global_tangent_stiffness = m_sim_state.GetQuadratureFunction("tangent_stiffness");
   auto matGrad_qf = m_sim_state.GetQuadratureFunction("tangent_stiffness", m_region);
   matGrad_qf->FillQuadratureFunction(*global_tangent_stiffness);

   // Unload library if using LOAD_ON_SETUP strategy
   if (use_dynamic_loading_ && load_strategy_ == DynamicUmatLoader::LoadStrategy::LOAD_ON_SETUP) {
      UnloadUmatLibrary();
   }
}

bool AbaqusUmatModel::SetUmatLibrary(const std::string& library_path, 
                                     DynamicUmatLoader::LoadStrategy strategy) {
   // Unload current library if loaded
   if (use_dynamic_loading_) {
      UnloadUmatLibrary();
   }

   umat_library_path_ = library_path;
   load_strategy_ = strategy;
   use_dynamic_loading_ = !library_path.empty();
   umat_function_ = nullptr;

   // Load immediately if using PERSISTENT strategy
   if (use_dynamic_loading_ && strategy == DynamicUmatLoader::LoadStrategy::PERSISTENT) {
      return LoadUmatLibrary();
   }

   return true;
}

bool AbaqusUmatModel::ReloadUmatLibrary() {
   if (!use_dynamic_loading_) {
      return false;
   }
   
   // Force unload and reload
   DynamicUmatLoader::UnloadUmat(umat_library_path_);
   umat_function_ = nullptr;
   
   return LoadUmatLibrary();
}

bool AbaqusUmatModel::LoadUmatLibrary() {
   if (!use_dynamic_loading_ || umat_function_ != nullptr) {
      return true; // Already loaded or not using dynamic loading
   }
   
   umat_function_ = DynamicUmatLoader::LoadUmat(umat_library_path_, load_strategy_);
   if (!umat_function_) {
      std::cerr << "Failed to load UMAT library: " << umat_library_path_ << std::endl;
      return false;
   }
   
   std::cout << "Successfully loaded UMAT library for region " << m_region 
             << ": " << umat_library_path_ << std::endl;
   return true;
}

void AbaqusUmatModel::UnloadUmatLibrary() {
   if (use_dynamic_loading_ && !umat_library_path_.empty()) {
      DynamicUmatLoader::UnloadUmat(umat_library_path_);
      umat_function_ = nullptr;
   }
}

void AbaqusUmatModel::CallUmat(double *stress, double *statev, double *ddsdde,
                               double *sse, double *spd, double *scd, double *rpl,
                               double *ddsdt, double *drplde, double *drpldt,
                               double *stran, double *dstran, double *time,
                               double *deltaTime, double *tempk, double *dtemp, double *predef,
                               double *dpred, double *cmname, int *ndi, int *nshr, int *ntens,
                               int *nstatv, double *props, int *nprops, double *coords,
                               double *drot, double *pnewdt, double *celent,
                               double *dfgrd0, double *dfgrd1, int *noel, int *npt,
                               int *layer, int *kspt, int *kstep, int *kinc) {

   if (use_dynamic_loading_) {
      // Use dynamically loaded function
      if (!umat_function_) {
            throw std::runtime_error("UMAT function not loaded for library: " + umat_library_path_);
      }
      umat_function_(stress, statev, ddsdde, sse, spd, scd, rpl,
                     ddsdt, drplde, drpldt, stran, dstran, time, deltaTime,
                     tempk, dtemp, predef, dpred, cmname, ndi, nshr, ntens,
                     nstatv, props, nprops, coords, drot, pnewdt, celent,
                     dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc);
   } else {
      // Use statically linked function (from userumat.h)
      umat_call(stress, statev, ddsdde, sse, spd, scd, rpl,
               ddsdt, drplde, drpldt, stran, dstran, time, deltaTime,
               tempk, dtemp, predef, dpred, cmname, ndi, nshr, ntens,
               nstatv, props, nprops, coords, drot, pnewdt, celent,
               dfgrd0, dfgrd1, noel, npt, layer, kspt, kstep, kinc);
   }
}

// UNCHANGED: This method doesn't access QuadratureFunctions
void AbaqusUmatModel::CalcElemLength(const double elemVol)
{
   // It can also be approximated as the cube root of the element's volume.
   // I think this one might be a little nicer to use because for distorted elements
   // you might not want the largest length.
   // According to https://abaqus-docs.mit.edu/2017/English/SIMACAEKEYRefMap/simakey-r-characteristiclength.htm
   // it looks like this might be the right way to do it...
   // although this does change from integration to integration point
   // since we're using the determinate instead of the actual volume. However,
   // it should be good enough for our needs...
   elemLength = cbrt(elemVol);

   return;
}