
#include "mechanics_operator.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_log.hpp"
#include "mechanics_ecmech.hpp"
#include "RAJA/RAJA.hpp"
#include "ECMech_const.h"

using namespace mfem;


NonlinearMechOperator::NonlinearMechOperator(ParFiniteElementSpace &fes,
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
   : NonlinearForm(&fes), fe_space(fes)
{
   CALI_CXX_MARK_SCOPE("mechop_class_setup");
   Vector * rhs;
   rhs = NULL;

   mech_type = options.mech_type;

   // Define the parallel nonlinear form
   Hform = new ParNonlinearForm(&fes);

   // Set the essential boundary conditions
   Hform->SetEssentialBCPartial(ess_bdr, rhs);

   partial_assembly = false;
   if (options.assembly == Assembly::PA) {
      partial_assembly = true;
   }

   if (options.mech_type == MechType::UMAT) {
      // Our class will initialize our deformation gradients and
      // our local shape function gradients which are taken with respect
      // to our initial mesh when 1st created.
      model = new AbaqusUmatModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                  &q_kinVars0, &beg_crds, &end_crds,
                                  &matProps, options.nProps, nStateVars, &fes, partial_assembly);

      // Add the user defined integrator
      Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model)));
   }
   else if (options.mech_type == MechType::EXACMECH) {
      // Time to go through a nice switch field to pick out the correct model to be run...
      // Should probably figure a better way to do this in the future so this doesn't become
      // one giant switch yard. Multiphase materials will probably require a complete revamp of things...
      // First we check the xtal symmetry type
      ecmech::ExecutionStrategy accel = ecmech::ExecutionStrategy::CPU;

      if (options.rtmodel == RTModel::CPU) {
         accel = ecmech::ExecutionStrategy::CPU;
      }
      else if (options.rtmodel == RTModel::OPENMP) {
         accel = ecmech::ExecutionStrategy::OPENMP;
      }
      else if (options.rtmodel == RTModel::CUDA) {
         accel = ecmech::ExecutionStrategy::CUDA;
      }

      if (options.xtal_type == XtalType::FCC) {
         // Now we find out what slip kinetics and hardening law were chosen.
         if (options.slip_type == SlipType::POWERVOCE) {
            // Our class will initialize our deformation gradients and
            // our local shape function gradients which are taken with respect
            // to our initial mesh when 1st created.
            model = new VoceFCCModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                     &beg_crds, &end_crds,
                                     &matProps, options.nProps, nStateVars, options.temp_k, accel,
                                     partial_assembly);

            // Add the user defined integrator
            Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<VoceFCCModel*>(model)));
         }
         else if (options.slip_type == SlipType::MTSDD) {
            // Our class will initialize our deformation gradients and
            // our local shape function gradients which are taken with respect
            // to our initial mesh when 1st created.
            model = new KinKMBalDDFCCModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                           &beg_crds, &end_crds,
                                           &matProps, options.nProps, nStateVars, options.temp_k, accel,
                                           partial_assembly);

            // Add the user defined integrator
            Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<KinKMBalDDFCCModel*>(model)));
         }
      }
      else if (options.xtal_type == XtalType::HCP) {
         if (options.slip_type == SlipType::MTSDD) {
            // Our class will initialize our deformation gradients and
            // our local shape function gradients which are taken with respect
            // to our initial mesh when 1st created.
            model = new KinKMBalDDHCPModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                           &beg_crds, &end_crds,
                                           &matProps, options.nProps, nStateVars, options.temp_k, accel,
                                           partial_assembly);

            // Add the user defined integrator
            Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<KinKMBalDDHCPModel*>(model)));
         }
      }
   }

   if (options.assembly == Assembly::PA) {
      pa_oper = new PANonlinearMechOperatorGradExt(Hform, Hform->GetEssentialTrueDofs());
      diag.SetSize(fe_space.GetTrueVSize(), Device::GetMemoryType());
      diag.UseDevice(true);
      diag = 1.0;
      prec_oper = new MechOperatorJacobiSmoother(diag, Hform->GetEssentialTrueDofs());
   }

   // So, we're going to originally support non tensor-product type elements originally.
   const ElementDofOrdering ordering = ElementDofOrdering::NATIVE;
   // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
   elem_restrict_lex = fe_space.GetElementRestriction(ordering);

   el_x.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
   el_x.UseDevice(true);
   px.SetSize(P->Height(), Device::GetMemoryType());
   px.UseDevice(true);

   {
      const FiniteElement &el = *fe_space.GetFE(0);
      const int space_dims = el.GetDim();
      const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));;

      const int nqpts = ir->GetNPoints();
      const int ndofs = el.GetDof();
      const int nelems = fe_space.GetNE();

      el_jac.SetSize(space_dims * space_dims * nqpts * nelems, Device::GetMemoryType());
      el_jac.UseDevice(true);

      qpts_dshape.SetSize(nqpts * space_dims * ndofs, Device::GetMemoryType());
      qpts_dshape.UseDevice(true);
      {
         DenseMatrix DSh(ndofs, space_dims);
         const int offset = ndofs * space_dims;
         double *qpts_dshape_data = qpts_dshape.HostReadWrite();
         for (int i = 0; i < nqpts; i++) {
            const IntegrationPoint &ip = ir->IntPoint(i);
            DSh.UseExternalData(&qpts_dshape_data[offset * i], ndofs, space_dims);
            el.CalcDShape(ip, DSh);
         }
      }
   }

   // We'll probably want to eventually add a print settings into our option class that tells us whether
   // or not we're going to be printing this.

   model->setVonMisesPtr(&q_vonMises);
}

const Array<int> &NonlinearMechOperator::GetEssTDofList()
{
   return Hform->GetEssentialTrueDofs();
}

ExaModel *NonlinearMechOperator::GetModel() const
{
   return model;
}

// compute: y = H(x,p)
void NonlinearMechOperator::Mult(const Vector &k, Vector &y) const
{
   CALI_CXX_MARK_SCOPE("mechop_Mult");
   // We first run a setup step before actually doing anything.
   // We'll want to move this outside of Mult() at some given point in time
   // and have it live in the NR solver itself or whatever solver
   // we're going to be using.
   Setup(k);
   // We now perform our element vector operation.
   if (!partial_assembly) {
      CALI_CXX_MARK_SCOPE("mechop_HformMult");
      Hform->Mult(k, y);
   }
   else {
      CALI_MARK_BEGIN("mechop_PAsetup");
      model->TransformMatGradTo4D();
      // Assemble our operator
      pa_oper->Assemble();
      CALI_MARK_END("mechop_PAsetup");
      CALI_CXX_MARK_SCOPE("mechop_PAMult");
      pa_oper->MultVec(k, y);
   }
}

void NonlinearMechOperator::Setup(const Vector &k) const
{
   CALI_CXX_MARK_SCOPE("mechop_setup");
   // Wanted to put this in the mechanics_solver.cpp file, but I would have needed to update
   // Solver class to use the NonlinearMechOperator instead of Operator class.
   // We now update our end coordinates based on the solved for velocity.
   UpdateEndCoords(k);

   // This performs the computation of the velocity gradient if needed,
   // det(J), material tangent stiffness matrix, state variable update,
   // stress update, and other stuff that might be needed in the integrators.

   Mesh *mesh = fe_space.GetMesh();
   const FiniteElement &el = *fe_space.GetFE(0);
   const int space_dims = el.GetDim();
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));;

   const int nqpts = ir->GetNPoints();
   const int ndofs = el.GetDof();
   const int nelems = fe_space.GetNE();
   // We need to make sure these are deleted at the start of each iteration
   // since we have meshes that are constantly changing.
   mesh->DeleteGeometricFactors();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);

   // Takes in k vector and transforms into into our E-vector array
   P->Mult(k, px);
   elem_restrict_lex->Mult(px, el_x);

   // geom->J really isn't going to work for us as of right now. We could just reorder it
   // to the version that we want it to be in instead...

   const int DIM4 = 4;
   std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
   // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
   RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ space_dims, space_dims, nqpts, nelems } }, perm4);
   RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > jac_view(el_jac.ReadWrite(), layout_jacob);

   RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{ nqpts, space_dims, space_dims, nelems } }, perm4);
   RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > geom_j_view(geom->J.Read(), layout_geom);

   MFEM_FORALL(i, nelems,
   {
      for (int j = 0; j < nqpts; j++) {
         for (int k = 0; k < space_dims; k++) {
            for (int l = 0; l < space_dims; l++) {
               jac_view(l, k, j, i) = geom_j_view(j, l, k, i);
            }
         }
      }
   });

   // We can now make the call to our material model set-up stage...
   // Everything else that we need should live on the class.
   // Within this function the model just needs to produce the Cauchy stress
   // and the material tangent matrix (d \sigma / d Vgrad_{sym})

   if (mech_type == MechType::UMAT) {
      model->ModelSetup(nqpts, nelems, space_dims, ndofs, el_jac, qpts_dshape, k);
   }
   else {
      model->ModelSetup(nqpts, nelems, space_dims, ndofs, el_jac, qpts_dshape, el_x);
   }
} // End of model setup

// Update the end coords used in our model
void NonlinearMechOperator::UpdateEndCoords(const Vector& vel) const
{
   model->UpdateEndCoords(vel);
}

// Compute the Jacobian from the nonlinear form
Operator &NonlinearMechOperator::GetGradient(const Vector &x) const
{
   CALI_CXX_MARK_SCOPE("mechop_getgrad");
   if (!partial_assembly) {
      Jacobian = &Hform->GetGradient(x);
      return *Jacobian;
   }
   else {
      // model->TransformMatGradTo4D();
      //// Assemble our operator
      // pa_oper->Assemble();
      pa_oper->AssembleDiagonal(diag);
      // Reset our preconditioner operator aka recompute the diagonal for our jacobi.
      prec_oper->Setup(diag);
      return *pa_oper;
   }
}

NonlinearMechOperator::~NonlinearMechOperator()
{
   delete model;
   delete Hform;
   if (partial_assembly) {
      delete pa_oper;
      // This will be deleted in the system driver class
      // before the preconditioner is deleted.
      // delete prec_oper;
   }
}
