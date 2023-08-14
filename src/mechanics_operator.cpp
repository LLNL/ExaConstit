
#include "mechanics_operator.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_log.hpp"
#include "mechanics_ecmech.hpp"
#include "mechanics_kernels.hpp"
#include "RAJA/RAJA.hpp"
#include "ECMech_const.h"
#include <iostream>

using namespace mfem;


NonlinearMechOperator::NonlinearMechOperator(ParFiniteElementSpace &fes,
                                             Array<int> &ess_bdr,
                                             Array2D<bool> &ess_bdr_comp,
                                             ExaOptions &options,
                                             QuadratureFunction &q_matVars0,
                                             QuadratureFunction &q_matVars1,
                                             QuadratureFunction &q_sigma0,
                                             QuadratureFunction &q_sigma1,
                                             QuadratureFunction &q_matGrad,
                                             QuadratureFunction &q_kinVars0,
                                             QuadratureFunction &q_vonMises,
                                             ParGridFunction &ref_crds,
                                             ParGridFunction &beg_crds,
                                             ParGridFunction &end_crds,
                                             Vector &matProps,
                                             int nStateVars)
   : NonlinearForm(&fes), fe_space(fes), x_ref(ref_crds), x_cur(end_crds), ess_bdr_comps(ess_bdr_comp)
{
   CALI_CXX_MARK_SCOPE("mechop_class_setup");
   Vector * rhs;
   rhs = NULL;

   mech_type = options.mech_type;

   // Define the parallel nonlinear form
   Hform = new ParNonlinearForm(&fes);

   // Set the essential boundary conditions
   Hform->SetEssentialBC(ess_bdr, ess_bdr_comps, rhs);

   // Set the essential boundary conditions that we can store on our class
   SetEssentialBC(ess_bdr, ess_bdr_comps, rhs);

   assembly = options.assembly;

   bool partial_assembly = false;
   if (assembly == Assembly::PA) {
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
      if (options.integ_type == IntegrationType::FULL) {
         Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model)));
      }
      else if (options.integ_type == IntegrationType::BBAR) {
         Hform->AddDomainIntegrator(new ICExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model)));
      }

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
      else if (options.rtmodel == RTModel::GPU) {
         accel = ecmech::ExecutionStrategy::GPU;
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
            if (options.integ_type == IntegrationType::FULL) {
               Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<VoceFCCModel*>(model)));
            }
            else if (options.integ_type == IntegrationType::BBAR) {
               Hform->AddDomainIntegrator(new ICExaNLFIntegrator(dynamic_cast<VoceFCCModel*>(model)));
            }
         }
         else if (options.slip_type == SlipType::POWERVOCENL) {
            // Our class will initialize our deformation gradients and
            // our local shape function gradients which are taken with respect
            // to our initial mesh when 1st created.
            model = new VoceNLFCCModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                       &beg_crds, &end_crds,
                                       &matProps, options.nProps, nStateVars, options.temp_k, accel,
                                       partial_assembly);

            // Add the user defined integrator
            if (options.integ_type == IntegrationType::FULL) {
               Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<VoceNLFCCModel*>(model)));
            }
            else if (options.integ_type == IntegrationType::BBAR) {
               Hform->AddDomainIntegrator(new ICExaNLFIntegrator(dynamic_cast<VoceNLFCCModel*>(model)));
            }
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
            if (options.integ_type == IntegrationType::FULL) {
               Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<KinKMBalDDFCCModel*>(model)));
            }
            else if (options.integ_type == IntegrationType::BBAR) {
               Hform->AddDomainIntegrator(new ICExaNLFIntegrator(dynamic_cast<KinKMBalDDFCCModel*>(model)));
            }
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
            if (options.integ_type == IntegrationType::FULL) {
               Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<KinKMBalDDHCPModel*>(model)));
            }
            else if (options.integ_type == IntegrationType::BBAR) {
               Hform->AddDomainIntegrator(new ICExaNLFIntegrator(dynamic_cast<KinKMBalDDHCPModel*>(model)));
            }
         }
      }
      else if (options.xtal_type == XtalType::BCC) {
         // Now we find out what slip kinetics and hardening law were chosen.
         if (options.slip_type == SlipType::POWERVOCE) {
            // Our class will initialize our deformation gradients and
            // our local shape function gradients which are taken with respect
            // to our initial mesh when 1st created.
            model = new VoceBCCModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                     &beg_crds, &end_crds,
                                     &matProps, options.nProps, nStateVars, options.temp_k, accel,
                                     partial_assembly);

            // Add the user defined integrator
            if (options.integ_type == IntegrationType::FULL) {
               Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<VoceBCCModel*>(model)));
            }
            else if (options.integ_type == IntegrationType::BBAR) {
               Hform->AddDomainIntegrator(new ICExaNLFIntegrator(dynamic_cast<VoceBCCModel*>(model)));
            }
         }
         else if (options.slip_type == SlipType::POWERVOCENL) {
            // Our class will initialize our deformation gradients and
            // our local shape function gradients which are taken with respect
            // to our initial mesh when 1st created.
            model = new VoceNLBCCModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                       &beg_crds, &end_crds,
                                       &matProps, options.nProps, nStateVars, options.temp_k, accel,
                                       partial_assembly);

            // Add the user defined integrator
            if (options.integ_type == IntegrationType::FULL) {
               Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<VoceNLBCCModel*>(model)));
            }
            else if (options.integ_type == IntegrationType::BBAR) {
               Hform->AddDomainIntegrator(new ICExaNLFIntegrator(dynamic_cast<VoceNLBCCModel*>(model)));
            }
         }
         else if (options.slip_type == SlipType::MTSDD) {
            // Our class will initialize our deformation gradients and
            // our local shape function gradients which are taken with respect
            // to our initial mesh when 1st created.
            model = new KinKMbalDDBCCModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                           &beg_crds, &end_crds,
                                           &matProps, options.nProps, nStateVars, options.temp_k, accel,
                                           partial_assembly);

            // Add the user defined integrator
            if (options.integ_type == IntegrationType::FULL) {
               Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<KinKMbalDDBCCModel*>(model)));
            }
            else if (options.integ_type == IntegrationType::BBAR) {
               Hform->AddDomainIntegrator(new ICExaNLFIntegrator(dynamic_cast<KinKMbalDDBCCModel*>(model)));
            }
         }
      }
   }

   if (assembly == Assembly::PA) {
      Hform->SetAssemblyLevel(mfem::AssemblyLevel::PARTIAL, ElementDofOrdering::NATIVE);
      diag.SetSize(fe_space.GetTrueVSize(), Device::GetMemoryType());
      diag.UseDevice(true);
      diag = 1.0;
      prec_oper = new MechOperatorJacobiSmoother(diag, Hform->GetEssentialTrueDofs());
   }
   else if (assembly == Assembly::EA) {
      Hform->SetAssemblyLevel(mfem::AssemblyLevel::ELEMENT, ElementDofOrdering::NATIVE);
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
         DenseMatrix DSh;
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

void NonlinearMechOperator::UpdateEssTDofs(const Array<int> &ess_bdr)
{
   // Set the essential boundary conditions
   Hform->SetEssentialBC(ess_bdr, ess_bdr_comps, nullptr);
   // Set the essential boundary conditions that we can store on our class
   SetEssentialBC(ess_bdr, ess_bdr_comps, nullptr);
}

// compute: y = H(x,p)
void NonlinearMechOperator::Mult(const Vector &k, Vector &y) const
{
   CALI_CXX_MARK_SCOPE("mechop_Mult");
   // We first run a setup step before actually doing anything.
   // We'll want to move this outside of Mult() at some given point in time
   // and have it live in the NR solver itself or whatever solver
   // we're going to be using.
   Setup<true>(k);
   // We now perform our element vector operation.
   if (assembly == Assembly::PA) {
      CALI_CXX_MARK_SCOPE("mechop_PA_PreSetup");
      model->TransformMatGradTo4D();
   }
   CALI_MARK_BEGIN("mechop_mult_setup");
   // Assemble our operator
   Hform->Setup();
   CALI_MARK_END("mechop_mult_setup");
   CALI_MARK_BEGIN("mechop_mult_Mult");
   Hform->Mult(k, y);
   CALI_MARK_END("mechop_mult_Mult");
}

template<bool upd_crds>
void NonlinearMechOperator::Setup(const Vector &k) const
{
   CALI_CXX_MARK_SCOPE("mechop_setup");
   // Wanted to put this in the mechanics_solver.cpp file, but I would have needed to update
   // Solver class to use the NonlinearMechOperator instead of Operator class.
   // We now update our end coordinates based on the solved for velocity.
   if(upd_crds) {
      UpdateEndCoords(k);
   }

   // This performs the computation of the velocity gradient if needed,
   // det(J), material tangent stiffness matrix, state variable update,
   // stress update, and other stuff that might be needed in the integrators.

   const FiniteElement &el = *fe_space.GetFE(0);
   const int space_dims = el.GetDim();
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));;

   const int nqpts = ir->GetNPoints();
   const int ndofs = el.GetDof();
   const int nelems = fe_space.GetNE();

   SetupJacobianTerms();

   // We can now make the call to our material model set-up stage...
   // Everything else that we need should live on the class.
   // Within this function the model just needs to produce the Cauchy stress
   // and the material tangent matrix (d \sigma / d Vgrad_{sym})
   if (mech_type == MechType::UMAT) {
      model->ModelSetup(nqpts, nelems, space_dims, ndofs, el_jac, qpts_dshape, k);
   }
   else {
      // Takes in k vector and transforms into into our E-vector array
      P->Mult(k, px);
      elem_restrict_lex->Mult(px, el_x);
      model->ModelSetup(nqpts, nelems, space_dims, ndofs, el_jac, qpts_dshape, el_x);
   }
} // End of model setup

void NonlinearMechOperator::SetupJacobianTerms() const
{

   Mesh *mesh = fe_space.GetMesh();
   const FiniteElement &el = *fe_space.GetFE(0);
   const int space_dims = el.GetDim();
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

   const int nqpts = ir->GetNPoints();
   const int nelems = fe_space.GetNE();

   // We need to make sure these are deleted at the start of each iteration
   // since we have meshes that are constantly changing.
   mesh->DeleteGeometricFactors();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   // geom->J really isn't going to work for us as of right now. We could just reorder it
   // to the version that we want it to be in instead...

   const int DIM4 = 4;
   std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
   // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
   RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ space_dims, space_dims, nqpts, nelems } }, perm4);
   RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > jac_view(el_jac.ReadWrite(), layout_jacob);

   RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{ nqpts, space_dims, space_dims, nelems } }, perm4);
   RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > geom_j_view(geom->J.Read(), layout_geom);

   const int nqpts1 = nqpts;
   const int space_dims1 = space_dims;
   MFEM_FORALL(i, nelems,
   {
      const int nqpts_ = nqpts1;
      const int space_dims_ = space_dims1;
      for (int j = 0; j < nqpts_; j++) {
         for (int k = 0; k < space_dims_; k++) {
            for (int l = 0; l < space_dims_; l++) {
               jac_view(l, k, j, i) = geom_j_view(j, l, k, i);
            }
         }
      }
   });
}

void NonlinearMechOperator::CalculateDeformationGradient(mfem::QuadratureFunction &def_grad) const
{
   Mesh *mesh = fe_space.GetMesh();

   //Since we never modify our mesh nodes during this operations this is okay.
   mfem::GridFunction *nodes = const_cast<mfem::ParGridFunction*>(&x_ref); // set a nodes grid function to global current configuration
   int owns_nodes = 0;
   mesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes
   SetupJacobianTerms();

   const IntegrationRule *ir = &(IntRules.Get(fe_space.GetFE(0)->GetGeomType(), 2 * fe_space.GetFE(0)->GetOrder() + 1));;

   const int nqpts = ir->GetNPoints();
   const int ndofs = fe_space.GetFE(0)->GetDof();
   const int nelems = fe_space.GetNE();

   Vector x_true(fe_space.TrueVSize(), mfem::Device::GetMemoryType());

   x_cur.GetTrueDofs(x_true);
   // Takes in k vector and transforms into into our E-vector array
   P->Mult(x_true, px);
   elem_restrict_lex->Mult(px, el_x);

   def_grad = 0.0;
   exaconstit::kernel::grad_calc(nqpts, nelems, ndofs, el_jac.Read(), qpts_dshape.Read(), el_x.Read(), def_grad.ReadWrite());

   //We're returning our mesh nodes to the original object they were pointing to.
   //So, we need to cast away the const here.
   //We just don't want other functions outside this changing things.
   nodes = const_cast<mfem::ParGridFunction*>(&x_cur);
   mesh->SwapNodes(nodes, owns_nodes);
   //Delete the old geometric factors since they dealt with the original reference frame.
   mesh->DeleteGeometricFactors();

}

// Update the end coords used in our model
void NonlinearMechOperator::UpdateEndCoords(const Vector& vel) const
{
   model->UpdateEndCoords(vel);
}

// Compute the Jacobian from the nonlinear form
Operator &NonlinearMechOperator::GetGradient(const Vector &x) const
{
   CALI_CXX_MARK_SCOPE("mechop_getgrad");
   Jacobian = &Hform->GetGradient(x);
   // Reset our preconditioner operator aka recompute the diagonal for our jacobi.
   Jacobian->AssembleDiagonal(diag);
   return *Jacobian;
}

// Compute the Jacobian from the nonlinear form
Operator& NonlinearMechOperator::GetUpdateBCsAction(const Vector &k, const Vector &x, Vector &y) const
{

   CALI_CXX_MARK_SCOPE("mechop_GetUpdateBCsAction");
   // We first run a setup step before actually doing anything.
   // We'll want to move this outside of Mult() at some given point in time
   // and have it live in the NR solver itself or whatever solver
   // we're going to be using.
   Setup<false>(k);
   // We now perform our element vector operation.
   Vector resid(y); resid.UseDevice(true);
   Array<int> zero_tdofs;
   if (assembly == Assembly::PA) {
      CALI_CXX_MARK_SCOPE("mechop_PA_BC_PreSetup");
      model->TransformMatGradTo4D();
   }

   CALI_MARK_BEGIN("mechop_Hform_LocalGrad");
   Hform->Setup();
   Hform->SetEssentialTrueDofs(zero_tdofs);
   auto &loc_jacobian = Hform->GetGradient(x);
   loc_jacobian.Mult(x, y);
   Hform->SetEssentialTrueDofs(ess_tdof_list);
   Hform->Mult(k, resid);
   Jacobian = &Hform->GetGradient(x);
   CALI_MARK_END("mechop_Hform_LocalGrad");

   {
      auto I = ess_tdof_list.Read();
      auto size = ess_tdof_list.Size();
      auto Y = y.Write();
      // Need to get rid of all the constrained values here
      MFEM_FORALL(i, size, Y[I[i]] = 0.0; );
   }

   y += resid;
   return *Jacobian;
}

NonlinearMechOperator::~NonlinearMechOperator()
{
   delete model;
   delete Hform;
}
