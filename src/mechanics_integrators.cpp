
#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_log.hpp"
#include "BCManager.hpp"
#include <math.h> // log
#include <algorithm>
#include <iostream> // cerr
#include "RAJA/RAJA.hpp"

using namespace mfem;
using namespace std;

// member functions for the ExaNLFIntegrator
double ExaNLFIntegrator::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun)
{
   // we are not interested in the element energy at this time
   (void) el;
   (void) Ttr;
   (void) elfun;

   return 0.0;
}

// Outside of the UMAT function calls this should be the function called
// to assemble our residual vectors.
void ExaNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, Vector &elvect)
{
   CALI_CXX_MARK_SCOPE("enlfi_assembleElemVec");
   int dof = el.GetDof(), dim = el.GetDim();

   DenseMatrix DSh, DS;
   DenseMatrix Jpt;
   DenseMatrix PMatI, PMatO;
   // This is our stress tensor
   DenseMatrix P(3);

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jpt.SetSize(dim);

   // PMatI would be our velocity in this case
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof * dim);

   // PMatO would be our residual vector
   elvect = 0.0;
   PMatO.UseExternalData(elvect.HostReadWrite(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (!ir) {
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1)); // must match quadrature space
   }

   for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);

      // compute Jacobian of the transformation
      Jpt = Ttr.InverseJacobian(); // Jrt = dxi / dX

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jpt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX

      double stress[6];
      model->GetElementStress(Ttr.ElementNo, i, false, stress, 6);
      // Could probably later have this only set once...
      // Would reduce the number mallocs that we're doing and
      // should potentially provide a small speed boost.
      P(0, 0) = stress[0];
      P(1, 1) = stress[1];
      P(2, 2) = stress[2];
      P(1, 2) = stress[3];
      P(0, 2) = stress[4];
      P(0, 1) = stress[5];

      P(2, 1) = P(1, 2);
      P(2, 0) = P(0, 2);
      P(1, 0) = P(0, 1);

      DS *= (Ttr.Weight() * ip.weight);
      AddMult(DS, P, PMatO);
   }

   return;
}

void ExaNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector & /*elfun*/, DenseMatrix &elmat)
{
   CALI_CXX_MARK_SCOPE("enlfi_assembleElemGrad");
   int dof = el.GetDof(), dim = el.GetDim();

   DenseMatrix DSh, DS, Jrt;

   // Now time to start assembling stuff
   DenseMatrix grad_trans, temp;
   DenseMatrix tan_stiff;

   int ngrad_dim2 = 36;
   double matGrad[ngrad_dim2];
   // Delta in our timestep
   double dt = model->GetModelDt();

   // temp1 is now going to become the transpose Bmatrix as seen in
   // [B^t][tan_stiff][B]
   grad_trans.SetSize(dof * dim, 6);
   // We need a temp matrix to store our first matrix results as seen in here
   temp.SetSize(6, dof * dim);

   tan_stiff.UseExternalData(&matGrad[0], 6, 6);

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   elmat.SetSize(dof * dim);

   const IntegrationRule *ir = IntRule;
   if (!ir) {
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1)); // <--- must match quadrature space
   }

   elmat = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);

      model->GetElementMatGrad(Ttr.ElementNo, i, matGrad, ngrad_dim2);
      // temp1 is B^t
      model->GenerateGradMatrix(DS, grad_trans);
      // We multiple our quadrature wts here to our tan_stiff matrix
      tan_stiff *= dt * ip.weight * Ttr.Weight();
      // We use kgeom as a temporary matrix
      // kgeom = [Cstiff][B]
      MultABt(tan_stiff, grad_trans, temp);
      // We now add our [B^t][kgeom] product to our tangent stiffness matrix that
      // we want to output to our material tangent stiffness matrix
      AddMult(grad_trans, temp, elmat);
   }

   return;
}

// This performs the assembly step of our RHS side of our system:
// f_ik =
void ExaNLFIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   CALI_CXX_MARK_SCOPE("enlfi_assemblePA");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   space_dims = el.GetDim();
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

   nqpts = ir->GetNPoints();
   nnodes = el.GetDof();
   nelems = fes.GetNE();

   auto W = ir->GetWeights().Read();
   geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);

   // return a pointer to beginning step stress. This is used for output visualization
   QuadratureFunction *stress_end = model->GetStress1();

   if ((space_dims == 1) || (space_dims == 2)) {
      MFEM_ABORT("Dimensions of 1 or 2 not supported.");
   }
   else {
      const int dim = 3;

      if (grad.Size() != (nqpts * dim * nnodes)) {
         grad.SetSize(nqpts * dim * nnodes, mfem::Device::GetMemoryType());
         {
            DenseMatrix DSh;
            const int offset = nnodes * dim;
            double *qpts_dshape_data = grad.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
               const IntegrationPoint &ip = ir->IntPoint(i);
               DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, dim);
               el.CalcDShape(ip, DSh);
            }
         }
         grad.UseDevice(true);
      }

      // geom->J really isn't going to work for us as of right now. We could just reorder it
      // to the version that we want it to be in instead...
      if (jacobian.Size() != (dim * dim * nqpts * nelems)) {
         jacobian.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
         jacobian.UseDevice(true);
      }

      if (dmat.Size() != (dim * dim * nqpts * nelems)) {
         dmat.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
         dmat.UseDevice(true);
      }

      const int DIM2 = 2;
      const int DIM3 = 3;
      const int DIM4 = 4;
      std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };

      RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
      RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.ReadWrite(), layout_jacob);

      RAJA::Layout<DIM3> layout_stress = RAJA::make_permuted_layout({{ 2 * dim, nqpts, nelems } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > S(stress_end->ReadWrite(),
                                                                           layout_stress);

      RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > D(dmat.ReadWrite(), layout_jacob);

      RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{ nqpts, dim, dim, nelems } }, perm4);
      RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > geom_j_view(geom->J.Read(), layout_geom);

      MFEM_FORALL(i, nelems, {
         for (int j = 0; j < nqpts; j++) {
            for (int k = 0; k < dim; k++) {
               for (int l = 0; l < dim; l++) {
                  J(l, k, j, i) = geom_j_view(j, l, k, i);
               }
            }
         }
      });

      MFEM_FORALL(i_elems, nelems, {
         double adj[dim * dim];
         // So, we're going to say this view is constant however we're going to mutate the values only in
         // that one scoped section for the quadrature points.
         // adj is actually in row major memory order but if we set this to col. major than this view
         // will act as the transpose of adj A which is what we want.
         RAJA::View<const double, RAJA::Layout<DIM2> > A(&adj[0], dim, dim);
         // RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > A(&adj[0], layout_adj);
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around with us for
            // the adjugate term.
            {
               const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
               const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
               const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
               const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
               const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
               const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
               const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
               const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
               const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
               // adj(J)
               adj[0] = (J22 * J33) - (J23 * J32); // 0,0
               adj[1] = (J32 * J13) - (J12 * J33); // 0,1
               adj[2] = (J12 * J23) - (J22 * J13); // 0,2
               adj[3] = (J31 * J23) - (J21 * J33); // 1,0
               adj[4] = (J11 * J33) - (J13 * J31); // 1,1
               adj[5] = (J21 * J13) - (J11 * J23); // 1,2
               adj[6] = (J21 * J32) - (J31 * J22); // 2,0
               adj[7] = (J31 * J12) - (J11 * J32); // 2,1
               adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }

            D(0, 0, j_qpts, i_elems) = S(0, j_qpts, i_elems) * A(0, 0) +
                                       S(5, j_qpts, i_elems) * A(0, 1) +
                                       S(4, j_qpts, i_elems) * A(0, 2);
            D(1, 0, j_qpts, i_elems) = S(0, j_qpts, i_elems) * A(1, 0) +
                                       S(5, j_qpts, i_elems) * A(1, 1) +
                                       S(4, j_qpts, i_elems) * A(1, 2);
            D(2, 0, j_qpts, i_elems) = S(0, j_qpts, i_elems) * A(2, 0) +
                                       S(5, j_qpts, i_elems) * A(2, 1) +
                                       S(4, j_qpts, i_elems) * A(2, 2);

            D(0, 1, j_qpts, i_elems) = S(5, j_qpts, i_elems) * A(0, 0) +
                                       S(1, j_qpts, i_elems) * A(0, 1) +
                                       S(3, j_qpts, i_elems) * A(0, 2);
            D(1, 1, j_qpts, i_elems) = S(5, j_qpts, i_elems) * A(1, 0) +
                                       S(1, j_qpts, i_elems) * A(1, 1) +
                                       S(3, j_qpts, i_elems) * A(1, 2);
            D(2, 1, j_qpts, i_elems) = S(5, j_qpts, i_elems) * A(2, 0) +
                                       S(1, j_qpts, i_elems) * A(2, 1) +
                                       S(3, j_qpts, i_elems) * A(2, 2);

            D(0, 2, j_qpts, i_elems) = S(4, j_qpts, i_elems) * A(0, 0) +
                                       S(3, j_qpts, i_elems) * A(0, 1) +
                                       S(2, j_qpts, i_elems) * A(0, 2);
            D(1, 2, j_qpts, i_elems) = S(4, j_qpts, i_elems) * A(1, 0) +
                                       S(3, j_qpts, i_elems) * A(1, 1) +
                                       S(2, j_qpts, i_elems) * A(1, 2);
            D(2, 2, j_qpts, i_elems) = S(4, j_qpts, i_elems) * A(2, 0) +
                                       S(3, j_qpts, i_elems) * A(2, 1) +
                                       S(2, j_qpts, i_elems) * A(2, 2);
         } // End of doing J_{ij}\sigma_{jk} / nqpts loop
      }); // End of elements
      MFEM_FORALL(i_elems, nelems, {
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            for (int i = 0; i < dim; i++) {
               for (int j = 0; j < dim; j++) {
                  D(j, i, j_qpts, i_elems) *= W[j_qpts];
               }
            }
         }
      });
   } // End of if statement
}

// In the below function we'll be applying the below action on our material
// tangent matrix C^{tan} at each quadrature point as:
// D_{ijkm} = 1 / det(J) * w_{qpt} * adj(J)^T_{ij} C^{tan}_{ijkl} adj(J)_{lm}
// where D is our new 4th order tensor, J is our jacobian calculated from the
// mesh geometric factors, and adj(J) is the adjugate of J.
void ExaNLFIntegrator::AssemblePAGrad(const FiniteElementSpace &fes)
{
   CALI_CXX_MARK_SCOPE("enlfi_assemblePAG");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   space_dims = el.GetDim();
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

   nqpts = ir->GetNPoints();
   nnodes = el.GetDof();
   nelems = fes.GetNE();
   auto W = ir->GetWeights().Read();

   if ((space_dims == 1) || (space_dims == 2)) {
      MFEM_ABORT("Dimensions of 1 or 2 not supported.");
   }
   else {
      const int dim = 3;

      if (grad.Size() != (nqpts * dim * nnodes)) {
         grad.SetSize(nqpts * dim * nnodes, mfem::Device::GetMemoryType());
         {
            DenseMatrix DSh;
            const int offset = nnodes * dim;
            double *qpts_dshape_data = grad.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
               const IntegrationPoint &ip = ir->IntPoint(i);
               DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, dim);
               el.CalcDShape(ip, DSh);
            }
         }
         grad.UseDevice(true);
      }

      // geom->J really isn't going to work for us as of right now. We could just reorder it
      // to the version that we want it to be in instead...
      if (jacobian.Size() != (dim * dim * nqpts * nelems)) {
         jacobian.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
         jacobian.UseDevice(true);

         geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);

         const int DIM4 = 4;
         std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };

         RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
         RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.ReadWrite(), layout_jacob);

         RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{ nqpts, dim, dim, nelems } }, perm4);
         RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > geom_j_view(geom->J.Read(), layout_geom);

         MFEM_FORALL(i, nelems, {
            for (int j = 0; j < nqpts; j++) {
               for (int k = 0; k < dim; k++) {
                  for (int l = 0; l < dim; l++) {
                     J(l, k, j, i) = geom_j_view(j, l, k, i);
                  }
               }
            }
         });
      }

      if (pa_dmat.Size() != (dim * dim * dim * dim * nqpts * nelems)) {
         pa_dmat.SetSize(dim * dim * dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
         pa_dmat.UseDevice(true);
      }

      pa_dmat = 0.0;

      const int DIM2 = 2;
      const int DIM4 = 4;
      const int DIM6 = 6;
      std::array<RAJA::idx_t, DIM6> perm6 {{ 5, 4, 3, 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };

      // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

      RAJA::Layout<DIM6> layout_4Dtensor = RAJA::make_permuted_layout({{ dim, dim, dim, dim, nqpts, nelems } }, perm6);
      RAJA::View<const double, RAJA::Layout<DIM6, RAJA::Index_type, 0> > C(model->GetMTanData(), layout_4Dtensor);
      // Swapped over to row order since it makes sense in later applications...
      // Should make C row order as well for PA operations
      RAJA::View<double, RAJA::Layout<DIM6> > D(pa_dmat.ReadWrite(), nelems, nqpts, dim, dim, dim, dim);

      RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
      RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.ReadWrite(), layout_jacob);

      RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{ dim, dim } }, perm2);

      double dt = model->GetModelDt();
      // This loop we'll want to parallelize the rest are all serial for now.
      MFEM_FORALL(i_elems, nelems, {
         double adj[dim * dim];
         double c_detJ;
         // So, we're going to say this view is constant however we're going to mutate the values only in
         // that one scoped section for the quadrature points.
         RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > A(&adj[0], layout_adj);
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around with us for
            // the adjugate term.
            {
               const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
               const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
               const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
               const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
               const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
               const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
               const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
               const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
               const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                   /* */ J21 * (J12 * J33 - J32 * J13) +
                                   /* */ J31 * (J12 * J23 - J22 * J13);
               c_detJ = 1.0 / detJ * W[j_qpts] * dt;
               // adj(J)
               adj[0] = (J22 * J33) - (J23 * J32); // 0,0
               adj[1] = (J32 * J13) - (J12 * J33); // 0,1
               adj[2] = (J12 * J23) - (J22 * J13); // 0,2
               adj[3] = (J31 * J23) - (J21 * J33); // 1,0
               adj[4] = (J11 * J33) - (J13 * J31); // 1,1
               adj[5] = (J21 * J13) - (J11 * J23); // 1,2
               adj[6] = (J21 * J32) - (J31 * J22); // 2,0
               adj[7] = (J31 * J12) - (J11 * J32); // 2,1
               adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            // Unrolled part of the loops just so we wouldn't have so many nested ones.
            // If we were to get really ambitious we could eliminate also the m indexed
            // loop...
            for (int n = 0; n < dim; n++) {
               for (int m = 0; m < dim; m++) {
                  for (int l = 0; l < dim; l++) {
                     D(i_elems, j_qpts, 0, 0, l, n) += (A(0, 0) * C(0, 0, l, m, j_qpts, i_elems) +
                                                        A(1, 0) * C(1, 0, l, m, j_qpts, i_elems) +
                                                        A(2, 0) * C(2, 0, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 0, 1, l, n) += (A(0, 0) * C(0, 1, l, m, j_qpts, i_elems) +
                                                        A(1, 0) * C(1, 1, l, m, j_qpts, i_elems) +
                                                        A(2, 0) * C(2, 1, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 0, 2, l, n) += (A(0, 0) * C(0, 2, l, m, j_qpts, i_elems) +
                                                        A(1, 0) * C(1, 2, l, m, j_qpts, i_elems) +
                                                        A(2, 0) * C(2, 2, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 1, 0, l, n) += (A(0, 1) * C(0, 0, l, m, j_qpts, i_elems) +
                                                        A(1, 1) * C(1, 0, l, m, j_qpts, i_elems) +
                                                        A(2, 1) * C(2, 0, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 1, 1, l, n) += (A(0, 1) * C(0, 1, l, m, j_qpts, i_elems) +
                                                        A(1, 1) * C(1, 1, l, m, j_qpts, i_elems) +
                                                        A(2, 1) * C(2, 1, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 1, 2, l, n) += (A(0, 1) * C(0, 2, l, m, j_qpts, i_elems) +
                                                        A(1, 1) * C(1, 2, l, m, j_qpts, i_elems) +
                                                        A(2, 1) * C(2, 2, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 2, 0, l, n) += (A(0, 2) * C(0, 0, l, m, j_qpts, i_elems) +
                                                        A(1, 2) * C(1, 0, l, m, j_qpts, i_elems) +
                                                        A(2, 2) * C(2, 0, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 2, 1, l, n) += (A(0, 2) * C(0, 1, l, m, j_qpts, i_elems) +
                                                        A(1, 2) * C(1, 1, l, m, j_qpts, i_elems) +
                                                        A(2, 2) * C(2, 1, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 2, 2, l, n) += (A(0, 2) * C(0, 2, l, m, j_qpts, i_elems) +
                                                        A(1, 2) * C(1, 2, l, m, j_qpts, i_elems) +
                                                        A(2, 2) * C(2, 2, l, m, j_qpts, i_elems)) * A(m, n);
                  }
               }
            } // End of Dikln = adj(J)_{ji} C_{jklm} adj(J)_{mn} loop

            // Unrolled part of the loops just so we wouldn't have so many nested ones.
            for (int n = 0; n < dim; n++) {
               for (int l = 0; l < dim; l++) {
                  D(i_elems, j_qpts, l, n, 0, 0) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 0, 1) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 0, 2) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 1, 0) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 1, 1) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 1, 2) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 2, 0) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 2, 1) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 2, 2) *= c_detJ;
               }
            } // End of D_{ijkl} *= 1/det(J) * w_{qpt} loop
         } // End of quadrature loop
      }); // End of Elements loop
   } // End of else statement
}

// Here we're applying the following action operation using the assembled "D" 2nd order
// tensor found above:
// y_{ik} = \nabla_{ij}\phi^T_{\epsilon} D_{jk}
void ExaNLFIntegrator::AddMultPA(const mfem::Vector & /*x*/, mfem::Vector &y) const
{
   CALI_CXX_MARK_SCOPE("enlfi_amPAV");
   if ((space_dims == 1) || (space_dims == 2)) {
      MFEM_ABORT("Dimensions of 1 or 2 not supported.");
   }
   else {
      const int dim = 3;
      const int DIM3 = 3;
      const int DIM4 = 4;

      std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
      // Swapped over to row order since it makes sense in later applications...
      // Should make C row order as well for PA operations
      RAJA::Layout<DIM4> layout_tensor = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
      RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > D(dmat.Read(), layout_tensor);
      // Our field variables that are inputs and outputs
      RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{ nnodes, dim, nelems } }, perm3);
      RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Y(y.ReadWrite(), layout_field);
      // Transpose of the local gradient variable
      RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{ nnodes, dim, nqpts } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Gt(grad.Read(), layout_grads);

      MFEM_FORALL(i_elems, nelems, {
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            for (int k = 0; k < dim; k++) {
               for (int j = 0; j < dim; j++) {
                  for (int i = 0; i < nnodes; i++) {
                     Y(i, k, i_elems) += Gt(i, j, j_qpts) * D(j, k, j_qpts, i_elems);
                  }
               }
            } // End of the final action of Y_{ik} += Gt_{ij} T_{jk}
         } // End of nQpts
      }); // End of nelems
   } // End of if statement
}

// Here we're applying the following action operation using the assembled "D" 4th order
// tensor found above:
// y_{ik} = \nabla_{ij}\phi^T_{\epsilon} D_{jklm} \nabla_{mn}\phi_{\epsilon} x_{nl}
void ExaNLFIntegrator::AddMultPAGrad(const mfem::Vector &x, mfem::Vector &y)
{
   CALI_CXX_MARK_SCOPE("enlfi_amPAG");
   if ((space_dims == 1) || (space_dims == 2)) {
      MFEM_ABORT("Dimensions of 1 or 2 not supported.");
   }
   else {
      const int dim = 3;
      const int DIM2 = 2;
      const int DIM3 = 3;
      const int DIM6 = 6;

      std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
      // Swapped over to row order since it makes sense in later applications...
      // Should make C row order as well for PA operations
      RAJA::View<double, RAJA::Layout<DIM6> > D(pa_dmat.ReadWrite(), nelems, nqpts, dim, dim, dim, dim);
      // Our field variables that are inputs and outputs
      RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{ nnodes, dim, nelems } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > X(x.Read(), layout_field);
      RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Y(y.ReadWrite(), layout_field);
      // Transpose of the local gradient variable
      RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{ nnodes, dim, nqpts } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Gt(grad.Read(), layout_grads);

      // View for our temporary 2d array
      RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{ dim, dim } }, perm2);
      MFEM_FORALL(i_elems, nelems, {
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            double T[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int i = 0; i < dim; i++) {
               for (int j = 0; j < dim; j++) {
                  for (int k = 0; k < nnodes; k++) {
                     T[0] += D(i_elems, j_qpts, 0, 0, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[1] += D(i_elems, j_qpts, 1, 0, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[2] += D(i_elems, j_qpts, 2, 0, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[3] += D(i_elems, j_qpts, 0, 1, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[4] += D(i_elems, j_qpts, 1, 1, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[5] += D(i_elems, j_qpts, 2, 1, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[6] += D(i_elems, j_qpts, 0, 2, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[7] += D(i_elems, j_qpts, 1, 2, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[8] += D(i_elems, j_qpts, 2, 2, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                  }
               }
            } // End of doing tensor contraction of D_{jkmo}G_{op}X_{pm}

            RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > Tview(&T[0], layout_adj);
            for (int k = 0; k < dim; k++) {
               for (int j = 0; j < dim; j++) {
                  for (int i = 0; i < nnodes; i++) {
                     Y(i, k, i_elems) += Gt(i, j, j_qpts) * Tview(j, k);
                  }
               }
            } // End of the final action of Y_{ik} += Gt_{ij} T_{jk}
         } // End of nQpts
      }); // End of nelems
   } // End of if statement
}

// This assembles the diagonal of our LHS which can be used as a preconditioner
void ExaNLFIntegrator::AssembleDiagonalPA(Vector &y)
{
   CALI_CXX_MARK_SCOPE("enlfi_assembleDiagonalPA");

   const IntegrationRule &ir = model->GetMatGrad()->GetSpace()->GetElementIntRule(0);
   auto W = ir.GetWeights().Read();

   if ((space_dims == 1) || (space_dims == 2)) {
      MFEM_ABORT("Dimensions of 1 or 2 not supported.");
   }
   else {
      const int dim = 3;

      const int DIM2 = 2;
      const int DIM3 = 3;
      const int DIM4 = 4;

      std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };

      // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

      RAJA::Layout<DIM4> layout_tensor = RAJA::make_permuted_layout({{ 2 * dim, 2 * dim, nqpts, nelems } }, perm4);
      RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > K(model->GetMatGrad()->Read(), layout_tensor);

      // Our field variables that are inputs and outputs
      RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{ nnodes, dim, nelems } }, perm3);
      RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Y(y.ReadWrite(), layout_field);

      RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
      RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.ReadWrite(), layout_jacob);

      RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{ dim, dim } }, perm2);

      RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{ nnodes, dim, nqpts } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Gt(grad.Read(), layout_grads);

      double dt = model->GetModelDt();
      // This loop we'll want to parallelize the rest are all serial for now.
      MFEM_FORALL(i_elems, nelems, {
         double adj[dim * dim];
         double c_detJ;
         // So, we're going to say this view is constant however we're going to mutate the values only in
         // that one scoped section for the quadrature points.
         RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > A(&adj[0], layout_adj);
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around with us for
            // the adjugate term.
            {
               const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
               const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
               const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
               const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
               const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
               const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
               const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
               const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
               const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                   /* */ J21 * (J12 * J33 - J32 * J13) +
                                   /* */ J31 * (J12 * J23 - J22 * J13);
               c_detJ = 1.0 / detJ * W[j_qpts] * dt;
               // adj(J)
               adj[0] = (J22 * J33) - (J23 * J32); // 0,0
               adj[1] = (J32 * J13) - (J12 * J33); // 0,1
               adj[2] = (J12 * J23) - (J22 * J13); // 0,2
               adj[3] = (J31 * J23) - (J21 * J33); // 1,0
               adj[4] = (J11 * J33) - (J13 * J31); // 1,1
               adj[5] = (J21 * J13) - (J11 * J23); // 1,2
               adj[6] = (J21 * J32) - (J31 * J22); // 2,0
               adj[7] = (J31 * J12) - (J11 * J32); // 2,1
               adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            for (int knodes = 0; knodes < nnodes; knodes++) {
               const double bx = Gt(knodes, 0, j_qpts) * A(0, 0)
                                 + Gt(knodes, 1, j_qpts) * A(0, 1)
                                 + Gt(knodes, 2, j_qpts) * A(0, 2);

               const double by = Gt(knodes, 0, j_qpts) * A(1, 0)
                                 + Gt(knodes, 1, j_qpts) * A(1, 1)
                                 + Gt(knodes, 2, j_qpts) * A(1, 2);

               const double bz = Gt(knodes, 0, j_qpts) * A(2, 0)
                                 + Gt(knodes, 1, j_qpts) * A(2, 1)
                                 + Gt(knodes, 2, j_qpts) * A(2, 2);

               Y(knodes, 0, i_elems) += c_detJ * (bx * (bx * K(0, 0, j_qpts, i_elems)
                                                        + by * K(0, 5, j_qpts, i_elems)
                                                        + bz * K(0, 4, j_qpts, i_elems))
                                                  + by * (bx * K(5, 0, j_qpts, i_elems)
                                                          + by * K(5, 5, j_qpts, i_elems)
                                                          + bz * K(5, 4, j_qpts, i_elems))
                                                  + bz * (bx * K(4, 0, j_qpts, i_elems)
                                                          + by * K(4, 5, j_qpts, i_elems)
                                                          + bz * K(4, 4, j_qpts, i_elems)));

               Y(knodes, 1, i_elems) += c_detJ * (bx * (bx * K(5, 5, j_qpts, i_elems)
                                                        + by * K(5, 1, j_qpts, i_elems)
                                                        + bz * K(5, 3, j_qpts, i_elems))
                                                  + by * (bx * K(1, 5, j_qpts, i_elems)
                                                          + by * K(1, 1, j_qpts, i_elems)
                                                          + bz * K(1, 3, j_qpts, i_elems))
                                                  + bz * (bx * K(3, 5, j_qpts, i_elems)
                                                          + by * K(3, 1, j_qpts, i_elems)
                                                          + bz * K(3, 3, j_qpts, i_elems)));

               Y(knodes, 2, i_elems) += c_detJ * (bx * (bx * K(4, 4, j_qpts, i_elems)
                                                        + by * K(4, 3, j_qpts, i_elems)
                                                        + bz * K(4, 2, j_qpts, i_elems))
                                                  + by * (bx * K(3, 4, j_qpts, i_elems)
                                                          + by * K(3, 3, j_qpts, i_elems)
                                                          + bz * K(3, 2, j_qpts, i_elems))
                                                  + bz * (bx * K(2, 4, j_qpts, i_elems)
                                                          + by * K(2, 3, j_qpts, i_elems)
                                                          + bz * K(2, 2, j_qpts, i_elems)));
            }
         }
      });
   }
}

/// Method defining element assembly.
/** The result of the element assembly is added and stored in the @a emat
 Vector. */
void ExaNLFIntegrator::AssembleEA(const FiniteElementSpace &fes, Vector &emat)
{
   CALI_CXX_MARK_SCOPE("enlfi_assembleEA");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   space_dims = el.GetDim();
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

   nqpts = ir->GetNPoints();
   nnodes = el.GetDof();
   nelems = fes.GetNE();
   auto W = ir->GetWeights().Read();

   if ((space_dims == 1) || (space_dims == 2)) {
      MFEM_ABORT("Dimensions of 1 or 2 not supported.");
   }
   else {
      const int dim = 3;

      if (grad.Size() != (nqpts * dim * nnodes)) {
         grad.SetSize(nqpts * dim * nnodes, mfem::Device::GetMemoryType());
         {
            DenseMatrix DSh;
            const int offset = nnodes * dim;
            double *qpts_dshape_data = grad.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
               const IntegrationPoint &ip = ir->IntPoint(i);
               DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, dim);
               el.CalcDShape(ip, DSh);
            }
         }
         grad.UseDevice(true);
      }

      // geom->J really isn't going to work for us as of right now. We could just reorder it
      // to the version that we want it to be in instead...
      if (jacobian.Size() != (dim * dim * nqpts * nelems)) {
         jacobian.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
         jacobian.UseDevice(true);

         geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);

         const int DIM4 = 4;
         std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };

         RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
         RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.ReadWrite(), layout_jacob);

         RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{ nqpts, dim, dim, nelems } }, perm4);
         RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > geom_j_view(geom->J.Read(), layout_geom);

         MFEM_FORALL(i, nelems, {
            for (int j = 0; j < nqpts; j++) {
               for (int k = 0; k < dim; k++) {
                  for (int l = 0; l < dim; l++) {
                     J(l, k, j, i) = geom_j_view(j, l, k, i);
                  }
               }
            }
         });
      }

      const int DIM2 = 2;
      const int DIM3 = 3;
      const int DIM4 = 4;

      std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };

      // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

      RAJA::Layout<DIM4> layout_tensor = RAJA::make_permuted_layout({{ 2 * dim, 2 * dim, nqpts, nelems } }, perm4);
      RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > K(model->GetMatGrad()->Read(), layout_tensor);

      // Our field variables that are inputs and outputs
      RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{ nnodes * dim, nnodes * dim, nelems } }, perm3);
      RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > E(emat.ReadWrite(), layout_field);

      RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
      RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.ReadWrite(), layout_jacob);

      RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{ dim, dim } }, perm2);

      RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{ nnodes, dim, nqpts } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Gt(grad.Read(), layout_grads);

      double dt = model->GetModelDt();
      // This loop we'll want to parallelize the rest are all serial for now.
      MFEM_FORALL(i_elems, nelems, {
         double adj[dim * dim];
         double c_detJ;
         // So, we're going to say this view is constant however we're going to mutate the values only in
         // that one scoped section for the quadrature points.
         RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > A(&adj[0], layout_adj);
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around with us for
            // the adjugate term.
            {
               const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
               const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
               const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
               const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
               const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
               const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
               const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
               const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
               const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                   /* */ J21 * (J12 * J33 - J32 * J13) +
                                   /* */ J31 * (J12 * J23 - J22 * J13);
               c_detJ = 1.0 / detJ * W[j_qpts] * dt;
               // adj(J)
               adj[0] = (J22 * J33) - (J23 * J32); // 0,0
               adj[1] = (J32 * J13) - (J12 * J33); // 0,1
               adj[2] = (J12 * J23) - (J22 * J13); // 0,2
               adj[3] = (J31 * J23) - (J21 * J33); // 1,0
               adj[4] = (J11 * J33) - (J13 * J31); // 1,1
               adj[5] = (J21 * J13) - (J11 * J23); // 1,2
               adj[6] = (J21 * J32) - (J31 * J22); // 2,0
               adj[7] = (J31 * J12) - (J11 * J32); // 2,1
               adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            for (int knds = 0; knds < nnodes; knds++) {
               const double bx = Gt(knds, 0, j_qpts) * A(0, 0)
                                 + Gt(knds, 1, j_qpts) * A(0, 1)
                                 + Gt(knds, 2, j_qpts) * A(0, 2);

               const double by = Gt(knds, 0, j_qpts) * A(1, 0)
                                 + Gt(knds, 1, j_qpts) * A(1, 1)
                                 + Gt(knds, 2, j_qpts) * A(1, 2);

               const double bz = Gt(knds, 0, j_qpts) * A(2, 0)
                                 + Gt(knds, 1, j_qpts) * A(2, 1)
                                 + Gt(knds, 2, j_qpts) * A(2, 2);


               const double k11x = c_detJ * (bx * K(0, 0, j_qpts, i_elems)
                                             + by * K(0, 5, j_qpts, i_elems)
                                             + bz * K(0, 4, j_qpts, i_elems));
               const double k11y = c_detJ * (bx * K(5, 0, j_qpts, i_elems)
                                             + by * K(5, 5, j_qpts, i_elems)
                                             + bz * K(5, 4, j_qpts, i_elems));
               const double k11z = c_detJ * (bx * K(4, 0, j_qpts, i_elems)
                                             + by * K(4, 5, j_qpts, i_elems)
                                             + bz * K(4, 4, j_qpts, i_elems));

               const double k12x = c_detJ * (bx * K(0, 5, j_qpts, i_elems)
                                             + by * K(0, 1, j_qpts, i_elems)
                                             + bz * K(0, 3, j_qpts, i_elems));
               const double k12y = c_detJ * (bx * K(5, 5, j_qpts, i_elems)
                                             + by * K(5, 1, j_qpts, i_elems)
                                             + bz * K(5, 3, j_qpts, i_elems));
               const double k12z = c_detJ * (bx * K(4, 5, j_qpts, i_elems)
                                             + by * K(4, 1, j_qpts, i_elems)
                                             + bz * K(4, 3, j_qpts, i_elems));

               const double k13x = c_detJ * (bx * K(0, 4, j_qpts, i_elems)
                                             + by * K(0, 3, j_qpts, i_elems)
                                             + bz * K(0, 2, j_qpts, i_elems));
               const double k13y = c_detJ * (bx * K(5, 4, j_qpts, i_elems)
                                             + by * K(5, 3, j_qpts, i_elems)
                                             + bz * K(5, 2, j_qpts, i_elems));
               const double k13z = c_detJ * (bx * K(4, 4, j_qpts, i_elems)
                                             + by * K(4, 3, j_qpts, i_elems)
                                             + bz * K(4, 2, j_qpts, i_elems));

               const double k21x = c_detJ * (bx * K(5, 0, j_qpts, i_elems)
                                             + by * K(5, 5, j_qpts, i_elems)
                                             + bz * K(5, 4, j_qpts, i_elems));
               const double k21y = c_detJ * (bx * K(1, 0, j_qpts, i_elems)
                                             + by * K(1, 5, j_qpts, i_elems)
                                             + bz * K(1, 4, j_qpts, i_elems));
               const double k21z = c_detJ * (bx * K(3, 0, j_qpts, i_elems)
                                             + by * K(3, 5, j_qpts, i_elems)
                                             + bz * K(3, 4, j_qpts, i_elems));

               const double k22x = c_detJ * (bx * K(5, 5, j_qpts, i_elems)
                                             + by * K(5, 1, j_qpts, i_elems)
                                             + bz * K(5, 3, j_qpts, i_elems));
               const double k22y = c_detJ * (bx * K(1, 5, j_qpts, i_elems)
                                             + by * K(1, 1, j_qpts, i_elems)
                                             + bz * K(1, 3, j_qpts, i_elems));
               const double k22z = c_detJ * (bx * K(3, 5, j_qpts, i_elems)
                                             + by * K(3, 1, j_qpts, i_elems)
                                             + bz * K(3, 3, j_qpts, i_elems));

               const double k23x = c_detJ * (bx * K(5, 4, j_qpts, i_elems)
                                             + by * K(5, 3, j_qpts, i_elems)
                                             + bz * K(5, 2, j_qpts, i_elems));
               const double k23y = c_detJ * (bx * K(1, 4, j_qpts, i_elems)
                                             + by * K(1, 3, j_qpts, i_elems)
                                             + bz * K(1, 2, j_qpts, i_elems));
               const double k23z = c_detJ * (bx * K(3, 4, j_qpts, i_elems)
                                             + by * K(3, 3, j_qpts, i_elems)
                                             + bz * K(3, 2, j_qpts, i_elems));

               const double k31x = c_detJ * (bx * K(4, 0, j_qpts, i_elems)
                                             + by * K(4, 5, j_qpts, i_elems)
                                             + bz * K(4, 4, j_qpts, i_elems));
               const double k31y = c_detJ * (bx * K(3, 0, j_qpts, i_elems)
                                             + by * K(3, 5, j_qpts, i_elems)
                                             + bz * K(3, 4, j_qpts, i_elems));
               const double k31z = c_detJ * (bx * K(2, 0, j_qpts, i_elems)
                                             + by * K(2, 5, j_qpts, i_elems)
                                             + bz * K(2, 4, j_qpts, i_elems));

               const double k32x = c_detJ * (bx * K(4, 5, j_qpts, i_elems)
                                             + by * K(4, 1, j_qpts, i_elems)
                                             + bz * K(4, 3, j_qpts, i_elems));
               const double k32y = c_detJ * (bx * K(3, 5, j_qpts, i_elems)
                                             + by * K(3, 1, j_qpts, i_elems)
                                             + bz * K(3, 3, j_qpts, i_elems));
               const double k32z = c_detJ * (bx * K(2, 5, j_qpts, i_elems)
                                             + by * K(2, 1, j_qpts, i_elems)
                                             + bz * K(2, 3, j_qpts, i_elems));

               const double k33x = c_detJ * (bx * K(4, 4, j_qpts, i_elems)
                                             + by * K(4, 3, j_qpts, i_elems)
                                             + bz * K(4, 2, j_qpts, i_elems));
               const double k33y = c_detJ * (bx * K(3, 4, j_qpts, i_elems)
                                             + by * K(3, 3, j_qpts, i_elems)
                                             + bz * K(3, 2, j_qpts, i_elems));
               const double k33z = c_detJ * (bx * K(2, 4, j_qpts, i_elems)
                                             + by * K(2, 3, j_qpts, i_elems)
                                             + bz * K(2, 2, j_qpts, i_elems));

               for (int lnds = 0; lnds < nnodes; lnds++) {
                  const double gx = Gt(lnds, 0, j_qpts) * A(0, 0)
                                    + Gt(lnds, 1, j_qpts) * A(0, 1)
                                    + Gt(lnds, 2, j_qpts) * A(0, 2);

                  const double gy = Gt(lnds, 0, j_qpts) * A(1, 0)
                                    + Gt(lnds, 1, j_qpts) * A(1, 1)
                                    + Gt(lnds, 2, j_qpts) * A(1, 2);

                  const double gz = Gt(lnds, 0, j_qpts) * A(2, 0)
                                    + Gt(lnds, 1, j_qpts) * A(2, 1)
                                    + Gt(lnds, 2, j_qpts) * A(2, 2);


                  E(lnds, knds, i_elems) += gx * k11x + gy * k11y + gz * k11z;
                  E(lnds, knds + nnodes, i_elems) += gx * k12x + gy * k12y + gz * k12z;
                  E(lnds, knds + 2 * nnodes, i_elems) += gx * k13x + gy * k13y + gz * k13z;

                  E(lnds + nnodes, knds, i_elems) += gx * k21x + gy * k21y + gz * k21z;
                  E(lnds + nnodes, knds + nnodes, i_elems) += gx * k22x + gy * k22y + gz * k22z;
                  E(lnds + nnodes, knds + 2 * nnodes, i_elems) += gx * k23x + gy * k23y + gz * k23z;

                  E(lnds + 2 * nnodes, knds, i_elems) += gx * k31x + gy * k31y + gz * k31z;
                  E(lnds + 2 * nnodes, knds + nnodes, i_elems) += gx * k32x + gy * k32y + gz * k32z;
                  E(lnds + 2 * nnodes, knds + 2 * nnodes, i_elems) += gx * k33x + gy * k33y + gz * k33z;
               }
            }
         }
      });
   }
}

// Outside of the UMAT function calls this should be the function called
// to assemble our residual vectors.
void ICExaNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, Vector &elvect)
{
   CALI_CXX_MARK_SCOPE("icenlfi_assembleElemVec");
   int dof = el.GetDof(), dim = el.GetDim();

   DenseMatrix DSh, DS, eDS;
   DenseMatrix Jpt;
   DenseMatrix PMatI, PMatO;
   // This is our stress tensor
   DenseMatrix P;
   DenseMatrix grad_trans;
   // temp1 is now going to become the transpose Bmatrix as seen in
   // [B^t][tan_stiff][B]
   grad_trans.SetSize(dof * dim, 6);

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   eDS.SetSize(dof, dim);
   eDS = 0.0;
   Jpt.SetSize(dim);

   // PMatI would be our velocity in this case
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof * dim);

   // PMatO would be our residual vector
   elvect = 0.0;
   PMatO.UseExternalData(elvect.HostReadWrite(), dof * dim, 1);

   const IntegrationRule *ir = IntRule;
   if (!ir) {
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1)); // must match quadrature space
   }

   const IntegrationRule *irc =  &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
   double eVol = 0.0;

   for (int i = 0; i < irc->GetNPoints(); i++) {
      const IntegrationPoint &ip = irc->IntPoint(i);
      Ttr.SetIntPoint(&ip);

      // compute Jacobian of the transformation
      Jpt = Ttr.InverseJacobian(); // Jrt = dxi / dX

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jpt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX
      DS *= (Ttr.Weight() * ip.weight);
      eDS += DS;

      eVol += (Ttr.Weight() * ip.weight);

   }

   eDS *= (1.0 / eVol);

   double stress[6];

   P.UseExternalData(&stress[0], 6, 1);

   for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);

      // compute Jacobian of the transformation
      Jpt = Ttr.InverseJacobian(); // Jrt = dxi / dX

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jpt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX

      model->GetElementStress(Ttr.ElementNo, i, false, &stress[0], 6);
      model->GenerateGradBarMatrix(DS, eDS, grad_trans);
      grad_trans *= (ip.weight * Ttr.Weight());
      AddMult(grad_trans, P, PMatO);

   }

   return;
}

void ICExaNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector & /*elfun*/, DenseMatrix &elmat)
{
   CALI_CXX_MARK_SCOPE("icenlfi_assembleElemGrad");
   int dof = el.GetDof(), dim = el.GetDim();

   DenseMatrix DSh, DS, eDS, Jrt;

   // Now time to start assembling stuff
   DenseMatrix grad_trans, temp;
   DenseMatrix tan_stiff;

   int ngrad_dim2 = 36;
   double matGrad[ngrad_dim2];
   // Delta in our timestep
   double dt = model->GetModelDt();

   // temp1 is now going to become the transpose Bmatrix as seen in
   // [B^t][tan_stiff][B]
   grad_trans.SetSize(dof * dim, 6);
   // We need a temp matrix to store our first matrix results as seen in here
   temp.SetSize(6, dof * dim);

   tan_stiff.UseExternalData(&matGrad[0], 6, 6);

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   eDS.SetSize(dof, dim);
   eDS = 0.0;
   Jrt.SetSize(dim);
   elmat.SetSize(dof * dim);

   const IntegrationRule *ir = IntRule;
   if (!ir) {
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1)); // <--- must match quadrature space
   }

   elmat = 0.0;

   const IntegrationRule *irc =  &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
   double eVol = 0.0; 

   for (int i = 0; i < irc->GetNPoints(); i++) {
      const IntegrationPoint &ip = irc->IntPoint(i);
      Ttr.SetIntPoint(&ip);

      // compute Jacobian of the transformation
      Jrt = Ttr.InverseJacobian(); // Jrt = dxi / dX

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX
      DS *= (Ttr.Weight() * ip.weight);
      eDS += DS;

      eVol += (Ttr.Weight() * ip.weight);

   }

   eDS *= (1.0 / eVol);

   for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);

      model->GetElementMatGrad(Ttr.ElementNo, i, matGrad, ngrad_dim2);
      // temp1 is B^t
      model->GenerateGradBarMatrix(DS, eDS, grad_trans);
      // We multiple our quadrature wts here to our tan_stiff matrix
      tan_stiff *= dt * ip.weight * Ttr.Weight();
      // We use kgeom as a temporary matrix
      // kgeom = [Cstiff][B]
      MultABt(tan_stiff, grad_trans, temp);
      // We now add our [B^t][kgeom] product to our tangent stiffness matrix that
      // we want to output to our material tangent stiffness matrix
      AddMult(grad_trans, temp, elmat);
   }

   return;
}

/// Method defining element assembly.
/** The result of the element assembly is added and stored in the @a emat
    Vector. */
void ICExaNLFIntegrator::AssembleEA(const mfem::FiniteElementSpace &fes, mfem::Vector &emat)
{
   CALI_CXX_MARK_SCOPE("icenlfi_assembleEA");
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   space_dims = el.GetDim();
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

   nqpts = ir->GetNPoints();
   nnodes = el.GetDof();
   nelems = fes.GetNE();
   auto W = ir->GetWeights().Read();

   if ((space_dims == 1) || (space_dims == 2)) {
      MFEM_ABORT("Dimensions of 1 or 2 not supported.");
   }

   else {
      const int dim = 3;

      if (eDS.Size() != (nnodes * dim * nelems)) {
         eDS.SetSize(nnodes * space_dims * nelems, mfem::Device::GetMemoryType());
         eDS.UseDevice();
         eDS = 0.0;
      }

      if (grad.Size() != (nqpts * dim * nnodes)) {
         grad.SetSize(nqpts * dim * nnodes, mfem::Device::GetMemoryType());
         {
            DenseMatrix DSh;
            const int offset = nnodes * dim;
            double *qpts_dshape_data = grad.HostReadWrite();
            for (int i = 0; i < nqpts; i++) {
               const IntegrationPoint &ip = ir->IntPoint(i);
               DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, dim);
               el.CalcDShape(ip, DSh);
            }
         }
         grad.UseDevice(true);
      }

      // geom->J really isn't going to work for us as of right now. We could just reorder it
      // to the version that we want it to be in instead...
      if (jacobian.Size() != (dim * dim * nqpts * nelems)) {
         jacobian.SetSize(dim * dim * nqpts * nelems, mfem::Device::GetMemoryType());
         jacobian.UseDevice(true);

         geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);

         const int DIM4 = 4;
         std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };

         RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
         RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.ReadWrite(), layout_jacob);

         RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{ nqpts, dim, dim, nelems } }, perm4);
         RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > geom_j_view(geom->J.Read(), layout_geom);

         MFEM_FORALL(i, nelems, {
            for (int j = 0; j < nqpts; j++) {
               for (int k = 0; k < dim; k++) {
                  for (int l = 0; l < dim; l++) {
                     J(l, k, j, i) = geom_j_view(j, l, k, i);
                  }
               }
            }
         });
      }

      const int DIM2 = 2;
      const int DIM3 = 3;
      const int DIM4 = 4;

      std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };

      // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

      RAJA::Layout<DIM4> layout_tensor = RAJA::make_permuted_layout({{ 2 * dim, 2 * dim, nqpts, nelems } }, perm4);
      RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > K(model->GetMatGrad()->Read(), layout_tensor);

      // Our field variables that are inputs and outputs
      RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{ nnodes * dim, nnodes * dim, nelems } }, perm3);
      RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > E(emat.ReadWrite(), layout_field);

      RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
      RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.ReadWrite(), layout_jacob);

      RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{ dim, dim } }, perm2);

      RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{ nnodes, dim, nqpts } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Gt(grad.Read(), layout_grads);

      RAJA::Layout<DIM3> layout_egrads = RAJA::make_permuted_layout({{ nnodes, dim, nelems } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > eDS_view(eDS.Read(), layout_egrads);

      double dt = model->GetModelDt();
      const double i3 = 1.0 / 3.0;
      // This loop we'll want to parallelize the rest are all serial for now.
      MFEM_FORALL(i_elems, nelems, {
         double adj[dim * dim];
         double c_detJ;
         // So, we're going to say this view is constant however we're going to mutate the values only in
         // that one scoped section for the quadrature points.
         RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > A(&adj[0], layout_adj);
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around with us for
            // the adjugate term.
            {
               const double J11 = J(0, 0, j_qpts, i_elems); // 0,0
               const double J21 = J(1, 0, j_qpts, i_elems); // 1,0
               const double J31 = J(2, 0, j_qpts, i_elems); // 2,0
               const double J12 = J(0, 1, j_qpts, i_elems); // 0,1
               const double J22 = J(1, 1, j_qpts, i_elems); // 1,1
               const double J32 = J(2, 1, j_qpts, i_elems); // 2,1
               const double J13 = J(0, 2, j_qpts, i_elems); // 0,2
               const double J23 = J(1, 2, j_qpts, i_elems); // 1,2
               const double J33 = J(2, 2, j_qpts, i_elems); // 2,2
               const double detJ = J11 * (J22 * J33 - J32 * J23) -
                                   /* */ J21 * (J12 * J33 - J32 * J13) +
                                   /* */ J31 * (J12 * J23 - J22 * J13);
               c_detJ = 1.0 / detJ * W[j_qpts] * dt;
               // adj(J)
               adj[0] = (J22 * J33) - (J23 * J32); // 0,0
               adj[1] = (J32 * J13) - (J12 * J33); // 0,1
               adj[2] = (J12 * J23) - (J22 * J13); // 0,2
               adj[3] = (J31 * J23) - (J21 * J33); // 1,0
               adj[4] = (J11 * J33) - (J13 * J31); // 1,1
               adj[5] = (J21 * J13) - (J11 * J23); // 1,2
               adj[6] = (J21 * J32) - (J31 * J22); // 2,0
               adj[7] = (J31 * J12) - (J11 * J32); // 2,1
               adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            for (int knds = 0; knds < nnodes; knds++) {
               const double bx = Gt(knds, 0, j_qpts) * A(0, 0)
                                 + Gt(knds, 1, j_qpts) * A(0, 1)
                                 + Gt(knds, 2, j_qpts) * A(0, 2);

               const double by = Gt(knds, 0, j_qpts) * A(1, 0)
                                 + Gt(knds, 1, j_qpts) * A(1, 1)
                                 + Gt(knds, 2, j_qpts) * A(1, 2);

               const double bz = Gt(knds, 0, j_qpts) * A(2, 0)
                                 + Gt(knds, 1, j_qpts) * A(2, 1)
                                 + Gt(knds, 2, j_qpts) * A(2, 2);
               const double b4 = i3 * (eDS_view(knds, 0, i_elems) - bx);
               const double b5 = b4 + bx;
               const double b6 = i3 * (eDS_view(knds, 1, i_elems) - by);
               const double b7 = b6 + by;
               const double b8 = i3 * (eDS_view(knds, 2, i_elems) - bz);
               const double b9 = b8 + bz;


               const double k11w = c_detJ * (b4 * K(1, 1, j_qpts, i_elems)
                                           + b4 * K(1, 2, j_qpts, i_elems)
                                           + b5 * K(1, 0, j_qpts, i_elems)
                                           + by * K(1, 5, j_qpts, i_elems)
                                           + bz * K(1, 4, j_qpts, i_elems)
                                           + b4 * K(2, 1, j_qpts, i_elems)
                                           + b4 * K(2, 2, j_qpts, i_elems)
                                           + b5 * K(2, 1, j_qpts, i_elems)
                                           + by * K(2, 5, j_qpts, i_elems)
                                           + bz * K(2, 4, j_qpts, i_elems));

               const double k11x = c_detJ * (b4 * K(0, 1, j_qpts, i_elems)
                                           + b4 * K(0, 2, j_qpts, i_elems)
                                           + b5 * K(0, 0, j_qpts, i_elems)
                                           + by * K(0, 5, j_qpts, i_elems)
                                           + bz * K(0, 4, j_qpts, i_elems));

               const double k11y = c_detJ * (b4 * K(5, 1, j_qpts, i_elems)
                                           + b4 * K(5, 2, j_qpts, i_elems)
                                           + b5 * K(5, 0, j_qpts, i_elems)
                                           + by * K(5, 5, j_qpts, i_elems)
                                           + bz * K(5, 4, j_qpts, i_elems));

               const double k11z = c_detJ * (b4 * K(4, 1, j_qpts, i_elems)
                                           + b4 * K(4, 2, j_qpts, i_elems)
                                           + b5 * K(4, 0, j_qpts, i_elems)
                                           + by * K(4, 5, j_qpts, i_elems)
                                           + bz * K(4, 4, j_qpts, i_elems));

               const double k12w = c_detJ * (b6 * K(1, 0, j_qpts, i_elems)
                                           + b6 * K(1, 2, j_qpts, i_elems)
                                           + b7 * K(1, 1, j_qpts, i_elems)
                                           + bx * K(1, 5, j_qpts, i_elems)
                                           + bz * K(1, 3, j_qpts, i_elems)
                                           + b6 * K(2, 0, j_qpts, i_elems)
                                           + b6 * K(2, 2, j_qpts, i_elems)
                                           + b7 * K(2, 1, j_qpts, i_elems)
                                           + bx * K(2, 5, j_qpts, i_elems)
                                           + bz * K(2, 3, j_qpts, i_elems));

               const double k12x = c_detJ * (b6 * K(0, 0, j_qpts, i_elems)
                                           + b6 * K(0, 2, j_qpts, i_elems)
                                           + b7 * K(0, 1, j_qpts, i_elems)
                                           + bx * K(0, 5, j_qpts, i_elems)
                                           + bz * K(0, 3, j_qpts, i_elems));

               const double k12y = c_detJ * (b6 * K(5, 0, j_qpts, i_elems)
                                           + b6 * K(5, 2, j_qpts, i_elems)
                                           + b7 * K(5, 1, j_qpts, i_elems)
                                           + bx * K(5, 5, j_qpts, i_elems)
                                           + bz * K(5, 3, j_qpts, i_elems));

               const double k12z = c_detJ * (b6 * K(4, 0, j_qpts, i_elems)
                                           + b6 * K(4, 2, j_qpts, i_elems)
                                           + b7 * K(4, 1, j_qpts, i_elems)
                                           + bx * K(4, 5, j_qpts, i_elems)
                                           + bz * K(4, 3, j_qpts, i_elems));

               const double k13w = c_detJ * (b8 * K(1, 0, j_qpts, i_elems)
                                           + b8 * K(1, 1, j_qpts, i_elems)
                                           + b9 * K(1, 2, j_qpts, i_elems)
                                           + bx * K(1, 4, j_qpts, i_elems)
                                           + by * K(1, 3, j_qpts, i_elems)
                                           + b8 * K(2, 0, j_qpts, i_elems)
                                           + b8 * K(2, 1, j_qpts, i_elems)
                                           + b9 * K(2, 2, j_qpts, i_elems)
                                           + bx * K(2, 4, j_qpts, i_elems)
                                           + by * K(2, 3, j_qpts, i_elems));

               const double k13x = c_detJ * (b8 * K(0, 0, j_qpts, i_elems)
                                           + b8 * K(0, 1, j_qpts, i_elems)
                                           + b9 * K(0, 2, j_qpts, i_elems)
                                           + bx * K(0, 4, j_qpts, i_elems)
                                           + by * K(0, 3, j_qpts, i_elems));

               const double k13y = c_detJ * (b8 * K(5, 0, j_qpts, i_elems)
                                           + b8 * K(5, 1, j_qpts, i_elems)
                                           + b9 * K(5, 2, j_qpts, i_elems)
                                           + bx * K(5, 4, j_qpts, i_elems)
                                           + by * K(5, 3, j_qpts, i_elems));

               const double k13z = c_detJ * (b8 * K(4, 0, j_qpts, i_elems)
                                           + b8 * K(4, 1, j_qpts, i_elems)
                                           + b9 * K(4, 2, j_qpts, i_elems)
                                           + bx * K(4, 4, j_qpts, i_elems)
                                           + by * K(4, 3, j_qpts, i_elems));

               const double k21w = c_detJ * (b4 * K(0, 1, j_qpts, i_elems)
                                           + b4 * K(0, 2, j_qpts, i_elems)
                                           + b5 * K(0, 0, j_qpts, i_elems)
                                           + by * K(0, 5, j_qpts, i_elems)
                                           + bz * K(0, 4, j_qpts, i_elems)
                                           + b4 * K(2, 1, j_qpts, i_elems)
                                           + b4 * K(2, 2, j_qpts, i_elems)
                                           + b5 * K(2, 1, j_qpts, i_elems)
                                           + by * K(2, 5, j_qpts, i_elems)
                                           + bz * K(2, 4, j_qpts, i_elems));

               const double k21x = c_detJ * (b4 * K(1, 1, j_qpts, i_elems)
                                           + b4 * K(1, 2, j_qpts, i_elems)
                                           + b5 * K(1, 0, j_qpts, i_elems)
                                           + by * K(1, 5, j_qpts, i_elems)
                                           + bz * K(1, 4, j_qpts, i_elems));

               const double k21y = c_detJ * (b4 * K(5, 1, j_qpts, i_elems)
                                           + b4 * K(5, 2, j_qpts, i_elems)
                                           + b5 * K(5, 0, j_qpts, i_elems)
                                           + by * K(5, 5, j_qpts, i_elems)
                                           + bz * K(5, 4, j_qpts, i_elems));

               const double k21z = c_detJ * (b4 * K(3, 1, j_qpts, i_elems)
                                           + b4 * K(3, 2, j_qpts, i_elems)
                                           + b5 * K(3, 0, j_qpts, i_elems)
                                           + by * K(3, 5, j_qpts, i_elems)
                                           + bz * K(3, 4, j_qpts, i_elems));

               const double k22w = c_detJ * (b6 * K(0, 0, j_qpts, i_elems)
                                           + b6 * K(0, 2, j_qpts, i_elems)
                                           + b7 * K(0, 1, j_qpts, i_elems)
                                           + bx * K(0, 5, j_qpts, i_elems)
                                           + bz * K(0, 3, j_qpts, i_elems)
                                           + b6 * K(2, 0, j_qpts, i_elems)
                                           + b6 * K(2, 2, j_qpts, i_elems)
                                           + b7 * K(2, 1, j_qpts, i_elems)
                                           + bx * K(2, 5, j_qpts, i_elems)
                                           + bz * K(2, 3, j_qpts, i_elems));

               const double k22x = c_detJ * (b6 * K(1, 0, j_qpts, i_elems)
                                           + b6 * K(1, 2, j_qpts, i_elems)
                                           + b7 * K(1, 1, j_qpts, i_elems)
                                           + bx * K(1, 5, j_qpts, i_elems)
                                           + bz * K(1, 3, j_qpts, i_elems));

               const double k22y = c_detJ * (b6 * K(5, 0, j_qpts, i_elems)
                                           + b6 * K(5, 2, j_qpts, i_elems)
                                           + b7 * K(5, 1, j_qpts, i_elems)
                                           + bx * K(5, 5, j_qpts, i_elems)
                                           + bz * K(5, 3, j_qpts, i_elems));

               const double k22z = c_detJ * (b6 * K(3, 0, j_qpts, i_elems)
                                           + b6 * K(3, 2, j_qpts, i_elems)
                                           + b7 * K(3, 1, j_qpts, i_elems)
                                           + bx * K(3, 5, j_qpts, i_elems)
                                           + bz * K(3, 3, j_qpts, i_elems));

               const double k23w = c_detJ * (b8 * K(0, 0, j_qpts, i_elems)
                                           + b8 * K(0, 1, j_qpts, i_elems)
                                           + b9 * K(0, 2, j_qpts, i_elems)
                                           + bx * K(0, 4, j_qpts, i_elems)
                                           + by * K(0, 3, j_qpts, i_elems)
                                           + b8 * K(2, 0, j_qpts, i_elems)
                                           + b8 * K(2, 1, j_qpts, i_elems)
                                           + b9 * K(2, 2, j_qpts, i_elems)
                                           + bx * K(2, 4, j_qpts, i_elems)
                                           + by * K(2, 3, j_qpts, i_elems));

               const double k23x = c_detJ * (b8 * K(1, 0, j_qpts, i_elems)
                                           + b8 * K(1, 1, j_qpts, i_elems)
                                           + b9 * K(1, 2, j_qpts, i_elems)
                                           + bx * K(1, 4, j_qpts, i_elems)
                                           + by * K(1, 3, j_qpts, i_elems));

               const double k23y = c_detJ * (b8 * K(5, 0, j_qpts, i_elems)
                                           + b8 * K(5, 1, j_qpts, i_elems)
                                           + b9 * K(5, 2, j_qpts, i_elems)
                                           + bx * K(5, 4, j_qpts, i_elems)
                                           + by * K(5, 3, j_qpts, i_elems));

               const double k23z = c_detJ * (b8 * K(3, 0, j_qpts, i_elems)
                                           + b8 * K(3, 1, j_qpts, i_elems)
                                           + b9 * K(3, 2, j_qpts, i_elems)
                                           + bx * K(3, 4, j_qpts, i_elems)
                                           + by * K(3, 3, j_qpts, i_elems));

               const double k31w = c_detJ * (b4 * K(0, 1, j_qpts, i_elems)
                                           + b4 * K(0, 2, j_qpts, i_elems)
                                           + b5 * K(0, 0, j_qpts, i_elems)
                                           + by * K(0, 5, j_qpts, i_elems)
                                           + bz * K(0, 4, j_qpts, i_elems)
                                           + b4 * K(1, 1, j_qpts, i_elems)
                                           + b4 * K(1, 2, j_qpts, i_elems)
                                           + b5 * K(1, 1, j_qpts, i_elems)
                                           + by * K(1, 5, j_qpts, i_elems)
                                           + bz * K(1, 4, j_qpts, i_elems));

               const double k31x = c_detJ * (b4 * K(2, 1, j_qpts, i_elems)
                                           + b4 * K(2, 2, j_qpts, i_elems)
                                           + b5 * K(2, 0, j_qpts, i_elems)
                                           + by * K(2, 5, j_qpts, i_elems)
                                           + bz * K(2, 4, j_qpts, i_elems));

               const double k31y = c_detJ * (b4 * K(4, 1, j_qpts, i_elems)
                                           + b4 * K(4, 2, j_qpts, i_elems)
                                           + b5 * K(4, 0, j_qpts, i_elems)
                                           + by * K(4, 5, j_qpts, i_elems)
                                           + bz * K(4, 4, j_qpts, i_elems));

               const double k31z = c_detJ * (b4 * K(3, 1, j_qpts, i_elems)
                                           + b4 * K(3, 2, j_qpts, i_elems)
                                           + b5 * K(3, 0, j_qpts, i_elems)
                                           + by * K(3, 5, j_qpts, i_elems)
                                           + bz * K(3, 4, j_qpts, i_elems));

               const double k32w = c_detJ * (b6 * K(0, 0, j_qpts, i_elems)
                                           + b6 * K(0, 2, j_qpts, i_elems)
                                           + b7 * K(0, 1, j_qpts, i_elems)
                                           + bx * K(0, 5, j_qpts, i_elems)
                                           + bz * K(0, 3, j_qpts, i_elems)
                                           + b6 * K(1, 0, j_qpts, i_elems)
                                           + b6 * K(1, 2, j_qpts, i_elems)
                                           + b7 * K(1, 1, j_qpts, i_elems)
                                           + bx * K(1, 5, j_qpts, i_elems)
                                           + bz * K(1, 3, j_qpts, i_elems));

               const double k32x = c_detJ * (b6 * K(2, 0, j_qpts, i_elems)
                                           + b6 * K(2, 2, j_qpts, i_elems)
                                           + b7 * K(2, 1, j_qpts, i_elems)
                                           + bx * K(2, 5, j_qpts, i_elems)
                                           + bz * K(2, 3, j_qpts, i_elems));

               const double k32y = c_detJ * (b6 * K(4, 0, j_qpts, i_elems)
                                           + b6 * K(4, 2, j_qpts, i_elems)
                                           + b7 * K(4, 1, j_qpts, i_elems)
                                           + bx * K(4, 5, j_qpts, i_elems)
                                           + bz * K(4, 3, j_qpts, i_elems));

               const double k32z = c_detJ * (b6 * K(3, 0, j_qpts, i_elems)
                                           + b6 * K(3, 2, j_qpts, i_elems)
                                           + b7 * K(3, 1, j_qpts, i_elems)
                                           + bx * K(3, 5, j_qpts, i_elems)
                                           + bz * K(3, 3, j_qpts, i_elems));

               const double k33w = c_detJ * (b8 * K(0, 0, j_qpts, i_elems)
                                           + b8 * K(0, 1, j_qpts, i_elems)
                                           + b9 * K(0, 2, j_qpts, i_elems)
                                           + bx * K(0, 4, j_qpts, i_elems)
                                           + by * K(0, 3, j_qpts, i_elems)
                                           + b8 * K(1, 0, j_qpts, i_elems)
                                           + b8 * K(1, 1, j_qpts, i_elems)
                                           + b9 * K(1, 2, j_qpts, i_elems)
                                           + bx * K(1, 4, j_qpts, i_elems)
                                           + by * K(1, 3, j_qpts, i_elems));

               const double k33x = c_detJ * (b8 * K(2, 0, j_qpts, i_elems)
                                           + b8 * K(2, 1, j_qpts, i_elems)
                                           + b9 * K(2, 2, j_qpts, i_elems)
                                           + bx * K(2, 4, j_qpts, i_elems)
                                           + by * K(2, 3, j_qpts, i_elems));

               const double k33y = c_detJ * (b8 * K(4, 0, j_qpts, i_elems)
                                           + b8 * K(4, 1, j_qpts, i_elems)
                                           + b9 * K(4, 2, j_qpts, i_elems)
                                           + bx * K(4, 4, j_qpts, i_elems)
                                           + by * K(4, 3, j_qpts, i_elems));

               const double k33z = c_detJ * (b8 * K(3, 0, j_qpts, i_elems)
                                           + b8 * K(3, 1, j_qpts, i_elems)
                                           + b9 * K(3, 2, j_qpts, i_elems)
                                           + bx * K(3, 4, j_qpts, i_elems)
                                           + by * K(3, 3, j_qpts, i_elems));

               for (int lnds = 0; lnds < nnodes; lnds++) {
                  const double gx = Gt(lnds, 0, j_qpts) * A(0, 0)
                                    + Gt(lnds, 1, j_qpts) * A(0, 1)
                                    + Gt(lnds, 2, j_qpts) * A(0, 2);

                  const double gy = Gt(lnds, 0, j_qpts) * A(1, 0)
                                    + Gt(lnds, 1, j_qpts) * A(1, 1)
                                    + Gt(lnds, 2, j_qpts) * A(1, 2);

                  const double gz = Gt(lnds, 0, j_qpts) * A(2, 0)
                                    + Gt(lnds, 1, j_qpts) * A(2, 1)
                                    + Gt(lnds, 2, j_qpts) * A(2, 2);

                  const double g4 = i3 * (eDS_view(lnds, 0, i_elems) - gx);
                  const double g5 = g4 + gx;
                  const double g6 = i3 * (eDS_view(lnds, 1, i_elems) - gy);
                  const double g7 = g6 + gy;
                  const double g8 = i3 * (eDS_view(lnds, 2, i_elems) - gz);
                  const double g9 = g8 + gz;

                  E(lnds, knds, i_elems) += g4 * k11w + g5 * k11x + gy * k11y + gz * k11z;
                  E(lnds, knds + nnodes, i_elems) += g4 * k12w + g5 * k12x + gy * k12y + gz * k12z; 
                  E(lnds, knds + 2 * nnodes, i_elems) += g4 * k13w + g5 * k13x + gy * k13y + gz * k13z;

                  E(lnds + nnodes, knds, i_elems) += g6 * k21w + g7 * k21x + gx * k21y + gz * k21z;
                  E(lnds + nnodes, knds + nnodes, i_elems) += g6 * k22w + g7 * k22x + gx * k22y + gz * k22z;
                  E(lnds + nnodes, knds + 2 * nnodes, i_elems) += g6 * k23w + g7 * k23x + gx * k23y + gz * k23z;

                  E(lnds + 2 * nnodes, knds, i_elems) += g8 * k31w + g9 * k31x + gx * k31y + gy * k31z;
                  E(lnds + 2 * nnodes, knds + nnodes, i_elems) += g8 * k32w + g9 * k32x + gx * k32y + gy * k32z;
                  E(lnds + 2 * nnodes, knds + 2 * nnodes, i_elems) += g8 * k33w + g9 * k33x + gx * k33y + gy * k33z;
               }
            }
         }
      });
   }

}