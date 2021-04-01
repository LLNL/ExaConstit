#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_integrators.hpp"
#include <string>
#include <sstream>
#include "RAJA/RAJA.hpp"

#include <gtest/gtest.h>

using namespace std;
using namespace mfem;

static int outputLevel = 0;

//Routine to test the deformation gradient to make sure we're getting out the right values.
//This applies the following displacement field to the nodal values:
//u_vec = (2x + 3y + 4z)i + (4x + 2y + 3z)j + (3x + 4y + 2z)k
void test_deformation_field_set(ParGridFunction *gf, ParGridFunction *disp)
{
   
   const double* temp_vals = gf->HostRead();
   double* vals = disp->HostReadWrite();

   int dim = gf->Size()/3;

   for (int i = 0; i < dim; ++i)
   {
      const double x1 = temp_vals[i];
      const double x2 = temp_vals[i + dim];
      const double x3 = temp_vals[i + 2 * dim];
      
      vals[i] = 2 * x1 + 3 * x2 + 4 * x3;
      vals[i + dim] = 4 * x1 + 2 * x2 + 3 * x3;
      vals[i + 2 * dim] = 3 * x1 + 4 * x2 + 2 * x3;
   }
}

// Performs all the calculations related to calculating the velocity gradient
// vel_grad_array should be set to 0.0 outside of this function.
void grad_calc(const int nqpts, const int nelems, const int nnodes,
               const double *jacobian_data, const double *loc_grad_data,
               const double *vel_data, double* vel_grad_array)
{
   const int DIM4 = 4;
   const int DIM3 = 3;
   const int DIM2 = 2;
   std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
   std::array<RAJA::idx_t, DIM3> perm3{{ 2, 1, 0 } };
   std::array<RAJA::idx_t, DIM2> perm2{{ 1, 0 } };

   const int dim = 3;
   const int space_dim2 = dim * dim;

   // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
   RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
   RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian_data, layout_jacob);
   // vgrad
   RAJA::Layout<DIM4> layout_vgrad = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
   RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > vgrad_view(vel_grad_array, layout_vgrad);
   // velocity
   RAJA::Layout<DIM3> layout_vel = RAJA::make_permuted_layout({{ nnodes, dim, nelems } }, perm3);
   RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > vel_view(vel_data, layout_vel);
   // loc_grad
   RAJA::Layout<DIM3> layout_loc_grad = RAJA::make_permuted_layout({{ nnodes, dim, nqpts } }, perm3);
   RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > loc_grad_view(loc_grad_data, layout_loc_grad);

   RAJA::Layout<DIM2> layout_jinv = RAJA::make_permuted_layout({{ dim, dim } }, perm2);

   MFEM_FORALL(i_elems, nelems, {
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
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
            const double c_detJ = 1.0 / detJ;
            // adj(J)
            const double A11 = c_detJ * ((J22 * J33) - (J23 * J32));
            const double A12 = c_detJ * ((J32 * J13) - (J12 * J33));
            const double A13 = c_detJ * ((J12 * J23) - (J22 * J13));
            const double A21 = c_detJ * ((J31 * J23) - (J21 * J33));
            const double A22 = c_detJ * ((J11 * J33) - (J13 * J31));
            const double A23 = c_detJ * ((J21 * J13) - (J11 * J23));
            const double A31 = c_detJ * ((J21 * J32) - (J31 * J22));
            const double A32 = c_detJ * ((J31 * J12) - (J11 * J32));
            const double A33 = c_detJ * ((J11 * J22) - (J12 * J21));
            const double A[space_dim2] = { A11, A21, A31, A12, A22, A32, A13, A23, A33 };
            // Raja view to make things easier again

            RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > jinv_view(&A[0], layout_jinv);
            // All of the data down below is ordered in column major order
            for (int t = 0; t < dim; t++) {
               for (int s = 0; s < dim; s++) {
                  for (int r = 0; r < nnodes; r++) {
                     for (int q = 0; q < dim; q++) {
                        vgrad_view(q, t, j_qpts, i_elems) += vel_view(r, q, i_elems) *
                                                             loc_grad_view(r, s, j_qpts) * jinv_view(s, t);
                     }
                  }
               }
            } // End of loop used to calculate velocity gradient
         } // end of forall loop for quadrature points
      }); // end of forall loop for number of elements
} // end of vgrad calc func

TEST(exaconstit, gradient)
{
   int dim = 3;
   mfem::Mesh *mesh;
   // Making this mesh and test real simple with 1 cubic element
   mesh = new mfem::Mesh(2, 2, 2, Element::HEXAHEDRON, 0, 1.0, 1.0, 1.0, false);
   int order = 3;
   mesh->SetCurvature(order);
   mfem::H1_FECollection fec(order, dim);

   mfem::ParMesh *pmesh = NULL;
   pmesh = new mfem::ParMesh(MPI_COMM_WORLD, *mesh);
   mfem::ParFiniteElementSpace fes(pmesh, &fec, dim);

   delete mesh;

   mfem::ParGridFunction x_ref(&fes);
   mfem::ParGridFunction x_cur(&fes);
   mfem::ParGridFunction disp(&fes);

   mesh->GetNodes(x_ref);

   test_deformation_field_set(&x_ref, &disp);

   x_cur = x_ref;
   x_cur += disp;

   {
      // fix me: should the mesh nodes be on the device?
      GridFunction *nodes = &x_cur; // set a nodes grid function to global current configuration
      int owns_nodes = 0;
      pmesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes
   }

   const int intOrder = 2 * order + 1;
   mfem::QuadratureSpace *qspace = new mfem::QuadratureSpace(pmesh, intOrder);
   mfem::QuadratureFunction raderiv(qspace, 9);
   mfem::QuadratureFunction rderiv(qspace, 9);

   const IntegrationRule *ir = &(IntRules.Get(fes.GetFE(0)->GetGeomType(), intOrder));

   const int nqpts = ir->GetNPoints();
   const int ndofs = fes.GetFE(0)->GetDof();
   const int nelems = fes.GetNE();
   const int space_dims = fes.GetFE(0)->GetDim();

   {
      auto coord = mfem::Reshape(raderiv.ReadWrite(), 3, 3, nqpts, nelems);
      // u_vec = (2x + 3y + 4z)i + (4x + 2y + 3z)j + (3x + 4y + 2z)k
      mfem::MFEM_FORALL(i, nelems, {
         for(int j = 0; j < nqpts; j++) {
            coord(0, 0, j, i) = 3.0;
            coord(0, 1, j, i) = 3.0;
            coord(0, 2, j, i) = 4.0;

            coord(1, 0, j, i) = 4.0;
            coord(1, 1, j, i) = 3.0;
            coord(1, 2, j, i) = 3.0;

            coord(2, 0, j, i) = 3.0;
            coord(2, 1, j, i) = 4.0;
            coord(2, 2, j, i) = 3.0;
         }
      });
   }

   {
      // fix me: should the mesh nodes be on the device?
      mfem::GridFunction *nodes = &x_ref; // set a nodes grid function to global current configuration
      int owns_nodes = 0;
      pmesh->SwapNodes(nodes, owns_nodes); // pmesh has current configuration nodes
   }

   {
      // We need to make sure these are deleted at the start of each iteration
      // since we have meshes that are constantly changing.
      pmesh->DeleteGeometricFactors();
      const mfem::GeometricFactors *geom = pmesh->GetGeometricFactors(*ir, mfem::GeometricFactors::JACOBIANS);
      mfem::Vector xcur(fes.TrueVSize());
      x_cur.GetTrueDofs(xcur);

      // So, we're going to originally support non tensor-product type elements originally.
      const mfem::ElementDofOrdering ordering = mfem::ElementDofOrdering::NATIVE;
      // const ElementDofOrdering ordering = ElementDofOrdering::LEXICOGRAPHIC;
      const mfem::Operator *elem_restrict_lex; // Not owned
      const mfem::Operator *P;

      elem_restrict_lex = fes.GetElementRestriction(ordering);
      P = fes.GetProlongationMatrix();

      mfem::Vector el_x;
      el_x.SetSize(elem_restrict_lex->Height(), Device::GetMemoryType());
      mfem::Vector px, el_jac;
      el_x.UseDevice(true);
      px.SetSize(P->Height(), Device::GetMemoryType());
      px.UseDevice(true);
      el_jac.SetSize(space_dims * space_dims * nqpts * nelems, Device::GetMemoryType());
      el_jac.UseDevice(true);

      // Takes in k vector and transforms into into our E-vector array
      P->Mult(xcur, px);
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
         const int nqpts_ = nqpts;
         const int space_dims_ = space_dims;
         for (int j = 0; j < nqpts_; j++) {
            for (int k = 0; k < space_dims_; k++) {
               for (int l = 0; l < space_dims_; l++) {
                  jac_view(l, k, j, i) = geom_j_view(j, l, k, i);
               }
            }
         }
      });

      const FiniteElement &el = *fes.GetFE(0);
      Vector qpts_dshape;
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
      rderiv = 0.0;
      grad_calc(nqpts, nelems, ndofs, el_jac.Read(), qpts_dshape.Read(), el_x.Read(), rderiv.ReadWrite());
   }

   raderiv -= rderiv;

   const double difference = raderiv.Norml2() / raderiv.Size();

   EXPECT_LT(fabs(difference), 3e-15) << "Did not get expected value for pa vec";
}

int main(int argc, char *argv[])
{
   // Initialize MPI.
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   // Testing the case for a dense CMat and then a sparser version of CMat

   Device device("cpu");
   printf("\n");
   device.Print();

   ::testing::InitGoogleTest(&argc, argv);
   if (argc > 1) {
      outputLevel = atoi(argv[1]);
   }
   std::cout << "got outputLevel : " << outputLevel << std::endl;

   int i = RUN_ALL_TESTS();

   MPI_Finalize();

   return i;
}
