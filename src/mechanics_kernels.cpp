#include "mechanics_kernels.hpp"
#include "mfem/general/forall.hpp"

namespace exaconstit{
namespace kernel {

void grad_calc(const int nqpts, const int nelems, const int nnodes,
                const double *jacobian_data, const double *loc_grad_data,
                const double *field_data, double* field_grad_array)
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
    RAJA::Layout<DIM4> layout_grad = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
    RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > field_grad_view(field_grad_array, layout_grad);
    // velocity
    RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{ nnodes, dim, nelems } }, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > field_view(field_data, layout_field);
    // loc_grad
    RAJA::Layout<DIM3> layout_loc_grad = RAJA::make_permuted_layout({{ nnodes, dim, nqpts } }, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > loc_grad_view(loc_grad_data, layout_loc_grad);

    RAJA::Layout<DIM2> layout_jinv = RAJA::make_permuted_layout({{ dim, dim } }, perm2);

    mfem::MFEM_FORALL(i_elems, nelems, {
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
                        field_grad_view(q, t, j_qpts, i_elems) += field_view(r, q, i_elems) *
                                                                loc_grad_view(r, s, j_qpts) * jinv_view(s, t);
                    }
                }
            }
            } // End of loop used to calculate field gradient
        } // end of forall loop for quadrature points
    }); // end of forall loop for number of elements
} // end of kernel_grad_calc

}
}