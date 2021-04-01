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

void ComputeVolAvgTensor(const mfem::ParFiniteElementSpace* fes,
                         const mfem::QuadratureFunction* qf,
                         mfem::Vector& tensor, int size,
                         RTModel &class_device)
{
    mfem::Mesh *mesh = fes->GetMesh();
    const mfem::FiniteElement &el = *fes->GetFE(0);
    const mfem::IntegrationRule *ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));;

    const int nqpts = ir->GetNPoints();
    const int nelems = fes->GetNE();
    const int npts = nqpts * nelems;

    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors *geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::DETERMINANTS);

    double el_vol = 0.0;
    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    double data[size];

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);

    mfem::Vector wts(geom->detJ);
    RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > wts_view(wts.ReadWrite(), layout_geom);
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);

    RAJA::RangeSegment default_range(0, npts);

    mfem::MFEM_FORALL(i, nelems, {
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
                const double* val = &(qf_data[i_npts * size]);
                seq_sum += wts_data[i_npts] * val[j];
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
                const double* val = &(qf_data[i_npts * size]);
                omp_sum += wts_data[i_npts] * val[j];
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
                const double* val = &(qf_data[i_npts * size]);
                cuda_sum += wts_data[i_npts] * val[j];
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

}
}