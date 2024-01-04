#ifndef MECHANICS_KERNELS
#define MECHANICS_KERNELS

#include "mfem.hpp"
#include "RAJA/RAJA.hpp"
#include "option_types.hpp"
#include "mfem/general/forall.hpp"

namespace exaconstit {
namespace kernel {
/// Performs all the calculations related to calculating the gradient of a 3D vector field
/// grad_array should be set to 0.0 outside of this function.
//  It is assumed that whatever data pointers being passed in is consistent with
//  with the execution strategy being used by the MFEM_FORALL.
void grad_calc(const int nqpts, const int nelems, const int nnodes,
                const double *jacobian_data, const double *loc_grad_data,
                const double *field_data, double* field_grad_array);
//Computes the volume average values of values that lie at the quadrature points
template<bool vol_avg>
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
        const int nqpts_ = nqpts;
        for (int j = 0; j < nqpts_; j++) {
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
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (class_device == RTModel::GPU) {
        const double* qf_data = qf->Read();
        const double* wts_data = wts.Read();
#if defined(RAJA_ENABLE_CUDA)
        using gpu_reduce = RAJA::cuda_reduce;
        using gpu_policy = RAJA::cuda_exec<1024>;
#else
        using gpu_reduce = RAJA::hip_reduce;
        using gpu_policy = RAJA::hip_exec<1024>;
#endif
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<gpu_reduce, double> gpu_sum(0.0);
            RAJA::ReduceSum<gpu_reduce, double> vol_sum(0.0);
            RAJA::forall<gpu_policy>(default_range, [ = ] RAJA_DEVICE(int i_npts){
                const double* val = &(qf_data[i_npts * size]);
                gpu_sum += wts_data[i_npts] * val[j];
                vol_sum += wts_data[i_npts];
            });
            data[j] = gpu_sum.get();
            el_vol = vol_sum.get();
        }
    }
#endif

    for (int i = 0; i < size; i++) {
        tensor[i] = data[i];
    }

    MPI_Allreduce(&data, tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (vol_avg) {
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
}
#endif
