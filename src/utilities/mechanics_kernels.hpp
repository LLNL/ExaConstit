#ifndef MECHANICS_KERNELS
#define MECHANICS_KERNELS

#include "mfem.hpp"
#include "RAJA/RAJA.hpp"
#include "options/option_parser_v2.hpp"
#include "mfem/general/forall.hpp"
#include "mfem_expt/partial_qfunc.hpp"

namespace exaconstit {
namespace kernel {

/// Main gradient calculation function with partial element mapping support
/// @param nqpts Number of quadrature points per element
/// @param nelems Number of local elements to process
/// @param global_nelems Total number of elements in global arrays (for input data sizing)
/// @param nnodes Number of nodes per element  
/// @param jacobian_data Global jacobian data array
/// @param loc_grad_data Global local gradient data array
/// @param field_data Global field data array
/// @param field_grad_array Local output array (sized for local elements)
/// @param local2global Optional mapping from local to global element indices
void grad_calc(const int nqpts, const int nelems, const int global_nelems, const int nnodes,
               const double *jacobian_data, const double *loc_grad_data,
               const double *field_data, double* field_grad_array,
               const mfem::Array<int>* local2global = nullptr);

/// Backward compatibility overload - assumes full element processing (no partial mapping)
/// @param nqpts Number of quadrature points per element
/// @param nelems Number of elements to process
/// @param nnodes Number of nodes per element
/// @param jacobian_data Jacobian data array
/// @param loc_grad_data Local gradient data array
/// @param field_data Field data array
/// @param field_grad_array Output gradient array
inline
void grad_calc(const int nqpts, const int nelems, const int nnodes,
               const double *jacobian_data, const double *loc_grad_data,
               const double *field_data, double* field_grad_array)
{
    // Call the full version with no partial mapping (backward compatibility)
    grad_calc(nqpts, nelems, nelems, nnodes, jacobian_data, loc_grad_data, 
    field_data, field_grad_array, nullptr);
}

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
    mfem::Vector data(size);

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
            RAJA::forall<RAJA::seq_exec>(default_range, [ = ] (int i_npts){
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

    MPI_Allreduce(data.HostRead(), tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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

//Computes the volume average values of values that lie at the quadrature points
//but only computes the values that aren't filtered out
// aka It only includes values that are set to true in filter
// It also returns the volume that corresponds to the values that were filtered
template<bool vol_avg>
double ComputeVolAvgTensorFilter(const mfem::ParFiniteElementSpace* fes,
                                 const mfem::QuadratureFunction* qf,
                                 const mfem::Array<bool>* filter,
                                 mfem::Vector& tensor, int size,
                                 const RTModel &class_device)
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
    mfem::Vector data(size);

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
        const bool* filter_data = filter->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::seq_reduce, double> seq_sum(0.0);
            RAJA::ReduceSum<RAJA::seq_reduce, double> vol_sum(0.0);
            RAJA::forall<RAJA::seq_exec>(default_range, [ = ] (int i_npts){
                if (!filter_data[i_npts]) return;
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
        const bool* filter_data = filter->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> omp_sum(0.0);
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> vol_sum(0.0);
            RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [ = ] (int i_npts){
                if (!filter_data[i_npts]) return;
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
        const bool* filter_data = filter->Read();
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
                if (!filter_data[i_npts]) return;
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

    MPI_Allreduce(data.HostRead(), tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double temp = el_vol;
    // Here we find what el_vol should be equal to
    MPI_Allreduce(&temp, &el_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (vol_avg) {
        // We meed to multiple by 1/V by our tensor values to get the appropriate
        // average value for the tensor in the end.
        double inv_vol = (fabs(el_vol) > 1e-14) ? 1.0 / el_vol : 0.0;

        for (int m = 0; m < size; m++) {
            tensor[m] *= inv_vol;
        }
    }
    return el_vol;
}

/**
 * @brief Compute volume average directly from PartialQuadratureFunction
 * 
 * This function works directly with PartialQuadratureFunction data which already
 * contains only the elements for a specific region. No filtering is needed.
 * 
 * @param pqf Partial quadrature function containing region-specific data
 * @param tensor Output vector for volume-averaged values
 * @param size Number of components per quadrature point
 * @param rtmodel Runtime model for device execution
 * @return Total volume of the region
 */
template<bool vol_avg>
double ComputeVolAvgTensorFilterFromPartial(const mfem::expt::PartialQuadratureFunction* pqf,
                                            const mfem::Array<bool>* filter,
                                            mfem::Vector& tensor, int size,
                                            const RTModel &class_device)
{
    auto pqs = pqf->GetPartialSpaceShared();
    auto mesh = pqs->GetMeshShared();

    // Get finite element and integration rule info
    // Note: We need to get this from the global finite element space since
    // the PartialQuadratureSpace doesn't have direct FE access
    const int fe_order = pqs->GetOrder();
    mfem::Geometry::Type geom_type = mesh->GetElementBaseGeometry(0); // Assume uniform elements
    const mfem::IntegrationRule *ir = &(mfem::IntRules.Get(geom_type, fe_order));
    
    const int nqpts = ir->GetNPoints();
    const int local_nelems = pqs->GetNE(); // Number of elements in this partial space
    const int nelems = mesh->GetNE();
    
    // Verify size matches vdim
#if defined(MFEM_USE_DEBUG)
    const int vdim = pqf->GetVDim();
    MFEM_ASSERT(size == vdim, "Size parameter must match quadrature function vector dimension");
#endif
    
    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors *geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::DETERMINANTS);
    
    // Get the local-to-global element mapping and data layout info
    auto l2g = pqs->getLocal2Global().Read();           // Maps local element index to global element index
    auto loc_offsets = pqs->getOffsets().Read();        // Offsets for local data layout
    auto global_offsets = (pqs->getGlobalOffset().Size() > 1) ? 
                         pqs->getGlobalOffset().Read() : loc_offsets; // Offsets for global data layout

    double el_vol = 0.0;
    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    mfem::Vector data(size);

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);

    mfem::Vector wts(geom->detJ);
    RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > wts_view(wts.ReadWrite(), layout_geom);
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);

    RAJA::RangeSegment default_range(0, local_nelems);

    mfem::MFEM_FORALL(i, nelems, {
        const int nqpts_ = nqpts;
        for (int j = 0; j < nqpts_; j++) {
            wts_view(j, i) = j_view(j, i) * W[j];
        }
    });

    if (class_device == RTModel::CPU) {
        const double* qf_data = pqf->HostRead();
        const bool* filter_data = filter->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::seq_reduce, double> data_sum(0.0);
            RAJA::ReduceSum<RAJA::seq_reduce, double> vol_sum(0.0);
            RAJA::forall<RAJA::seq_exec>(default_range, [ = ] (int ie) {
                const int global_elem = l2g[ie];             // Map local element to global element
                const int local_offset = loc_offsets[ie];     // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] - local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];

                for (int k = 0; k < npts_elem; k++) {
                    if (!filter_data[local_offset + k]) continue;
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    data_sum += wts_data[global_offset + k ] * val[j];
                    vol_sum += wts_data[global_offset + k ];
                }
            });
            data[j] = data_sum.get();
            el_vol = vol_sum.get();
        }
    }
#if defined(RAJA_ENABLE_OPENMP)
    if (class_device == RTModel::OPENMP) {
        const double* qf_data = pqf->HostRead();
        const bool* filter_data = filter->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> data_sum(0.0);
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> vol_sum(0.0);
            RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [ = ] (int ie) {
                const int global_elem = l2g[ie];             // Map local element to global element
                const int local_offset = loc_offsets[ie];     // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] - local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];

                for (int k = 0; k < npts_elem; k++) {
                    if (!filter_data[local_offset + k]) continue;
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    data_sum += wts_data[global_offset + k ] * val[j];
                    vol_sum += wts_data[global_offset + k ];
                }
            });
            data[j] = data_sum.get();
            el_vol = vol_sum.get();
        }
    }
#endif
#if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (class_device == RTModel::GPU) {
        const double* qf_data = pqf->Read();
        const bool* filter_data = filter->Read();
        const double* wts_data = wts.Read();
#if defined(RAJA_ENABLE_CUDA)
        using gpu_reduce = RAJA::cuda_reduce;
        using gpu_policy = RAJA::cuda_exec<1024>;
#else
        using gpu_reduce = RAJA::hip_reduce;
        using gpu_policy = RAJA::hip_exec<1024>;
#endif
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<gpu_reduce, double> data_sum(0.0);
            RAJA::ReduceSum<gpu_reduce, double> vol_sum(0.0);
            RAJA::forall<gpu_policy>(default_range, [ = ] RAJA_DEVICE(int ie){
                const int global_elem = l2g[ie];             // Map local element to global element
                const int local_offset = loc_offsets[ie];     // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] - local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];

                for (int k = 0; k < npts_elem; k++) {
                    if (!filter_data[local_offset + k]) continue;
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    data_sum += wts_data[global_offset + k ] * val[j];
                    vol_sum += wts_data[global_offset + k ];
                }
            });
            data[j] = data_sum.get();
            el_vol = vol_sum.get();
        }
    }
#endif

    for (int i = 0; i < size; i++) {
        tensor[i] = data[i];
    }

    MPI_Allreduce(data.HostRead(), tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double temp = el_vol;
    // Here we find what el_vol should be equal to
    MPI_Allreduce(&temp, &el_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (vol_avg) {
        // We meed to multiple by 1/V by our tensor values to get the appropriate
        // average value for the tensor in the end.
        double inv_vol = (fabs(el_vol) > 1e-14) ? 1.0 / el_vol : 0.0;

        for (int m = 0; m < size; m++) {
            tensor[m] *= inv_vol;
        }
    }
    return el_vol;
}

/**
 * @brief Compute volume average directly from PartialQuadratureFunction
 * 
 * This function works directly with PartialQuadratureFunction data which already
 * contains only the elements for a specific region. No filtering is needed.
 * 
 * @param pqf Partial quadrature function containing region-specific data
 * @param tensor Output vector for volume-averaged values
 * @param size Number of components per quadrature point
 * @param rtmodel Runtime model for device execution
 * @return Total volume of the region
 */
template<bool vol_avg>
double ComputeVolAvgTensorFromPartial(const mfem::expt::PartialQuadratureFunction* pqf,
                                     mfem::Vector& tensor, int size,
                                     const RTModel &class_device)
{
    auto pqs = pqf->GetPartialSpaceShared();
    auto mesh = pqs->GetMeshShared();
    
    // Get finite element and integration rule info
    // Note: We need to get this from the global finite element space since
    // the PartialQuadratureSpace doesn't have direct FE access
    const int fe_order = pqs->GetOrder();
    mfem::Geometry::Type geom_type = mesh->GetElementBaseGeometry(0); // Assume uniform elements
    const mfem::IntegrationRule *ir = &(mfem::IntRules.Get(geom_type, fe_order));
    
    const int nqpts = ir->GetNPoints();
    const int local_nelems = pqs->GetNE(); // Number of elements in this partial space
    const int nelems = mesh->GetNE();
    
    // Verify size matches vdim
#if defined(MFEM_USE_DEBUG)
    const int vdim = pqf->GetVDim();
    MFEM_ASSERT(size == vdim, "Size parameter must match quadrature function vector dimension");
#endif
    
    const double* W = ir->GetWeights().Read();
    const mfem::GeometricFactors *geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::DETERMINANTS);
    
    // Get the local-to-global element mapping and data layout info
    auto l2g = pqs->getLocal2Global().Read();           // Maps local element index to global element index
    auto loc_offsets = pqs->getOffsets().Read();        // Offsets for local data layout
    auto global_offsets = (pqs->getGlobalOffset().Size() > 1) ? 
                         pqs->getGlobalOffset().Read() : loc_offsets; // Offsets for global data layout
        
    // Initialize output tensor and volume
    tensor.SetSize(size);
    tensor = 0.0;
    double total_volume = 0.0;

    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    mfem::Vector data(size);

    const int DIM2 = 2;
    std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
    RAJA::Layout<DIM2> layout_geom = RAJA::make_permuted_layout({{ nqpts, nelems } }, perm2);

    mfem::Vector wts(geom->detJ);
    RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > wts_view(wts.ReadWrite(), layout_geom);
    RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > j_view(geom->detJ.Read(), layout_geom);

    RAJA::RangeSegment default_range(0, local_nelems);

    double vol_sum = 0.0;
    double* vpt = &vol_sum;
    mfem::MFEM_FORALL(i, nelems, {
        const int nqpts_ = nqpts;
        for (int j = 0; j < nqpts_; j++) {
            wts_view(j, i) = j_view(j, i) * W[j];
            *vpt += wts_view(j, i);
        }
    });

    if (class_device == RTModel::CPU) {
        const double* qf_data = pqf->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::seq_reduce, double> seq_sum(0.0);
            RAJA::ReduceSum<RAJA::seq_reduce, double> vol_sum(0.0);
            RAJA::forall<RAJA::seq_exec>(default_range, [ = ] (int ie) {
                const int global_elem = l2g[ie];             // Map local element to global element
                const int local_offset = loc_offsets[ie];     // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] - local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];
                for (int k = 0; k < npts_elem; k++) {
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    seq_sum += wts_data[global_offset + k ] * val[j];
                    vol_sum += wts_data[global_offset + k ];
                }
            });
            data[j] = seq_sum.get();
            total_volume = vol_sum.get();
        }
    }
    #if defined(RAJA_ENABLE_OPENMP)
    if (class_device == RTModel::OPENMP) {
        const double* qf_data = pqf->HostRead();
        const double* wts_data = wts.HostRead();
        for (int j = 0; j < size; j++) {
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> omp_sum(0.0);
            RAJA::ReduceSum<RAJA::omp_reduce_ordered, double> vol_sum(0.0);
            RAJA::forall<RAJA::omp_parallel_for_exec>(default_range, [ = ] (int i_npts){
                const int global_elem = l2g[ie];             // Map local element to global element
                const int local_offset = loc_offsets[ie];     // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] - local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];
                for (int k = 0; k < npts_elem; k++) {
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    omp_sum += wts_data[global_offset + k ] * val[j];
                    vol_sum += wts_data[global_offset + k ];
                }
            });
            data[j] = omp_sum.get();
            total_volume = vol_sum.get();
        }
    }
    #endif
    #if defined(RAJA_ENABLE_CUDA) || defined(RAJA_ENABLE_HIP)
    if (class_device == RTModel::GPU) {
        const double* qf_data = pqf->Read();
        const bool* filter_data = filter->Read();
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
                const int global_elem = l2g[ie];             // Map local element to global element
                const int local_offset = loc_offsets[ie];     // Offset into local data array
                const int npts_elem = loc_offsets[ie + 1] - local_offset; // Number of qpts for this element
                const int global_offset = global_offsets[global_elem];
                for (int k = 0; k < npts_elem; k++) {
                    const double* val = &(qf_data[local_offset * size + k * size]);
                    gpu_sum += wts_data[global_offset + k ] * val[j];
                    vol_sum += wts_data[global_offset + k ];
                }
            });
            data[j] = gpu_sum.get();
            total_volume = vol_sum.get();
        }
    }
    #endif

    for (int i = 0; i < size; i++) {
        tensor[i] = data[i];
    }

    MPI_Allreduce(data.HostRead(), tensor.HostReadWrite(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double temp = total_volume;
    // Here we find what el_vol should be equal to
    MPI_Allreduce(&temp, &total_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (vol_avg) {
        // We meed to multiple by 1/V by our tensor values to get the appropriate
        // average value for the tensor in the end.
        double inv_vol = (fabs(total_volume) > 1e-14) ? 1.0 / total_volume : 0.0;

        for (int m = 0; m < size; m++) {
            tensor[m] *= inv_vol;
        }
    }
    return total_volume;
}

}
}
#endif
