#ifndef MECHANICS_KERNELS
#define MECHANICS_KERNELS

#include "mfem.hpp"
#include "RAJA/RAJA.hpp"
#include "option_types.hpp"

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
void ComputeVolAvgTensor(const mfem::ParFiniteElementSpace* fes,
                        const mfem::QuadratureFunction* qf,
                        mfem::Vector& tensor, int size,
                        RTModel &class_device);
}
}
#endif