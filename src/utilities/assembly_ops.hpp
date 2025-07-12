#pragma once

#include "mfem_expt/partial_qfunc.hpp"

#include "mfem.hpp"

/**
 * @brief Construct standard B-matrix for finite element strain-displacement relations.
 * 
 * @param DS Dense matrix containing shape function derivatives in physical coordinates (∂N/∂x)
 * @param B Output B-matrix relating nodal displacements to strain components (modified in place)
 * 
 * This function constructs the standard B-matrix used in finite element assembly
 * operations for computing element stiffness matrices. The B-matrix relates nodal
 * displacements to strain measures through the relationship: strain = B * nodal_displacements.
 * 
 * The function generates the transpose of the traditional B-matrix to better match 
 * MFEM's internal memory layout and vectorization patterns. This organization enables
 * efficient computation of the material tangent stiffness matrix: K = ∫ B^T * C * B dV.
 * 
 * Matrix structure for 3D elements with symmetric material stiffness:
 * - Input DS: (dof × 3) matrix of shape function derivatives
 * - Output B: (3*dof × 6) matrix in Voigt notation order
 * - Strain ordering: [ε_xx, ε_yy, ε_zz, γ_xy, γ_xz, γ_yz]
 * 
 * The B-matrix structure for each node i follows the pattern:
 * ```
 * [∂N_i/∂x    0         0      ]  <- x-displacement DOF
 * [   0    ∂N_i/∂y      0      ]  <- y-displacement DOF  
 * [   0       0      ∂N_i/∂z   ]  <- z-displacement DOF
 * [   0    ∂N_i/∂z   ∂N_i/∂y  ]  <- xy-shear component
 * [∂N_i/∂z     0      ∂N_i/∂x  ]  <- xz-shear component  
 * [∂N_i/∂y  ∂N_i/∂x     0     ]  <- yz-shear component
 * ```
 * 
 * The function constructs the matrix in blocks corresponding to the three spatial
 * dimensions, following MFEM's internal vector ordering: [x₀...xₙ, y₀...yₙ, z₀...zₙ].
 * 
 * @note This function assumes 3D elements and unrolls loops for performance.
 * @note The DS matrix should contain shape function derivatives in physical coordinates.
 * @note The B matrix must be pre-sized to (3*dof, 6) before calling this function.
 * @note For problems with symmetric material stiffness, this generates the standard B-matrix.
 * 
 * @ingroup ExaConstit_utilities_assembly
 */
inline
void
GenerateGradMatrix(const mfem::DenseMatrix& DS, mfem::DenseMatrix& B)
{
    int dof = DS.Height();


    // The B matrix generally has the following structure that is
    // repeated for the number of dofs if we're dealing with something
    // that results in a symmetric Cstiff. If we aren't then it's a different
    // structure
    // [DS(i,0) 0 0]
    // [0 DS(i, 1) 0]
    // [0 0 DS(i, 2)]
    // [0 DS(i,2) DS(i,1)]
    // [DS(i,2) 0 DS(i,0)]
    // [DS(i,1) DS(i,0) 0]

    // Just going to go ahead and make the assumption that
    // this is for a 3D space. Should put either an assert
    // or an error here if it isn't
    // We should also put an assert if B doesn't have dimensions of
    // (dim*dof, 6)
    // fix_me
    // We've rolled out the above B matrix in the comments
    // This is definitely not the most efficient way of doing this memory wise.
    // However, it might be fine for our needs.
    // The ordering has now changed such that B matches up with mfem's internal
    // ordering of vectors such that it's [x0...xn, y0...yn, z0...zn] ordering

    // The previous single loop has been split into 3 so the B matrix
    // is constructed in chunks now instead of performing multiple striding
    // operations in a single loop.
    // x dofs
    for (int i = 0; i < dof; i++) {
        B(i, 0) = DS(i, 0);
        B(i, 1) = 0.0;
        B(i, 2) = 0.0;
        B(i, 3) = 0.0;
        B(i, 4) = DS(i, 2);
        B(i, 5) = DS(i, 1);
    }

    // y dofs
    for (int i = 0; i < dof; i++) {
        B(i + dof, 0) = 0.0;
        B(i + dof, 1) = DS(i, 1);
        B(i + dof, 2) = 0.0;
        B(i + dof, 3) = DS(i, 2);
        B(i + dof, 4) = 0.0;
        B(i + dof, 5) = DS(i, 0);
    }

    // z dofs
    for (int i = 0; i < dof; i++) {
        B(i + 2 * dof, 0) = 0.0;
        B(i + 2 * dof, 1) = 0.0;
        B(i + 2 * dof, 2) = DS(i, 2);
        B(i + 2 * dof, 3) = DS(i, 1);
        B(i + 2 * dof, 4) = DS(i, 0);
        B(i + 2 * dof, 5) = 0.0;
    }
}

/**
 * @brief Construct geometric B-matrix for finite element assembly operations.
 * 
 * @param DS Dense matrix containing shape function derivatives in physical coordinates
 * @param Bgeom Output B-matrix for geometric operations (modified in place)
 * @param dof Number of degrees of freedom per element
 * 
 * This function constructs the geometric B-matrix used in finite element assembly
 * operations, particularly for computing element stiffness matrices and residual
 * vectors. The B-matrix relates nodal displacements to strain measures through
 * the relationship: strain = B * nodal_displacements.
 * 
 * The function builds the B-matrix in blocks corresponding to the three spatial
 * dimensions, following MFEM's internal vector ordering: [x0...xn, y0...yn, z0...zn].
 * This organization is optimized for MFEM's assembly operations and vectorization.
 * 
 * Matrix structure for 3D elements:
 * - Rows: 3*dof (all DOFs for all nodes)
 * - Columns: 9 (components of 3x3 tensor, e.g., stress or strain)
 * - Block structure enables efficient computation of B^T * Sigma * B
 * 
 * The B-matrix can be used in operations like:
 * - K_element = ∫ B^T * C * B dV (stiffness matrix)
 * - F_element = ∫ B^T * σ dV (internal force vector)
 * 
 * where C is the material tangent matrix and σ is the stress tensor.
 * 
 * @note This function assumes 3D elements and unrolls the loops for performance.
 * @note The DS matrix should contain ∂N/∂x derivatives in physical coordinates.
 * @note The Bgeom matrix must be pre-sized to (3*dof, 9) before calling.
 * 
 * @ingroup ExaConstit_utilities_assembly
 */
inline
void
GenerateGradBarMatrix(const mfem::DenseMatrix& DS, const mfem::DenseMatrix& eDS, mfem::DenseMatrix& B)
{
    int dof = DS.Height();

    for (int i = 0; i < dof; i++) {
        const double B1 = (eDS(i, 0) - DS(i, 0)) / 3.0;
        B(i, 0) = B1 + DS(i, 0);
        B(i, 1) = B1;
        B(i, 2) = B1;
        B(i, 3) = 0.0;
        B(i, 4) = DS(i, 2);
        B(i, 5) = DS(i, 1);
    }

    // y dofs
    for (int i = 0; i < dof; i++) {
        const double B2 = (eDS(i, 1) - DS(i, 1)) / 3.0;
        B(i + dof, 0) = B2;
        B(i + dof, 1) = B2 + DS(i, 1);
        B(i + dof, 2) = B2;
        B(i + dof, 3) = DS(i, 2);
        B(i + dof, 4) = 0.0;
        B(i + dof, 5) = DS(i, 0);
    }

    // z dofs
    for (int i = 0; i < dof; i++) {
        const double B3 = (eDS(i, 2) - DS(i, 2)) / 3.0;
        B(i + 2 * dof, 0) = B3;
        B(i + 2 * dof, 1) = B3;
        B(i + 2 * dof, 2) = B3 + DS(i, 2);
        B(i + 2 * dof, 3) = DS(i, 1);
        B(i + 2 * dof, 4) = DS(i, 0);
        B(i + 2 * dof, 5) = 0.0;
    }
}

/**
 * @brief Construct geometric B-matrix for geometric stiffness operations.
 * 
 * @param DS Dense matrix containing shape function derivatives in physical coordinates (∂N/∂x)
 * @param Bgeom Output geometric B-matrix for nonlinear geometric stiffness computations
 * 
 * This function constructs the geometric B-matrix used in finite element assembly
 * for computing geometric stiffness contributions in nonlinear solid mechanics.
 * The geometric B-matrix is essential for capturing nonlinear effects due to
 * large deformations and finite rotations.
 * 
 * The geometric B-matrix is used in operations of the form:
 * K_geom = ∫ B_geom^T * Σ_bar * B_geom dV
 * 
 * where Σ_bar is a block-diagonal stress tensor repeated for each spatial dimension:
 * ```
 * Σ_bar = [σ   0   0  ]
 *         [0   σ   0  ]  
 *         [0   0   σ  ]
 * ```
 * 
 * Matrix structure for 3D elements:
 * - Input DS: (dof × 3) matrix of shape function derivatives
 * - Output Bgeom: (3*dof × 9) matrix organized in spatial dimension blocks
 * - Each block corresponds to x, y, z displacement components
 * 
 * The geometric B-matrix structure repeats the shape function derivatives
 * in each spatial direction:
 * ```
 * Block structure (for node i):
 * x-block: [∂N_i/∂x  ∂N_i/∂y  ∂N_i/∂z  0  0  0  0  0  0]
 * y-block: [0  0  0  ∂N_i/∂x  ∂N_i/∂y  ∂N_i/∂z  0  0  0]
 * z-block: [0  0  0  0  0  0  ∂N_i/∂x  ∂N_i/∂y  ∂N_i/∂z]
 * ```
 * 
 * This formulation enables efficient computation of geometric stiffness terms
 * that arise from the nonlinear strain-displacement relationships in updated
 * Lagrangian finite element formulations.
 * 
 * @note This function assumes 3D elements and is optimized for performance.
 * @note The DS matrix should contain shape function derivatives in physical coordinates.
 * @note The Bgeom matrix must be pre-sized to (3*dof, 9) before calling this function.
 * @note The function follows MFEM's vector ordering: [x₀...xₙ, y₀...yₙ, z₀...zₙ].
 * 
 * @ingroup ExaConstit_utilities_assembly
 */
inline
void
GenerateGradGeomMatrix(const mfem::DenseMatrix& DS, mfem::DenseMatrix& Bgeom)
{
    int dof = DS.Height();
    // For a 3D mesh Bgeom has the following shape:
    // [DS(i, 0), 0, 0]
    // [DS(i, 0), 0, 0]
    // [DS(i, 0), 0, 0]
    // [0, DS(i, 1), 0]
    // [0, DS(i, 1), 0]
    // [0, DS(i, 1), 0]
    // [0, 0, DS(i, 2)]
    // [0, 0, DS(i, 2)]
    // [0, 0, DS(i, 2)]
    // We'll be returning the transpose of this.
    // It turns out the Bilinear operator can't have this created using
    // the dense gradient matrix, DS.
    // It can be used in the following: Bgeom^T Sigma_bar Bgeom
    // where Sigma_bar is a block diagonal version of sigma repeated 3 times in 3D.

    // I'm assumming we're in 3D and have just unrolled the loop
    // The ordering has now changed such that Bgeom matches up with mfem's internal
    // ordering of vectors such that it's [x0...xn, y0...yn, z0...zn] ordering

    // The previous single loop has been split into 3 so the B matrix
    // is constructed in chunks now instead of performing multiple striding
    // operations in a single loop.

    // x dofs
    for (int i = 0; i < dof; i++) {
        Bgeom(i, 0) = DS(i, 0);
        Bgeom(i, 1) = DS(i, 1);
        Bgeom(i, 2) = DS(i, 2);
        Bgeom(i, 3) = 0.0;
        Bgeom(i, 4) = 0.0;
        Bgeom(i, 5) = 0.0;
        Bgeom(i, 6) = 0.0;
        Bgeom(i, 7) = 0.0;
        Bgeom(i, 8) = 0.0;
    }

    // y dofs
    for (int i = 0; i < dof; i++) {
        Bgeom(i + dof, 0) = 0.0;
        Bgeom(i + dof, 1) = 0.0;
        Bgeom(i + dof, 2) = 0.0;
        Bgeom(i + dof, 3) = DS(i, 0);
        Bgeom(i + dof, 4) = DS(i, 1);
        Bgeom(i + dof, 5) = DS(i, 2);
        Bgeom(i + dof, 6) = 0.0;
        Bgeom(i + dof, 7) = 0.0;
        Bgeom(i + dof, 8) = 0.0;
    }

    // z dofs
    for (int i = 0; i < dof; i++) {
        Bgeom(i + 2 * dof, 0) = 0.0;
        Bgeom(i + 2 * dof, 1) = 0.0;
        Bgeom(i + 2 * dof, 2) = 0.0;
        Bgeom(i + 2 * dof, 3) = 0.0;
        Bgeom(i + 2 * dof, 4) = 0.0;
        Bgeom(i + 2 * dof, 5) = 0.0;
        Bgeom(i + 2 * dof, 6) = DS(i, 0);
        Bgeom(i + 2 * dof, 7) = DS(i, 1);
        Bgeom(i + 2 * dof, 8) = DS(i, 2);
    }
}


/**
 * @brief Get quadrature function data at a specific element and integration point.
 * 
 * @param elID Global element index
 * @param ipNum Integration point number within the element
 * @param qfdata Output array to store the retrieved data
 * @param qf Shared pointer to the PartialQuadratureFunction
 * 
 * This function extracts data from a PartialQuadratureFunction at a specific
 * element and integration point. It handles the indexing and memory layout
 * automatically, providing a convenient interface for accessing quadrature
 * point data during assembly operations.
 * 
 * The function:
 * 1. Computes the correct offset based on element ID and integration point
 * 2. Accounts for the vector dimension of the quadrature function
 * 3. Copies the data to the provided output array
 * 4. Handles both full and partial quadrature spaces transparently
 * 
 * Data layout assumptions:
 * - Data is stored element-by-element
 * - Within each element, data is stored point-by-point
 * - Within each point, components are stored sequentially
 * 
 * Usage example:
 * @code
 * double stress[6];  // For symmetric stress tensor
 * GetQFData(elem_id, qp_id, stress, stress_qf);
 * // stress[0] = σ_xx, stress[1] = σ_yy, etc.
 * @endcode
 * 
 * @note The qfdata array must be pre-allocated with size qf->GetVDim().
 * @note This function uses host-side memory access patterns.
 * 
 * @ingroup ExaConstit_utilities_assembly
 */
inline
void
GetQFData(const int elID, const int ipNum, double* qfdata, std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf)
{   
    const auto data = qf->HostRead();
    const int qf_offset = qf->GetVDim();
    auto qspace = qf->GetSpaceShared();
 
    const mfem::IntegrationRule *ir = &(qf->GetSpaceShared()->GetIntRule(elID));
    int elem_offset = qf_offset * ir->GetNPoints();
 
    for (int i = 0; i < qf_offset; ++i) {
        qfdata[i] = data[elID * elem_offset + ipNum * qf_offset + i];
    }
}


/**
 * @brief Set quadrature function data at a specific element and integration point.
 * 
 * @param elID Global element index
 * @param ipNum Integration point number within the element
 * @param qfdata Input array containing the data to store
 * @param qf Shared pointer to the PartialQuadratureFunction
 * 
 * This function stores data into a PartialQuadratureFunction at a specific
 * element and integration point. It provides the complementary operation to
 * GetQFData(), enabling efficient storage of computed values during assembly.
 * 
 * The function:
 * 1. Computes the correct offset based on element ID and integration point
 * 2. Accounts for the vector dimension of the quadrature function
 * 3. Copies the data from the input array to the quadrature function
 * 4. Handles both full and partial quadrature spaces transparently
 * 
 * This function is commonly used to store:
 * - Updated stress tensors after material model evaluation
 * - Computed material tangent stiffness matrices
 * - State variables and internal variables
 * - Derived quantities like plastic strain
 * 
 * Usage example:
 * @code
 * double new_stress[6] = {s11, s22, s33, s12, s13, s23};
 * SetQFData(elem_id, qp_id, new_stress, stress_qf);
 * @endcode
 * 
 * @note The qfdata array must contain qf->GetVDim() values.
 * @note This function uses host-side memory access patterns.
 * @note Data is written directly to the quadrature function's internal storage.
 * 
 * @ingroup ExaConstit_utilities_assembly
 */
inline
void
SetQFData(const int elID, const int ipNum, double* qfdata, std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf)
{   
    auto data = qf->HostReadWrite();
    const int qf_offset = qf->GetVDim();
    auto qspace = qf->GetSpaceShared();
 
    const mfem::IntegrationRule *ir = &(qf->GetSpaceShared()->GetIntRule(elID));
    int elem_offset = qf_offset * ir->GetNPoints();
 
    for (int i = 0; i < qf_offset; ++i) {
        data[elID * elem_offset + ipNum * qf_offset + i] = qfdata[i];
    }
}

/**
 * @brief Transform material gradient to 4D layout for partial assembly.
 * 
 * @param matGrad Shared pointer to material gradient PartialQuadratureFunction
 * @param matGradPA Output vector with 4D layout for partial assembly
 * 
 * This function transforms material gradient data (typically tangent stiffness
 * matrices) from the standard quadrature function layout to a 4D layout
 * optimized for MFEM's partial assembly operations.
 * 
 * The transformation reorganizes data to enable efficient vectorized operations
 * during partial assembly, where material properties are applied element-wise
 * rather than globally assembled into a sparse matrix.
 * 
 * Layout transformation:
 * - Input: Standard QF layout with material gradients per quadrature point
 * - Output: 4D RAJA view layout optimized for partial assembly kernels
 * - Uses permuted layouts to optimize memory access patterns
 * 
 * The function uses RAJA views with specific permutations to:
 * 1. Optimize cache performance for the target architecture
 * 2. Enable vectorization in assembly kernels
 * 3. Support both CPU and GPU execution
 * 
 * This transformation is essential for high-performance partial assembly
 * operations in ExaConstit's finite element solver.
 * 
 * @note The matGradPA vector is resized automatically to accommodate the data.
 * @note The function assumes 3D problems with 6x6 material tangent matrices.
 * @note RAJA views use specific permutations for optimal performance.
 * 
 * @ingroup ExaConstit_utilities_assembly
 */
inline
void
TransformMatGradTo4D(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> matGrad, mfem::Vector& matGradPA)
{
    const int npts = matGrad->Size() / matGrad->GetVDim();
 
    const int dim = 3;
    const int dim2 = 6;
 
    const int DIM5 = 5;
    const int DIM3 = 3;
    std::array<RAJA::idx_t, DIM5> perm5 {{ 4, 3, 2, 1, 0 } };
    std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };
 
    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
    RAJA::Layout<DIM5> layout_4Dtensor = RAJA::make_permuted_layout({{ dim, dim, dim, dim, npts } }, perm5);
    RAJA::View<double, RAJA::Layout<DIM5, RAJA::Index_type, 0> > cmat_4d(matGradPA.ReadWrite(), layout_4Dtensor);
 
    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
    RAJA::Layout<DIM3> layout_2Dtensor = RAJA::make_permuted_layout({{ dim2, dim2, npts } }, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > cmat(matGrad->Read(), layout_2Dtensor);
 
    // This sets up our 4D tensor to be the same as the 2D tensor which takes advantage of symmetry operations
    mfem::forall(npts, [=] MFEM_HOST_DEVICE (int i) {
       cmat_4d(0, 0, 0, 0, i) = cmat(0, 0, i);
       cmat_4d(1, 1, 0, 0, i) = cmat(1, 0, i);
       cmat_4d(2, 2, 0, 0, i) = cmat(2, 0, i);
       cmat_4d(1, 2, 0, 0, i) = cmat(3, 0, i);
       cmat_4d(2, 1, 0, 0, i) = cmat_4d(1, 2, 0, 0, i);
       cmat_4d(2, 0, 0, 0, i) = cmat(4, 0, i);
       cmat_4d(0, 2, 0, 0, i) = cmat_4d(2, 0, 0, 0, i);
       cmat_4d(0, 1, 0, 0, i) = cmat(5, 0, i);
       cmat_4d(1, 0, 0, 0, i) = cmat_4d(0, 1, 0, 0, i);
 
       cmat_4d(0, 0, 1, 1, i) = cmat(0, 1, i);
       cmat_4d(1, 1, 1, 1, i) = cmat(1, 1, i);
       cmat_4d(2, 2, 1, 1, i) = cmat(2, 1, i);
       cmat_4d(1, 2, 1, 1, i) = cmat(3, 1, i);
       cmat_4d(2, 1, 1, 1, i) = cmat_4d(1, 2, 1, 1, i);
       cmat_4d(2, 0, 1, 1, i) = cmat(4, 1, i);
       cmat_4d(0, 2, 1, 1, i) = cmat_4d(2, 0, 1, 1, i);
       cmat_4d(0, 1, 1, 1, i) = cmat(5, 1, i);
       cmat_4d(1, 0, 1, 1, i) = cmat_4d(0, 1, 1, 1, i);
 
       cmat_4d(0, 0, 2, 2, i) = cmat(0, 2, i);
       cmat_4d(1, 1, 2, 2, i) = cmat(1, 2, i);
       cmat_4d(2, 2, 2, 2, i) = cmat(2, 2, i);
       cmat_4d(1, 2, 2, 2, i) = cmat(3, 2, i);
       cmat_4d(2, 1, 2, 2, i) = cmat_4d(1, 2, 2, 2, i);
       cmat_4d(2, 0, 2, 2, i) = cmat(4, 2, i);
       cmat_4d(0, 2, 2, 2, i) = cmat_4d(2, 0, 2, 2, i);
       cmat_4d(0, 1, 2, 2, i) = cmat(5, 2, i);
       cmat_4d(1, 0, 2, 2, i) = cmat_4d(0, 1, 2, 2, i);
 
       cmat_4d(0, 0, 1, 2, i) = cmat(0, 3, i);
       cmat_4d(1, 1, 1, 2, i) = cmat(1, 3, i);
       cmat_4d(2, 2, 1, 2, i) = cmat(2, 3, i);
       cmat_4d(1, 2, 1, 2, i) = cmat(3, 3, i);
       cmat_4d(2, 1, 1, 2, i) = cmat_4d(1, 2, 1, 2, i);
       cmat_4d(2, 0, 1, 2, i) = cmat(4, 3, i);
       cmat_4d(0, 2, 1, 2, i) = cmat_4d(2, 0, 1, 2, i);
       cmat_4d(0, 1, 1, 2, i) = cmat(5, 3, i);
       cmat_4d(1, 0, 1, 2, i) = cmat_4d(0, 1, 1, 2, i);
 
       cmat_4d(0, 0, 2, 1, i) = cmat(0, 3, i);
       cmat_4d(1, 1, 2, 1, i) = cmat(1, 3, i);
       cmat_4d(2, 2, 2, 1, i) = cmat(2, 3, i);
       cmat_4d(1, 2, 2, 1, i) = cmat(3, 3, i);
       cmat_4d(2, 1, 2, 1, i) = cmat_4d(1, 2, 1, 2, i);
       cmat_4d(2, 0, 2, 1, i) = cmat(4, 3, i);
       cmat_4d(0, 2, 2, 1, i) = cmat_4d(2, 0, 1, 2, i);
       cmat_4d(0, 1, 2, 1, i) = cmat(5, 3, i);
       cmat_4d(1, 0, 2, 1, i) = cmat_4d(0, 1, 1, 2, i);
 
       cmat_4d(0, 0, 2, 0, i) = cmat(0, 4, i);
       cmat_4d(1, 1, 2, 0, i) = cmat(1, 4, i);
       cmat_4d(2, 2, 2, 0, i) = cmat(2, 4, i);
       cmat_4d(1, 2, 2, 0, i) = cmat(3, 4, i);
       cmat_4d(2, 1, 2, 0, i) = cmat_4d(1, 2, 2, 0, i);
       cmat_4d(2, 0, 2, 0, i) = cmat(4, 4, i);
       cmat_4d(0, 2, 2, 0, i) = cmat_4d(2, 0, 2, 0, i);
       cmat_4d(0, 1, 2, 0, i) = cmat(5, 4, i);
       cmat_4d(1, 0, 2, 0, i) = cmat_4d(0, 1, 2, 0, i);
 
       cmat_4d(0, 0, 0, 2, i) = cmat(0, 4, i);
       cmat_4d(1, 1, 0, 2, i) = cmat(1, 4, i);
       cmat_4d(2, 2, 0, 2, i) = cmat(2, 4, i);
       cmat_4d(1, 2, 0, 2, i) = cmat(3, 4, i);
       cmat_4d(2, 1, 0, 2, i) = cmat_4d(1, 2, 2, 0, i);
       cmat_4d(2, 0, 0, 2, i) = cmat(4, 4, i);
       cmat_4d(0, 2, 0, 2, i) = cmat_4d(2, 0, 2, 0, i);
       cmat_4d(0, 1, 0, 2, i) = cmat(5, 4, i);
       cmat_4d(1, 0, 0, 2, i) = cmat_4d(0, 1, 2, 0, i);
 
       cmat_4d(0, 0, 0, 1, i) = cmat(0, 5, i);
       cmat_4d(1, 1, 0, 1, i) = cmat(1, 5, i);
       cmat_4d(2, 2, 0, 1, i) = cmat(2, 5, i);
       cmat_4d(1, 2, 0, 1, i) = cmat(3, 5, i);
       cmat_4d(2, 1, 0, 1, i) = cmat_4d(1, 2, 0, 1, i);
       cmat_4d(2, 0, 0, 1, i) = cmat(4, 5, i);
       cmat_4d(0, 2, 0, 1, i) = cmat_4d(2, 0, 0, 1, i);
       cmat_4d(0, 1, 0, 1, i) = cmat(5, 5, i);
       cmat_4d(1, 0, 0, 1, i) = cmat_4d(0, 1, 0, 1, i);
 
       cmat_4d(0, 0, 1, 0, i) = cmat(0, 5, i);
       cmat_4d(1, 1, 1, 0, i) = cmat(1, 5, i);
       cmat_4d(2, 2, 1, 0, i) = cmat(2, 5, i);
       cmat_4d(1, 2, 1, 0, i) = cmat(3, 5, i);
       cmat_4d(2, 1, 1, 0, i) = cmat_4d(1, 2, 0, 1, i);
       cmat_4d(2, 0, 1, 0, i) = cmat(4, 5, i);
       cmat_4d(0, 2, 1, 0, i) = cmat_4d(2, 0, 0, 1, i);
       cmat_4d(0, 1, 1, 0, i) = cmat(5, 5, i);
       cmat_4d(1, 0, 1, 0, i) = cmat_4d(0, 1, 0, 1, i);
    });
 }
