#pragma once

#include "mfem_expt/partial_qfunc.hpp"

#include "mfem.hpp"

/**
 * @brief Construct standard B-matrix for finite element strain-displacement relations.
 *
 * @param deriv_shapes Dense matrix containing shape function derivatives in physical coordinates
 * (∂N/∂x)
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
 * - Input deriv_shapes: (dof × 3) matrix of shape function derivatives
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
 * @note The deriv_shapes matrix should contain shape function derivatives in physical coordinates.
 * @note The B matrix must be pre-sized to (3*dof, 6) before calling this function.
 * @note For problems with symmetric material stiffness, this generates the standard B-matrix.
 *
 * @ingroup ExaConstit_utilities_assembly
 */
inline void GenerateGradMatrix(const mfem::DenseMatrix& deriv_shapes, mfem::DenseMatrix& B) {
    int dof = deriv_shapes.Height();

    // The B matrix generally has the following structure that is
    // repeated for the number of dofs if we're dealing with something
    // that results in a symmetric Cstiff. If we aren't then it's a different
    // structure
    // [deriv_shapes(i,0) 0 0]
    // [0 deriv_shapes(i, 1) 0]
    // [0 0 deriv_shapes(i, 2)]
    // [0 deriv_shapes(i,2) deriv_shapes(i,1)]
    // [deriv_shapes(i,2) 0 deriv_shapes(i,0)]
    // [deriv_shapes(i,1) deriv_shapes(i,0) 0]

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
        B(i, 0) = deriv_shapes(i, 0);
        B(i, 1) = 0.0;
        B(i, 2) = 0.0;
        B(i, 3) = 0.0;
        B(i, 4) = deriv_shapes(i, 2);
        B(i, 5) = deriv_shapes(i, 1);
    }

    // y dofs
    for (int i = 0; i < dof; i++) {
        B(i + dof, 0) = 0.0;
        B(i + dof, 1) = deriv_shapes(i, 1);
        B(i + dof, 2) = 0.0;
        B(i + dof, 3) = deriv_shapes(i, 2);
        B(i + dof, 4) = 0.0;
        B(i + dof, 5) = deriv_shapes(i, 0);
    }

    // z dofs
    for (int i = 0; i < dof; i++) {
        B(i + 2 * dof, 0) = 0.0;
        B(i + 2 * dof, 1) = 0.0;
        B(i + 2 * dof, 2) = deriv_shapes(i, 2);
        B(i + 2 * dof, 3) = deriv_shapes(i, 1);
        B(i + 2 * dof, 4) = deriv_shapes(i, 0);
        B(i + 2 * dof, 5) = 0.0;
    }
}

/**
 * @brief Construct B-bar matrix for selective reduced integration and volumetric locking
 * mitigation.
 *
 * @param deriv_shapes Dense matrix containing shape function derivatives in physical coordinates
 * (∂N/∂x)
 * @param elem_deriv_shapes Dense matrix containing element-averaged shape function derivatives
 * (∂N̄/∂x)
 * @param B Output B-bar matrix relating nodal displacements to strain components (modified in
 * place)
 *
 * This function constructs the B-bar matrix using the classical Hughes formulation for
 * treating nearly incompressible materials. The B-bar method applies selective reduced
 * integration by splitting the strain into volumetric and deviatoric components, then
 * using element-averaged shape function derivatives for the volumetric part while
 * retaining full integration for the deviatoric part.
 *
 * The B-bar matrix is constructed using the decomposition:
 * B̄ = B_dev + B_vol
 *
 * where:
 * - B_dev represents the deviatoric strain contribution (full integration)
 * - B_vol represents the volumetric strain contribution (reduced integration via averaging)
 *
 * The volumetric modification is applied to the normal strain components through:
 * B̄_ii = B_ii + (∂N̄/∂x_i - ∂N/∂x_i)/3
 *
 * where the factor of 1/3 distributes the volumetric correction equally across the
 * three normal strain components, ensuring proper treatment of the volumetric constraint
 * for nearly incompressible materials.
 *
 * Matrix structure for 3D elements:
 * - Input deriv_shapes: (dof × 3) matrix of shape function derivatives at integration point
 * - Input elem_deriv_shapes: (dof × 3) matrix of element-averaged shape function derivatives
 * - Output B: (3*dof × 6) matrix in Voigt notation order
 * - Strain ordering: [ε_xx, ε_yy, ε_zz, γ_xy, γ_xz, γ_yz]
 *
 * The B-bar matrix structure for each node i follows the pattern:
 * ```
 * [B̄₁ + ∂N_i/∂x    B̄₁           B̄₁         0         ∂N_i/∂z   ∂N_i/∂y]  <- x-displacement DOF
 * [    B̄₂       B̄₂ + ∂N_i/∂y    B̄₂      ∂N_i/∂z      0        ∂N_i/∂x]  <- y-displacement DOF
 * [    B̄₃           B̄₃       B̄₃ + ∂N_i/∂z ∂N_i/∂y   ∂N_i/∂x      0    ]  <- z-displacement DOF
 * ```
 *
 * where B̄ₖ = (∂N̄_i/∂x_k - ∂N_i/∂x_k)/3 for k = 1,2,3
 *
 * Note that the shear strain components (columns 4-6) use the standard B-matrix
 * formulation without volumetric correction, as they do not contribute to volumetric
 * deformation.
 *
 * This formulation effectively prevents volumetric locking in low-order elements
 * (such as linear hexahedra and tetrahedra) when analyzing nearly incompressible
 * materials with Poisson's ratios approaching 0.5.
 *
 * @note This function assumes 3D elements and unrolls loops for performance.
 * @note The elem_deriv_shapes matrix should contain element-averaged derivatives: ∂N̄/∂x = (1/V)∫_V
 * ∂N/∂x dV.
 * @note The B matrix must be pre-sized to (3*dof, 6) before calling this function.
 * @note For compressible materials, this reduces to the standard B-matrix as elem_deriv_shapes →
 * deriv_shapes.
 * @note Follows MFEM's vector ordering: [x₀...xₙ, y₀...yₙ, z₀...zₙ].
 *
 * @see T.J.R. Hughes, "The Finite Element Method: Linear Static and Dynamic Finite Element
 * Analysis"
 *
 * @ingroup ExaConstit_utilities_assembly
 */
inline void GenerateGradBarMatrix(const mfem::DenseMatrix& deriv_shapes,
                                  const mfem::DenseMatrix& elem_deriv_shapes,
                                  mfem::DenseMatrix& B) {
    int dof = deriv_shapes.Height();

    for (int i = 0; i < dof; i++) {
        const double B1 = (elem_deriv_shapes(i, 0) - deriv_shapes(i, 0)) / 3.0;
        B(i, 0) = B1 + deriv_shapes(i, 0);
        B(i, 1) = B1;
        B(i, 2) = B1;
        B(i, 3) = 0.0;
        B(i, 4) = deriv_shapes(i, 2);
        B(i, 5) = deriv_shapes(i, 1);
    }

    // y dofs
    for (int i = 0; i < dof; i++) {
        const double B2 = (elem_deriv_shapes(i, 1) - deriv_shapes(i, 1)) / 3.0;
        B(i + dof, 0) = B2;
        B(i + dof, 1) = B2 + deriv_shapes(i, 1);
        B(i + dof, 2) = B2;
        B(i + dof, 3) = deriv_shapes(i, 2);
        B(i + dof, 4) = 0.0;
        B(i + dof, 5) = deriv_shapes(i, 0);
    }

    // z dofs
    for (int i = 0; i < dof; i++) {
        const double B3 = (elem_deriv_shapes(i, 2) - deriv_shapes(i, 2)) / 3.0;
        B(i + 2 * dof, 0) = B3;
        B(i + 2 * dof, 1) = B3;
        B(i + 2 * dof, 2) = B3 + deriv_shapes(i, 2);
        B(i + 2 * dof, 3) = deriv_shapes(i, 1);
        B(i + 2 * dof, 4) = deriv_shapes(i, 0);
        B(i + 2 * dof, 5) = 0.0;
    }
}

/**
 * @brief Construct geometric B-matrix for geometric stiffness operations.
 *
 * @param deriv_shapes Dense matrix containing shape function derivatives in physical coordinates
 * (∂N/∂x)
 * @param B_geom Output geometric B-matrix for nonlinear geometric stiffness computations
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
 * - Input deriv_shapes: (dof × 3) matrix of shape function derivatives
 * - Output B_geom: (3*dof × 9) matrix organized in spatial dimension blocks
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
 * @note The deriv_shapes matrix should contain shape function derivatives in physical coordinates.
 * @note The B_geom matrix must be pre-sized to (3*dof, 9) before calling this function.
 * @note The function follows MFEM's vector ordering: [x₀...xₙ, y₀...yₙ, z₀...zₙ].
 *
 * @ingroup ExaConstit_utilities_assembly
 */
inline void GenerateGradGeomMatrix(const mfem::DenseMatrix& deriv_shapes,
                                   mfem::DenseMatrix& B_geom) {
    int dof = deriv_shapes.Height();
    // For a 3D mesh B_geom has the following shape:
    // [deriv_shapes(i, 0), 0, 0]
    // [deriv_shapes(i, 0), 0, 0]
    // [deriv_shapes(i, 0), 0, 0]
    // [0, deriv_shapes(i, 1), 0]
    // [0, deriv_shapes(i, 1), 0]
    // [0, deriv_shapes(i, 1), 0]
    // [0, 0, deriv_shapes(i, 2)]
    // [0, 0, deriv_shapes(i, 2)]
    // [0, 0, deriv_shapes(i, 2)]
    // We'll be returning the transpose of this.
    // It turns out the Bilinear operator can't have this created using
    // the dense gradient matrix, deriv_shapes.
    // It can be used in the following: B_geom^T Sigma_bar B_geom
    // where Sigma_bar is a block diagonal version of sigma repeated 3 times in 3D.

    // I'm assumming we're in 3D and have just unrolled the loop
    // The ordering has now changed such that B_geom matches up with mfem's internal
    // ordering of vectors such that it's [x0...xn, y0...yn, z0...zn] ordering

    // The previous single loop has been split into 3 so the B matrix
    // is constructed in chunks now instead of performing multiple striding
    // operations in a single loop.

    // x dofs
    for (int i = 0; i < dof; i++) {
        B_geom(i, 0) = deriv_shapes(i, 0);
        B_geom(i, 1) = deriv_shapes(i, 1);
        B_geom(i, 2) = deriv_shapes(i, 2);
        B_geom(i, 3) = 0.0;
        B_geom(i, 4) = 0.0;
        B_geom(i, 5) = 0.0;
        B_geom(i, 6) = 0.0;
        B_geom(i, 7) = 0.0;
        B_geom(i, 8) = 0.0;
    }

    // y dofs
    for (int i = 0; i < dof; i++) {
        B_geom(i + dof, 0) = 0.0;
        B_geom(i + dof, 1) = 0.0;
        B_geom(i + dof, 2) = 0.0;
        B_geom(i + dof, 3) = deriv_shapes(i, 0);
        B_geom(i + dof, 4) = deriv_shapes(i, 1);
        B_geom(i + dof, 5) = deriv_shapes(i, 2);
        B_geom(i + dof, 6) = 0.0;
        B_geom(i + dof, 7) = 0.0;
        B_geom(i + dof, 8) = 0.0;
    }

    // z dofs
    for (int i = 0; i < dof; i++) {
        B_geom(i + 2 * dof, 0) = 0.0;
        B_geom(i + 2 * dof, 1) = 0.0;
        B_geom(i + 2 * dof, 2) = 0.0;
        B_geom(i + 2 * dof, 3) = 0.0;
        B_geom(i + 2 * dof, 4) = 0.0;
        B_geom(i + 2 * dof, 5) = 0.0;
        B_geom(i + 2 * dof, 6) = deriv_shapes(i, 0);
        B_geom(i + 2 * dof, 7) = deriv_shapes(i, 1);
        B_geom(i + 2 * dof, 8) = deriv_shapes(i, 2);
    }
}

/**
 * @brief Get quadrature function data at a specific element and integration point.
 *
 * @param elem_id Global element index
 * @param int_point_num Integration point number within the element
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
inline void GetQFData(const int elem_id,
                      const int int_point_num,
                      double* qfdata,
                      std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf) {
    const auto data = qf->HostRead();
    const int qf_offset = qf->GetVDim();
    auto qspace = qf->GetSpaceShared();

    const mfem::IntegrationRule* ir = &(qf->GetSpaceShared()->GetIntRule(elem_id));
    int elem_offset = qf_offset * ir->GetNPoints();

    for (int i = 0; i < qf_offset; ++i) {
        qfdata[i] = data[elem_id * elem_offset + int_point_num * qf_offset + i];
    }
}

/**
 * @brief Set quadrature function data at a specific element and integration point.
 *
 * @param elem_id Global element index
 * @param int_point_num Integration point number within the element
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
inline void SetQFData(const int elem_id,
                      const int int_point_num,
                      double* qfdata,
                      std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf) {
    auto data = qf->HostReadWrite();
    const int qf_offset = qf->GetVDim();
    auto qspace = qf->GetSpaceShared();

    const mfem::IntegrationRule* ir = &(qf->GetSpaceShared()->GetIntRule(elem_id));
    int elem_offset = qf_offset * ir->GetNPoints();

    for (int i = 0; i < qf_offset; ++i) {
        data[elem_id * elem_offset + int_point_num * qf_offset + i] = qfdata[i];
    }
}

/**
 * @brief Transform material gradient to 4D layout for partial assembly.
 *
 * @param mat_grad Shared pointer to material gradient PartialQuadratureFunction
 * @param mat_grad_PA Output vector with 4D layout for partial assembly
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
 * @note The mat_grad_PA vector is resized automatically to accommodate the data.
 * @note The function assumes 3D problems with 6x6 material tangent matrices.
 * @note RAJA views use specific permutations for optimal performance.
 *
 * @ingroup ExaConstit_utilities_assembly
 */
inline void
TransformMatGradTo4D(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> mat_grad,
                     mfem::Vector& mat_grad_PA) {
    const int npts = mat_grad->Size() / mat_grad->GetVDim();

    const int dim = 3;
    const int dim2 = 6;

    const int DIM5 = 5;
    const int DIM3 = 3;
    std::array<RAJA::idx_t, DIM5> perm5{{4, 3, 2, 1, 0}};
    std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0}};

    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
    RAJA::Layout<DIM5> layout_4Dtensor = RAJA::make_permuted_layout({{dim, dim, dim, dim, npts}},
                                                                    perm5);
    RAJA::View<double, RAJA::Layout<DIM5, RAJA::Index_type, 0>> cmat_4d(mat_grad_PA.ReadWrite(),
                                                                        layout_4Dtensor);

    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
    RAJA::Layout<DIM3> layout_2Dtensor = RAJA::make_permuted_layout({{dim2, dim2, npts}}, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0>> cmat(mat_grad->Read(),
                                                                           layout_2Dtensor);

    // This sets up our 4D tensor to be the same as the 2D tensor which takes advantage of symmetry
    // operations
    mfem::forall(npts, [=] MFEM_HOST_DEVICE(int i) {
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
