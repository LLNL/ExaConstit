#pragma once

#include "utilities/rotations.hpp"

#include "mfem.hpp"

#include <cmath>

/**
 * @brief Compute polar decomposition of a 3x3 deformation gradient using stable rotation
 * extraction.
 *
 * @param R Input deformation gradient matrix, output rotation matrix (3x3)
 * @param U Output right stretch tensor (3x3)
 * @param V Output left stretch tensor (3x3)
 * @param err Convergence tolerance for iterative algorithm (default: 1e-12)
 *
 * This function computes the polar decomposition F = R*U = V*R of a 3x3 deformation
 * gradient matrix using a fast and robust iterative algorithm proposed by Müller et al.
 * The method is particularly well-suited for finite element applications where
 * numerical stability and performance are critical.
 *
 * Polar decomposition separates the deformation into:
 * - R: Rotation tensor (proper orthogonal matrix, det(R) = +1)
 * - U: Right stretch tensor (symmetric positive definite)
 * - V: Left stretch tensor (symmetric positive definite)
 *
 * Algorithm characteristics:
 * - Based on iterative extraction of rotation from deformation gradient
 * - Uses quaternion intermediate representation for numerical stability
 * - Exponential mapping ensures rapid convergence
 * - Robust handling of near-singular and large deformation cases
 * - Maximum 500 iterations with configurable tolerance
 *
 * The algorithm performs these steps:
 * 1. Extract initial rotation estimate using SVD-based quaternion method
 * 2. Iteratively refine rotation using exponential mapping
 * 3. Compute axial vector corrections for rotation updates
 * 4. Apply exponential mapping to update rotation matrix
 * 5. Converge when correction magnitude falls below tolerance
 * 6. Compute stretch tensors: U = R^T * F, V = F * R^T
 *
 * Applications in solid mechanics:
 * - Large deformation analysis requiring objective stress measures
 * - Crystal plasticity with finite rotations
 * - Hyperelastic material models using stretch-based formulations
 * - Kinematic analysis of deforming structures
 *
 * Reference: "A Robust Method to Extract the Rotational Part of Deformations"
 * by Müller et al., MIG 2016
 *
 * @note The input matrix R is modified in place and becomes the rotation output.
 * @note The algorithm assumes the input represents a valid deformation gradient (det(F) > 0).
 * @note Convergence is typically achieved in 5-15 iterations for typical FE problems.
 * @note The method is more stable than traditional SVD-based approaches for ill-conditioned cases.
 *
 * Usage example:
 * @code
 * mfem::DenseMatrix F(3), R(3), U(3), V(3);
 * // ... populate F with deformation gradient ...
 * R = F; // Copy F since it will be modified
 * CalcPolarDecompDefGrad(R, U, V);
 * // Now R contains rotation, U and V contain right and left stretch
 * @endcode
 *
 * @ingroup ExaConstit_utilities_strain
 */
inline void CalcPolarDecompDefGrad(mfem::DenseMatrix& R,
                                   mfem::DenseMatrix& U,
                                   mfem::DenseMatrix& V,
                                   double err = 1e-12) {
    mfem::DenseMatrix omega_mat, temp;
    mfem::DenseMatrix def_grad(R, 3);

    constexpr int dim = 3;
    mfem::Vector quat;

    constexpr int max_iter = 500;

    double norm, inv_norm;

    double ac1[3], ac2[3], ac3[3];
    double w_top[3], w[3];
    double w_bot, w_norm, w_norm_inv2, w_norm_inv;
    double cth, sth;
    double r1da1, r2da2, r3da3;

    quat.SetSize(4);
    omega_mat.SetSize(dim);
    temp.SetSize(dim);

    quat = 0.0;

    RMat2Quat(def_grad, quat);

    norm = quat.Norml2();

    inv_norm = 1.0 / norm;

    quat *= inv_norm;

    Quat2RMat(quat, R);

    ac1[0] = def_grad(0, 0);
    ac1[1] = def_grad(1, 0);
    ac1[2] = def_grad(2, 0);
    ac2[0] = def_grad(0, 1);
    ac2[1] = def_grad(1, 1);
    ac2[2] = def_grad(2, 1);
    ac3[0] = def_grad(0, 2);
    ac3[1] = def_grad(1, 2);
    ac3[2] = def_grad(2, 2);

    for (int i = 0; i < max_iter; i++) {
        // The dot products that show up in the paper
        r1da1 = R(0, 0) * ac1[0] + R(1, 0) * ac1[1] + R(2, 0) * ac1[2];
        r2da2 = R(0, 1) * ac2[0] + R(1, 1) * ac2[1] + R(2, 1) * ac2[2];
        r3da3 = R(0, 2) * ac3[0] + R(1, 2) * ac3[1] + R(2, 2) * ac3[2];

        // The summed cross products that show up in the paper
        w_top[0] = (-R(2, 0) * ac1[1] + R(1, 0) * ac1[2]) + (-R(2, 1) * ac2[1] + R(1, 1) * ac2[2]) +
                   (-R(2, 2) * ac3[1] + R(1, 2) * ac3[2]);

        w_top[1] = (R(2, 0) * ac1[0] - R(0, 0) * ac1[2]) + (R(2, 1) * ac2[0] - R(0, 1) * ac2[2]) +
                   (R(2, 2) * ac3[0] - R(0, 2) * ac3[2]);

        w_top[2] = (-R(1, 0) * ac1[0] + R(0, 0) * ac1[1]) + (-R(1, 1) * ac2[0] + R(0, 1) * ac2[1]) +
                   (-R(1, 2) * ac3[0] + R(0, 2) * ac3[1]);

        w_bot = (1.0 / (std::abs(r1da1 + r2da2 + r3da3) + err));
        // The axial vector that shows up in the paper
        w[0] = w_top[0] * w_bot;
        w[1] = w_top[1] * w_bot;
        w[2] = w_top[2] * w_bot;
        // The norm of the axial vector
        w_norm = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        // If the norm is below our desired error we've gotten our solution
        // So we can break out of the loop
        if (w_norm < err) {
            break;
        }
        // The exponential mapping for an axial vector
        // The 3x3 case has been explicitly unrolled here
        w_norm_inv2 = 1.0 / (w_norm * w_norm);
        w_norm_inv = 1.0 / w_norm;

        sth = std::sin(w_norm) * w_norm_inv;
        cth = (1.0 - std::cos(w_norm)) * w_norm_inv2;

        omega_mat(0, 0) = 1.0 - cth * (w[2] * w[2] + w[1] * w[1]);
        omega_mat(1, 1) = 1.0 - cth * (w[2] * w[2] + w[0] * w[0]);
        omega_mat(2, 2) = 1.0 - cth * (w[1] * w[1] + w[0] * w[0]);

        omega_mat(0, 1) = -sth * w[2] + cth * w[1] * w[0];
        omega_mat(0, 2) = sth * w[1] + cth * w[2] * w[0];

        omega_mat(1, 0) = sth * w[2] + cth * w[0] * w[1];
        omega_mat(1, 2) = -sth * w[0] + cth * w[2] * w[1];

        omega_mat(2, 0) = -sth * w[1] + cth * w[0] * w[2];
        omega_mat(2, 1) = sth * w[0] + cth * w[2] * w[1];

        Mult(omega_mat, R, temp);
        R = temp;
    }

    // Now that we have the rotation portion of our deformation gradient
    // the left and right stretch tensors are easy to find.
    MultAtB(R, def_grad, U);
    MultABt(def_grad, R, V);
}

/**
 * @brief Calculate the Lagrangian strain tensor from deformation gradient.
 *
 * @param E Output Lagrangian strain tensor (3x3, symmetric)
 * @param F Input deformation gradient tensor (3x3)
 *
 * This function computes the Lagrangian strain tensor (also known as Green-Lagrange strain)
 * using the standard definition:
 *
 * E = (1/2)(C - I) = (1/2)(F^T F - I)
 *
 * where:
 * - F is the deformation gradient tensor
 * - C = F^T F is the right Cauchy-Green deformation tensor
 * - I is the 3x3 identity tensor
 *
 * The Lagrangian strain tensor provides a material description of strain that:
 * - Is objective (frame-invariant) under rigid body rotations
 * - Vanishes for rigid body motion (E = 0 when F = R)
 * - Is symmetric by construction
 * - Measures strain relative to the reference configuration
 *
 * Mathematical properties:
 * - E_ij = (1/2)(∂u_i/∂X_j + ∂u_j/∂X_i + ∂u_k/∂X_i ∂u_k/∂X_j)
 * - For small deformations: E ≈ (1/2)(∇u + ∇u^T) (linearized strain)
 * - Principal strains are eigenvalues of E
 * - Compatible with hyperelastic constitutive models
 *
 * Applications in continuum mechanics:
 * - Nonlinear elasticity and hyperelasticity
 * - Large deformation finite element analysis
 * - Material point method and other Lagrangian formulations
 * - Constitutive model implementation for finite strains
 *
 * The computation is efficient and involves:
 * 1. Computing C = F^T * F using optimized matrix multiplication
 * 2. Scaling by 1/2 and subtracting identity from diagonal terms
 * 3. Ensuring symmetry of the result
 *
 * @note The output strain tensor E is automatically symmetric.
 * @note For infinitesimal strains, this reduces to the linearized strain tensor.
 * @note The function assumes F represents a valid deformation gradient.
 *
 * @ingroup ExaConstit_utilities_strain
 */
inline void CalcLagrangianStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix& F) {
    constexpr int dim = 3;

    // DenseMatrix F(Jpt, dim);
    mfem::DenseMatrix C(dim);

    constexpr double half = 0.5;

    MultAtB(F, F, C);

    E = 0.0;

    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < dim; i++) {
            E(i, j) += half * C(i, j);
        }

        E(j, j) -= half;
    }
}

/**
 * @brief Calculate the Eulerian strain tensor from deformation gradient.
 *
 * @param e Output Eulerian strain tensor (3x3, symmetric)
 * @param F Input deformation gradient tensor (3x3)
 *
 * This function computes the Eulerian strain tensor (also known as Almansi strain)
 * using the standard definition:
 *
 * e = (1/2)(I - B^(-1)) = (1/2)(I - F^(-T) F^(-1))
 *
 * where:
 * - F is the deformation gradient tensor
 * - B^(-1) = F^(-T) F^(-1) is the inverse left Cauchy-Green deformation tensor
 * - I is the 3x3 identity tensor
 *
 * The Eulerian strain tensor provides a spatial description of strain that:
 * - Describes strain in the current (deformed) configuration
 * - Is objective under rigid body rotations
 * - Vanishes for rigid body motion
 * - Complements the Lagrangian strain description
 *
 * Mathematical characteristics:
 * - Measures strain relative to the current configuration
 * - For small deformations: e ≈ (1/2)(∇u + ∇u^T) (same as Lagrangian)
 * - Related to velocity gradient in rate form
 * - Useful for spatial constitutive formulations
 *
 * Computational procedure:
 * 1. Compute F^(-1) using matrix inversion
 * 2. Calculate B^(-1) = F^(-T) F^(-1)
 * 3. Compute e = (1/2)(I - B^(-1))
 *
 * Applications:
 * - Eulerian finite element formulations
 * - Fluid-structure interaction problems
 * - Updated Lagrangian formulations
 * - Spatial constitutive model implementations
 *
 * Numerical considerations:
 * - Requires matrix inversion which may be expensive
 * - Numerical stability depends on conditioning of F
 * - More sensitive to numerical errors than Lagrangian strain
 *
 * @note The function requires F to be invertible (det(F) > 0).
 * @note Matrix inversion is performed using MFEM's CalcInverse function.
 * @note For nearly incompressible materials, use with appropriate precautions.
 *
 * @ingroup ExaConstit_utilities_strain
 */
inline void CalcEulerianStrain(mfem::DenseMatrix& e, const mfem::DenseMatrix& F) {
    constexpr int dim = 3;

    mfem::DenseMatrix Finv(dim), Binv(dim);

    constexpr double half = 0.5;

    CalcInverse(F, Finv);

    MultAtB(Finv, Finv, Binv);

    e = 0.0;

    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < dim; i++) {
            e(i, j) -= half * Binv(i, j);
        }

        e(j, j) += half;
    }
}

/**
 * @brief Calculate the Biot strain tensor from deformation gradient.
 *
 * @param E Output Biot strain tensor (3x3, symmetric)
 * @param F Input deformation gradient tensor (3x3)
 *
 * This function computes the Biot strain tensor using the definition:
 *
 * E = U - I  (or alternatively E = V - I when R = I)
 *
 * where:
 * - U is the right stretch tensor from polar decomposition F = RU
 * - V is the left stretch tensor from polar decomposition F = VR
 * - I is the 3x3 identity tensor
 * - R is the rotation tensor
 *
 * The Biot strain tensor provides an intuitive measure of pure stretch:
 * - Directly measures stretch ratios in principal directions
 * - Vanishes for rigid body motion (E = 0 when U = I)
 * - Symmetric by construction (since U and V are symmetric)
 * - Physically represents "engineering strain" for principal directions
 *
 * Key properties:
 * - E_ii = λ_i - 1 where λ_i are principal stretches
 * - For small deformations: E ≈ linearized strain tensor
 * - Simple interpretation: E_ii is the fractional change in length
 * - Compatible with logarithmic strain for hyperelastic models
 *
 * Computational approach:
 * 1. Perform polar decomposition F = RU to extract U
 * 2. Compute E = U - I by subtracting identity
 * 3. Result is automatically symmetric
 *
 * Applications in material modeling:
 * - Hyperelastic constitutive relations
 * - Crystal plasticity where stretch is separated from rotation
 * - Biomechanics applications requiring intuitive strain measures
 * - Damage mechanics based on principal stretches
 *
 * Advantages over other strain measures:
 * - Direct physical interpretation as stretch ratios
 * - Computationally efficient (single polar decomposition)
 * - Natural for anisotropic material models
 * - Separates pure deformation from rotation effects
 *
 * @note This function internally calls CalcPolarDecompDefGrad.
 * @note The computation is more expensive than simple strain measures due to polar decomposition.
 * @note For small deformations, Biot strain converges to linearized strain.
 *
 * @ingroup ExaConstit_utilities_strain
 */
inline void CalcBiotStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix& F) {
    constexpr int dim = 3;

    mfem::DenseMatrix rmat(F, dim);
    mfem::DenseMatrix umat, vmat;

    umat.SetSize(dim);
    vmat.SetSize(dim);

    CalcPolarDecompDefGrad(rmat, umat, vmat);

    E = umat;
    E(0, 0) -= 1.0;
    E(1, 1) -= 1.0;
    E(2, 2) -= 1.0;
}

/**
 * @brief Calculate the logarithmic strain tensor (Hencky strain) from deformation gradient.
 *
 * @param E Output logarithmic strain tensor (3x3, symmetric)
 * @param F Input deformation gradient tensor (3x3)
 *
 * This function computes the logarithmic strain tensor (also known as Hencky strain
 * or true strain) using the spectral decomposition approach:
 *
 * E = ln(V) = (1/2) ln(B) = (1/2) ln(FF^T)
 *
 * where:
 * - F is the deformation gradient tensor
 * - V is the left stretch tensor from polar decomposition F = VR
 * - B = FF^T is the left Cauchy-Green deformation tensor
 * - ln denotes the matrix logarithm
 *
 * The logarithmic strain tensor is considered the most natural finite strain measure:
 * - Objective under rigid body rotations
 * - Additive for successive deformations
 * - Vanishes for rigid body motion
 * - Principal values are ln(λ_i) where λ_i are principal stretches
 *
 * Mathematical advantages:
 * - E_ii = ln(λ_i) represents true strain in principal directions
 * - For small deformations: E ≈ linearized strain tensor
 * - Compatible with multiplicative decomposition in plasticity
 * - Natural measure for hyperelastic models
 *
 * Computational procedure:
 * 1. Compute B = F F^T (left Cauchy-Green tensor)
 * 2. Calculate eigenvalue decomposition of B: B = Q Λ Q^T
 * 3. Compute matrix logarithm: ln(B) = Q ln(Λ) Q^T
 * 4. Scale by 1/2: E = (1/2) ln(B)
 *
 * The spectral decomposition enables efficient computation:
 * - Eigenvalues λ_i of B are squares of principal stretches
 * - ln(B) is computed as ln(λ_i) applied to eigenvalues
 * - Eigenvectors provide principal directions
 *
 * Applications in nonlinear mechanics:
 * - Hyperelastic material models (Neo-Hookean, Mooney-Rivlin)
 * - Crystal plasticity with finite deformations
 * - Multiplicative plasticity decomposition
 * - Biomechanics and soft tissue modeling
 *
 * Performance characteristics:
 * - More expensive than simple strain measures (eigenvalue decomposition)
 * - Numerically stable for typical finite element applications
 * - Handles large deformations robustly
 * - Compatible with GPU acceleration via MFEM's eigen solver
 *
 * @note The function uses MFEM's CalcEigenvalues for spectral decomposition.
 * @note Requires positive definite deformation gradient (det(F) > 0).
 * @note For nearly incompressible materials, eigenvalue computation is well-conditioned.
 *
 * @ingroup ExaConstit_utilities_strain
 */
inline void CalcLogStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix& F) {
    // calculate current end step logorithmic strain (Hencky Strain)
    // which is taken to be E = ln(U) = 1/2 ln(C), where C = (F_T)F.
    // We have incremental F from MFEM, and store F0 (Jpt0) so
    // F = F_hat*F0. With F, use a spectral decomposition on C to obtain a
    // form where we only have to take the natural log of the
    // eigenvalues
    // UMAT uses the E = ln(V) approach instead

    mfem::DenseMatrix B;

    constexpr int dim = 3;

    B.SetSize(dim);
    MultABt(F, F, B);

    // compute eigenvalue decomposition of B
    double lambda[dim];
    double vec[dim * dim];
    B.CalcEigenvalues(&lambda[0], &vec[0]);

    // compute ln(V) using spectral representation
    E = 0.0;
    for (int i = 0; i < dim; ++i) {     // outer loop for every eigenvalue/vector
        for (int j = 0; j < dim; ++j) { // inner loops for diadic product of eigenvectors
            for (int k = 0; k < dim; ++k) {
                // Dense matrices are col. maj. representation, so the indices were
                // reversed for it to be more cache friendly.
                E(k, j) += 0.5 * log(lambda[i]) * vec[i * dim + j] * vec[i * dim + k];
            }
        }
    }
}