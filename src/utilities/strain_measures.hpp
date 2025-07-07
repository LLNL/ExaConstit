#pragma once

#include "utilities/rotations.hpp"
#include "mfem.hpp"

#include <cmath>

// The below method computes the polar decomposition of a 3x3 matrix using a method
// proposed in: https://animation.rwth-aachen.de/media/papers/2016-MIG-StableRotation.pdf
// The paper listed provides a fast and robust way to obtain the rotation portion
// of a positive definite 3x3 matrix which then allows for the easy computation
// of U and V.
inline
void 
CalcPolarDecompDefGrad(mfem::DenseMatrix& R, mfem::DenseMatrix& U,
                       mfem::DenseMatrix& V, double err = 1e-12)
{
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
 
    ac1[0] = def_grad(0, 0); ac1[1] = def_grad(1, 0); ac1[2] = def_grad(2, 0);
    ac2[0] = def_grad(0, 1); ac2[1] = def_grad(1, 1); ac2[2] = def_grad(2, 1);
    ac3[0] = def_grad(0, 2); ac3[1] = def_grad(1, 2); ac3[2] = def_grad(2, 2);
 
    for (int i = 0; i < max_iter; i++) {
       // The dot products that show up in the paper
       r1da1 = R(0, 0) * ac1[0] + R(1, 0) * ac1[1] + R(2, 0) * ac1[2];
       r2da2 = R(0, 1) * ac2[0] + R(1, 1) * ac2[1] + R(2, 1) * ac2[2];
       r3da3 = R(0, 2) * ac3[0] + R(1, 2) * ac3[1] + R(2, 2) * ac3[2];
 
       // The summed cross products that show up in the paper
       w_top[0] = (-R(2, 0) * ac1[1] + R(1, 0) * ac1[2]) +
                  (-R(2, 1) * ac2[1] + R(1, 1) * ac2[2]) +
                  (-R(2, 2) * ac3[1] + R(1, 2) * ac3[2]);
 
       w_top[1] = (R(2, 0) * ac1[0] - R(0, 0) * ac1[2]) +
                  (R(2, 1) * ac2[0] - R(0, 1) * ac2[2]) +
                  (R(2, 2) * ac3[0] - R(0, 2) * ac3[2]);
 
       w_top[2] = (-R(1, 0) * ac1[0] + R(0, 0) * ac1[1]) +
                  (-R(1, 1) * ac2[0] + R(0, 1) * ac2[1]) +
                  (-R(1, 2) * ac3[0] + R(0, 2) * ac3[1]);
 
       w_bot = (1.0 / (std::abs(r1da1 + r2da2 + r3da3) + err));
       // The axial vector that shows up in the paper
       w[0] = w_top[0] * w_bot; w[1] = w_top[1] * w_bot; w[2] = w_top[2] * w_bot;
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

// This method calculates the Lagrangian strain which is given as:
// E = 1/2 (C - I) = 1/2 (F^(T)F - I)
inline
void
CalcLagrangianStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F)
{
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

// This method calculates the Eulerian strain which is given as:
// e = 1/2 (I - B^(-1)) = 1/2 (I - F(^-T)F^(-1))
inline
void
CalcEulerianStrain(mfem::DenseMatrix& e, const mfem::DenseMatrix &F)
{
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

// This method calculates the Biot strain which is given as:
// E = (U - I) or sometimes seen as E = (V - I) if R = I
inline
void
CalcBiotStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F)
{
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

inline
void
CalcLogStrain(mfem::DenseMatrix& E, const mfem::DenseMatrix &F)
{
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
   for (int i = 0; i<dim; ++i) { // outer loop for every eigenvalue/vector
      for (int j = 0; j<dim; ++j) { // inner loops for diadic product of eigenvectors
         for (int k = 0; k<dim; ++k) {
            // Dense matrices are col. maj. representation, so the indices were
            // reversed for it to be more cache friendly.
            E(k, j) += 0.5 * log(lambda[i]) * vec[i * dim + j] * vec[i * dim + k];
         }
      }
   }
}