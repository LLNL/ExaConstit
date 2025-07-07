#pragma once

#include "mfem.hpp"
#include "ECMech_gpu_portability.h"

#include <cmath>
#include <limits>

// A helper function that takes in a 3x3 rotation matrix and converts it over
// to a unit quaternion.
// rmat should be constant here...
inline
void
RMat2Quat(const mfem::DenseMatrix& rmat, mfem::Vector& quat)
{
   constexpr double inv2 = 0.5;
   double phi = 0.0;
   static const double eps = std::numeric_limits<double>::epsilon();
   double tr_r = 0.0;
   double inv_sin = 0.0;
   double s = 0.0;


   quat = 0.0;

   tr_r = rmat(0, 0) + rmat(1, 1) + rmat(2, 2);
   phi = inv2 * (tr_r - 1.0);
   phi = std::min(phi, 1.0);
   phi = std::max(phi, -1.0);
   phi = std::acos(phi);
   if (std::abs(phi) < eps) {
      quat[3] = 1.0;
   }
   else {
      inv_sin = 1.0 / sin(phi);
      quat[0] = phi;
      quat[1] = inv_sin * inv2 * (rmat(2, 1) - rmat(1, 2));
      quat[2] = inv_sin * inv2 * (rmat(0, 2) - rmat(2, 0));
      quat[3] = inv_sin * inv2 * (rmat(1, 0) - rmat(0, 1));
   }

   s = std::sin(inv2 * quat[0]);
   quat[0] = std::cos(quat[0] * inv2);
   quat[1] = s * quat[1];
   quat[2] = s * quat[2];
   quat[3] = s * quat[3];

}

// A helper function that takes in a unit quaternion and and returns a 3x3 rotation
// matrix.
inline
void
Quat2RMat(const mfem::Vector& quat, mfem::DenseMatrix& rmat)
{
   double qbar = 0.0;

   qbar = quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

   rmat(0, 0) = qbar + 2.0 * quat[1] * quat[1];
   rmat(1, 0) = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
   rmat(2, 0) = 2.0 * (quat[1] * quat[3] - quat[0] * quat[2]);

   rmat(0, 1) = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
   rmat(1, 1) = qbar + 2.0 * quat[2] * quat[2];
   rmat(2, 1) = 2.0 * (quat[2] * quat[3] + quat[0] * quat[1]);

   rmat(0, 2) = 2.0 * (quat[1] * quat[3] + quat[0] * quat[2]);
   rmat(1, 2) = 2.0 * (quat[2] * quat[3] - quat[0] * quat[1]);
   rmat(2, 2) = qbar + 2.0 * quat[3] * quat[3];
}

__ecmech_hdev__
inline
void 
quat2rmat(const double* const quat,
          double* const rmats) 
{
    const double qbar =  quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

    double* rmat[3] = {&rmats[0], &rmats[3], &rmats[6]};

    rmat[0][0] = qbar + 2.0 * quat[1] * quat[1];
    rmat[1][0] = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
    rmat[2][0] = 2.0 * (quat[1] * quat[3] - quat[0] * quat[2]);

    rmat[0][1] = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
    rmat[1][1] = qbar + 2.0 * quat[2] * quat[2];
    rmat[2][1] = 2.0 * (quat[2] * quat[3] + quat[0] * quat[1]);

    rmat[0][2] = 2.0 * (quat[1] * quat[3] + quat[0] * quat[2]);
    rmat[1][2] = 2.0 * (quat[2] * quat[3] - quat[0] * quat[1]);
    rmat[2][2] = qbar + 2.0 * quat[3] * quat[3];
}