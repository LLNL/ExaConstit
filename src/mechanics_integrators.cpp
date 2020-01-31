
#include "mfem.hpp"
#include "mechanics_integrators.hpp"
#include "BCManager.hpp"
#include <math.h> // log
#include <algorithm>
#include <iostream> // cerr
#include "RAJA/RAJA.hpp"

using namespace mfem;
using namespace std;

void computeDefGrad(const QuadratureFunction *qf, ParFiniteElementSpace *fes,
                    const Vector &x0)
{
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpace* qspace = qf->GetSpace();

   ParGridFunction x_gf;

   double* vals = x0.GetData();

   const int NE = fes->GetNE();

   x_gf.MakeTRef(fes, vals);
   x_gf.SetFromTrueVector();


   // loop over elements
   for (int i = 0; i < NE; ++i) {
      // get element transformation for the ith element
      ElementTransformation* Ttr = fes->GetElementTransformation(i);
      fe = fes->GetFE(i);

      // declare data to store shape function gradients
      // and element Jacobians
      DenseMatrix Jrt, DSh, DS, PMatI, Jpt, F0, F1;
      int dof = fe->GetDof(), dim = fe->GetDim();

      if (qf_offset != (dim * dim)) {
         mfem_error("computeDefGrd0 stride input arg not dim*dim");
      }

      DSh.SetSize(dof, dim);
      DS.SetSize(dof, dim);
      Jrt.SetSize(dim);
      Jpt.SetSize(dim);
      F0.SetSize(dim);
      F1.SetSize(dim);
      PMatI.SetSize(dof, dim);

      // get element physical coordinates
      // Array<int> vdofs;
      // Vector el_x;
      // fes->GetElementVDofs(i, vdofs);
      // x0.GetSubVector(vdofs, el_x);
      // PMatI.UseExternalData(el_x.GetData(), dof, dim);

      // get element physical coordinates
      Array<int> vdofs(dof * dim);
      Vector el_x(PMatI.Data(), dof * dim);
      fes->GetElementVDofs(i, vdofs);

      x_gf.GetSubVector(vdofs, el_x);

      ir = &(qspace->GetElementIntRule(i));
      int elem_offset = qf_offset * ir->GetNPoints();

      // loop over integration points where the quadrature function is
      // stored
      for (int j = 0; j < ir->GetNPoints(); ++j) {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Ttr->SetIntPoint(&ip);
         CalcInverse(Ttr->Jacobian(), Jrt);

         fe->CalcDShape(ip, DSh);
         Mult(DSh, Jrt, DS);
         MultAtB(PMatI, DS, Jpt);

         // store local beginning step deformation gradient for a given
         // element and integration point from the quadrature function
         // input argument. We want to set the new updated beginning
         // step deformation gradient (prior to next time step) to the current
         // end step deformation gradient associated with the converged
         // incremental solution. The converged _incremental_ def grad is Jpt
         // that we just computed above. We compute the updated beginning
         // step def grad as F1 = Jpt*F0; F0 = F1; We do this because we
         // are not storing F1.
         int k = 0;
         for (int n = 0; n < dim; ++n) {
            for (int m = 0; m < dim; ++m) {
               F0(m, n) = qf_data[i * elem_offset + j * qf_offset + k];
               ++k;
            }
         }

         // compute F1 = Jpt*F0;
         Mult(Jpt, F0, F1);

         // set new F0 = F1
         F0 = F1;

         // loop over element Jacobian data and populate
         // quadrature function with the new F0 in preparation for the next
         // time step. Note: offset0 should be the
         // number of true state variables.
         k = 0;
         for (int m = 0; m < dim; ++m) {
            for (int n = 0; n < dim; ++n) {
               qf_data[i * elem_offset + j * qf_offset + k] =
                  F0(n, m);
               ++k;
            }
         }
      }

      Ttr = NULL;
   }

   fe = NULL;
   ir = NULL;
   qf_data = NULL;
   qspace = NULL;

   return;
}

// member functions for the Abaqus Umat base class
int ExaModel::GetStressOffset()
{
   QuadratureFunction* qf = stress0.GetQuadFunction();
   int qf_offset = qf->GetVDim();

   qf = NULL;

   return qf_offset;
}

int ExaModel::GetMatGradOffset()
{
   QuadratureFunction* qf = matGrad.GetQuadFunction();
   int qf_offset = qf->GetVDim();

   qf = NULL;

   return qf_offset;
}

int ExaModel::GetMatVarsOffset()
{
   QuadratureFunction* qf = matVars0.GetQuadFunction();
   int qf_offset = qf->GetVDim();

   qf = NULL;

   return qf_offset;
}

// This method sets the end time step stress to the beginning step
// and then returns the internal data pointer of the end time step
// array.
double* ExaModel::StressSetup()
{
   const QuadratureFunction *stress_beg = stress0.GetQuadFunction();
   QuadratureFunction *stress_end = stress1.GetQuadFunction();

   *stress_end = *stress_beg;

   return stress_end->GetData();
}

// This methods set the end time step state variable array to the
// beginning time step values and then returns the internal data pointer
// of the end time step array.
double* ExaModel::StateVarsSetup()
{
   const QuadratureFunction *state_vars_beg = matVars0.GetQuadFunction();
   QuadratureFunction *state_vars_end = matVars1.GetQuadFunction();

   *state_vars_end = *state_vars_beg;

   return state_vars_end->GetData();
}

// the getter simply returns the beginning step stress
void ExaModel::GetElementStress(const int elID, const int ipNum,
                                bool beginStep, double* stress, int numComps)
{
   const IntegrationRule *ir = NULL;
   double* qf_data = NULL;
   int qf_offset = 0;
   QuadratureFunction* qf = NULL;
   QuadratureSpace* qspace = NULL;

   if (beginStep) {
      qf = stress0.GetQuadFunction();
   }
   else {
      qf = stress1.GetQuadFunction();
   }

   qf_data = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nGetElementStress: number of components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i = 0; i<numComps; ++i) {
      stress[i] = qf_data[elID * elem_offset + ipNum * qf_offset + i];
   }

   ir = NULL;
   qf_data = NULL;
   qf = NULL;
   qspace = NULL;

   return;
}

void ExaModel::SetElementStress(const int elID, const int ipNum,
                                bool beginStep, double* stress, int numComps)
{
   // printf("inside ExaModel::SetElementStress, elID, ipNum %d %d \n", elID, ipNum);
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   if (beginStep) {
      qf = stress0.GetQuadFunction();
   }
   else {
      qf = stress1.GetQuadFunction();
   }

   qf_data = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nSetElementStress: number of components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i = 0; i<qf_offset; ++i) {
      int k = elID * elem_offset + ipNum * qf_offset + i;
      qf_data[k] = stress[i];
   }

   return;
}

void ExaModel::GetElementStateVars(const int elID, const int ipNum,
                                   bool beginStep, double* stateVars,
                                   int numComps)
{
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   if (beginStep) {
      qf = matVars0.GetQuadFunction();
   }
   else {
      qf = matVars1.GetQuadFunction();
   }

   qf_data = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nGetElementStateVars: num. components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i = 0; i<numComps; ++i) {
      stateVars[i] = qf_data[elID * elem_offset + ipNum * qf_offset + i];
   }

   ir = NULL;
   qf_data = NULL;
   qf = NULL;
   qspace = NULL;

   return;
}

void ExaModel::SetElementStateVars(const int elID, const int ipNum,
                                   bool beginStep, double* stateVars,
                                   int numComps)
{
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   if (beginStep) {
      qf = matVars0.GetQuadFunction();
   }
   else {
      qf = matVars1.GetQuadFunction();
   }

   qf_data = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nSetElementStateVars: num. components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i = 0; i<qf_offset; ++i) {
      qf_data[elID * elem_offset + ipNum * qf_offset + i] = stateVars[i];
   }

   ir = NULL;
   qf_data = NULL;
   qf = NULL;
   qspace = NULL;

   return;
}

void ExaModel::GetElementMatGrad(const int elID, const int ipNum, double* grad,
                                 int numComps)
{
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   qf = matGrad.GetQuadFunction();

   qf_data = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nGetElementMatGrad: num. components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i = 0; i<numComps; ++i) {
      grad[i] = qf_data[elID * elem_offset + ipNum * qf_offset + i];
   }

   ir = NULL;
   qf_data = NULL;
   qf = NULL;
   qspace = NULL;

   return;
}

void ExaModel::SetElementMatGrad(const int elID, const int ipNum,
                                 double* grad, int numComps)
{
   const IntegrationRule *ir;
   double* qf_data;
   int qf_offset;
   QuadratureFunction* qf;
   QuadratureSpace* qspace;

   qf = matGrad.GetQuadFunction();

   qf_data = qf->GetData();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nSetElementMatGrad: num. components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetElementIntRule(elID));
   int elem_offset = qf_offset * ir->GetNPoints();

   for (int i = 0; i<qf_offset; ++i) {
      int k = elID * elem_offset + ipNum * qf_offset + i;
      qf_data[k] = grad[i];
   }

   ir = NULL;
   qf_data = NULL;
   qf = NULL;
   qspace = NULL;

   return;
}

void ExaModel::GetMatProps(double* props)
{
   double* mpdata = matProps->GetData();
   for (int i = 0; i < matProps->Size(); i++) {
      props[i] = mpdata[i];
   }

   return;
}

void ExaModel::SetMatProps(double* props, int size)
{
   matProps->NewDataAndSize(props, size);
   return;
}

void ExaModel::UpdateStress()
{
   QuadratureFunction* qf0;
   QuadratureFunction* qf1;

   qf0 = stress0.GetQuadFunction();
   qf1 = stress1.GetQuadFunction();
   qf0->Swap(*qf1);
}

void ExaModel::UpdateStateVars()
{
   QuadratureFunction* qf0;
   QuadratureFunction* qf1;

   qf0 = matVars0.GetQuadFunction();
   qf1 = matVars1.GetQuadFunction();
   qf0->Swap(*qf1);
}

void ExaModel::UpdateEndCoords(const Vector& vel)
{
   int size;

   size = vel.Size();

   Vector end_crds(size);

   end_crds = 0.0;

   // tdofs sounds like it should hold the data points of interest, since the GetTrueDofs()
   // points to the underlying data in the GridFunction if all the TDofs lie on a processor
   Vector bcrds;
   bcrds.SetSize(size);
   // beg_coords is the beginning time step coordinates
   beg_coords->GetTrueDofs(bcrds);
   int size2 = bcrds.Size();

   if (size != size2) {
      mfem_error("TrueDofs and Vel Solution vector sizes are different");
   }

   // Perform a simple time integration to get our new end time step coordinates
   for (int i = 0; i < size; ++i) {
      end_crds[i] = vel[i] * dt + bcrds[i];
   }

   // Now make sure the update gets sent to all the other processors that have ghost copies
   // of our data.
   end_coords->Distribute(end_crds);

   return;
}

void ExaModel::ComputeVonMises(const int elemID, const int ipID)
{
   QuadratureFunction *vm_qf = vonMises.GetQuadFunction();
   QuadratureSpace* vm_qspace = vm_qf->GetSpace();
   const IntegrationRule *ir;

   if (vm_qspace == NULL) {
      QuadratureFunction *qf_stress0 = stress0.GetQuadFunction();
      QuadratureSpace* qspace = qf_stress0->GetSpace();
      int vdim = 1; // scalar von Mises data at each IP
      vm_qf->SetSpace(qspace, vdim); // construct object

      qf_stress0 = NULL;
      qspace = NULL;
   }

   QuadratureSpace* qspace = vm_qf->GetSpace();
   double* vmData = vm_qf->GetData();
   int vmOffset = vm_qf->GetVDim();

   ir = &(qspace->GetElementIntRule(elemID));
   int elemVmOffset = vmOffset * ir->GetNPoints();

   double istress[6];
   GetElementStress(elemID, ipID, true, istress, 6);

   double term1 = istress[0] - istress[1];
   term1 *= term1;

   double term2 = istress[1] - istress[2];
   term2 *= term2;

   double term3 = istress[2] - istress[0];
   term3 *= term3;

   double term4 = istress[3] * istress[3] + istress[4] * istress[4]
                  + istress[5] * istress[5];
   term4 *= 6.0;

   double vm = sqrt(0.5 * (term1 + term2 + term3 + term4));

   // set the von Mises quadrature function data
   vmData[elemID * elemVmOffset + ipID * vmOffset] = vm;

   ir = NULL;
   vm_qspace = NULL;
   vm_qf = NULL;
   qspace = NULL;
   vmData = NULL;

   return;
}

// A helper function that takes in a 3x3 rotation matrix and converts it over
// to a unit quaternion.
// rmat should be constant here...
void ExaModel::RMat2Quat(const DenseMatrix& rmat, Vector& quat)
{
   double inv2 = 1.0 / 2.0;
   double phi = 0.0;
   static const double eps = numeric_limits<double>::epsilon();
   double tr_r = 0.0;
   double inv_sin = 0.0;
   double s = 0.0;


   quat = 0.0;

   tr_r = rmat(0, 0) + rmat(1, 1) + rmat(2, 2);
   phi = inv2 * (tr_r - 1.0);
   phi = min(phi, 1.0);
   phi = max(phi, -1.0);
   phi = acos(phi);
   if (abs(phi) < eps) {
      quat[3] = 1.0;
   }
   else {
      inv_sin = 1.0 / sin(phi);
      quat[0] = phi;
      quat[1] = inv_sin * inv2 * (rmat(2, 1) - rmat(1, 2));
      quat[2] = inv_sin * inv2 * (rmat(0, 2) - rmat(2, 0));
      quat[3] = inv_sin * inv2 * (rmat(1, 0) - rmat(0, 1));
   }

   s = sin(inv2 * quat[0]);
   quat[0] = cos(quat[0] * inv2);
   quat[1] = s * quat[1];
   quat[2] = s * quat[2];
   quat[3] = s * quat[3];

   return;
}

// A helper function that takes in a unit quaternion and and returns a 3x3 rotation
// matrix.
void ExaModel::Quat2RMat(const Vector& quat, DenseMatrix& rmat)
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

   return;
}

// The below method computes the polar decomposition of a 3x3 matrix using a method
// proposed in: https://animation.rwth-aachen.de/media/papers/2016-MIG-StableRotation.pdf
// The paper listed provides a fast and robust way to obtain the rotation portion
// of a positive definite 3x3 matrix which then allows for the easy computation
// of U and V.
void ExaModel::CalcPolarDecompDefGrad(DenseMatrix& R, DenseMatrix& U,
                                      DenseMatrix& V, double err)
{
   DenseMatrix omega_mat, temp;
   DenseMatrix def_grad(R, 3);

   int dim = 3;
   Vector quat;

   int max_iter = 500;

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

      w_bot = (1.0 / (abs(r1da1 + r2da2 + r3da3) + err));
      // The axial vector that shows up in the paper
      w[0] = w_top[0] * w_bot; w[1] = w_top[1] * w_bot; w[2] = w_top[2] * w_bot;
      // The norm of the axial vector
      w_norm = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
      // If the norm is below our desired error we've gotten our solution
      // So we can break out of the loop
      if (w_norm < err) {
         break;
      }
      // The exponential mapping for an axial vector
      // The 3x3 case has been explicitly unrolled here
      w_norm_inv2 = 1.0 / (w_norm * w_norm);
      w_norm_inv = 1.0 / w_norm;

      sth = sin(w_norm) * w_norm_inv;
      cth = (1.0 - cos(w_norm)) * w_norm_inv2;

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

   return;
}

// This method calculates the Eulerian strain which is given as:
// e = 1/2 (I - B^(-1)) = 1/2 (I - F(^-T)F^(-1))
void ExaModel::CalcEulerianStrain(DenseMatrix& E, const DenseMatrix &F)
{
   int dim = 3;

   DenseMatrix Finv(dim), Binv(dim);

   double half = 1.0 / 2.0;

   CalcInverse(F, Finv);

   MultAtB(Finv, Finv, Binv);

   E = 0.0;

   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         E(i, j) -= half * Binv(i, j);
      }

      E(j, j) += half;
   }

   return;
}

// This method calculates the Lagrangian strain which is given as:
// E = 1/2 (C - I) = 1/2 (F^(T)F - I)
void ExaModel::CalcLagrangianStrain(DenseMatrix& E, const DenseMatrix &F)
{
   int dim = 3;

   // DenseMatrix F(Jpt, dim);
   DenseMatrix C(dim);

   double half = 1.0 / 2.0;

   MultAtB(F, F, C);

   E = 0.0;

   for (int j = 0; j < dim; j++) {
      for (int i = 0; i < dim; i++) {
         E(i, j) += half * C(i, j);
      }

      E(j, j) -= half;
   }

   return;
}

// This method calculates the Biot strain which is given as:
// E = (U - I) or sometimes seen as E = (V - I) if R = I
void ExaModel::CalcBiotStrain(DenseMatrix& E, const DenseMatrix &F)
{
   int dim = 3;

   DenseMatrix rmat(F, dim);
   DenseMatrix umat, vmat;

   umat.SetSize(dim);
   vmat.SetSize(dim);

   CalcPolarDecompDefGrad(rmat, umat, vmat);

   E = umat;
   E(0, 0) -= 1.0;
   E(1, 1) -= 1.0;
   E(2, 2) -= 1.0;

   return;
}

void ExaModel::CalcLogStrain(DenseMatrix& E, const DenseMatrix &F)
{
   // calculate current end step logorithmic strain (Hencky Strain)
   // which is taken to be E = ln(U) = 1/2 ln(C), where C = (F_T)F.
   // We have incremental F from MFEM, and store F0 (Jpt0) so
   // F = F_hat*F0. With F, use a spectral decomposition on C to obtain a
   // form where we only have to take the natural log of the
   // eigenvalues
   // UMAT uses the E = ln(V) approach instead

   DenseMatrix B;

   int dim = 3;

   B.SetSize(dim);
   // F.SetSize(dim);

   // F = Jpt;

   MultABt(F, F, B);

   // compute eigenvalue decomposition of B
   double lambda[dim];
   double vec[dim * dim];
   // fix_me: was failing
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

   return;
}

// This function is used in generating the B matrix commonly seen in the formation of
// the material tangent stiffness matrix in mechanics [B^t][Cstiff][B]
// Although we're goint to return really B^t here since it better matches up
// with how DS is set up memory wise
// The B matrix should have dimensions equal to (dof*dim, 6).
// We assume it hasn't been initialized ahead of time or it's already
// been written in, so we rewrite over everything in the below.
void ExaModel::GenerateGradMatrix(const DenseMatrix& DS, DenseMatrix& B)
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

   return;
}

void ExaModel::GenerateGradGeomMatrix(const DenseMatrix& DS, DenseMatrix& Bgeom)
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

// member functions for the ExaNLFIntegrator
double ExaNLFIntegrator::GetElementEnergy(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun)
{
   // we are not interested in the element energy at this time
   (void) el;
   (void) Ttr;
   (void) elfun;

   return 0.0;
}

// Outside of the UMAT function calls this should be the function called
// to assemble our residual vectors.
void ExaNLFIntegrator::AssembleElementVector(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector &elfun, Vector &elvect)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DenseMatrix DSh, DS;
   DenseMatrix Jpt;
   DenseMatrix PMatI, PMatO;
   // This is our stress tensor
   DenseMatrix P(3);

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jpt.SetSize(dim);

   // PMatI would be our velocity in this case
   PMatI.UseExternalData(elfun.GetData(), dof, dim);
   elvect.SetSize(dof * dim);

   // PMatO would be our residual vector
   elvect = 0.0;
   PMatO.UseExternalData(elvect.GetData(), dof, dim);

   const IntegrationRule *ir = IntRule;
   if (!ir) {
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1)); // must match quadrature space
   }

   for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);

      // compute Jacobian of the transformation
      Jpt = Ttr.InverseJacobian(); // Jrt = dxi / dX

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jpt, DS); // dN_a(xi) / dX = dN_a(xi)/dxi * dxi/dX

      double stress[6];
      model->GetElementStress(Ttr.ElementNo, i, false, stress, 6);
      // Could probably later have this only set once...
      // Would reduce the number mallocs that we're doing and
      // should potentially provide a small speed boost.
      P(0, 0) = stress[0];
      P(1, 1) = stress[1];
      P(2, 2) = stress[2];
      P(1, 2) = stress[3];
      P(0, 2) = stress[4];
      P(0, 1) = stress[5];

      P(2, 1) = P(1, 2);
      P(2, 0) = P(0, 2);
      P(1, 0) = P(0, 1);

      DS *= (Ttr.Weight() * ip.weight);
      AddMult(DS, P, PMatO);
   }

   return;
}

void ExaNLFIntegrator::AssembleElementGrad(
   const FiniteElement &el,
   ElementTransformation &Ttr,
   const Vector & /*elfun*/, DenseMatrix &elmat)
{
   int dof = el.GetDof(), dim = el.GetDim();

   DenseMatrix DSh, DS, Jrt;

   // Now time to start assembling stuff
   DenseMatrix grad_trans, temp;
   DenseMatrix tan_stiff(6);

   int ngrad_dim2 = 36;
   double matGrad[ngrad_dim2];
   // Delta in our timestep
   double dt = model->GetModelDt();

   // temp1 is now going to become the transpose Bmatrix as seen in
   // [B^t][tan_stiff][B]
   grad_trans.SetSize(dof * dim, 6);
   // We need a temp matrix to store our first matrix results as seen in here
   temp.SetSize(6, dof * dim);

   tan_stiff.UseExternalData(&matGrad[0], 6, 6);

   DSh.SetSize(dof, dim);
   DS.SetSize(dof, dim);
   Jrt.SetSize(dim);
   elmat.SetSize(dof * dim);

   const IntegrationRule *ir = IntRule;
   if (!ir) {
      ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1)); // <--- must match quadrature space
   }

   elmat = 0.0;

   for (int i = 0; i < ir->GetNPoints(); i++) {
      const IntegrationPoint &ip = ir->IntPoint(i);
      Ttr.SetIntPoint(&ip);
      CalcInverse(Ttr.Jacobian(), Jrt);

      el.CalcDShape(ip, DSh);
      Mult(DSh, Jrt, DS);

      model->GetElementMatGrad(Ttr.ElementNo, i, matGrad, ngrad_dim2);
      // temp1 is B^t
      model->GenerateGradMatrix(DS, grad_trans);
      // We multiple our quadrature wts here to our tan_stiff matrix
      tan_stiff *= dt * ip.weight * Ttr.Weight();
      // We use kgeom as a temporary matrix
      // kgeom = [Cstiff][B]
      MultABt(tan_stiff, grad_trans, temp);
      // We now add our [B^t][kgeom] product to our tangent stiffness matrix that
      // we want to output to our material tangent stiffness matrix
      AddMult(grad_trans, temp, elmat);
   }

   return;
}

// In the below function we'll be applying the below action on our material
// tangent matrix C^{tan} at each quadrature point as:
// D_{ijkm} = 1 / det(J) * w_{qpt} * adj(J)^T_{ij} C^{tan}_{ijkl} adj(J)_{lm}
// where D is our new 4th order tensor, J is our jacobian calculated from the
// mesh geometric factors, and adj(J) is the adjugate of J.
void ExaNLFIntegrator::AssemblePAGrad(const FiniteElementSpace &fes)
{
   Mesh *mesh = fes.GetMesh();
   const FiniteElement &el = *fes.GetFE(0);
   space_dims = el.GetDim();
   const IntegrationRule *ir = &(IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

   nqpts = ir->GetNPoints();
   nnodes = el.GetDof();
   nelems = fes.GetNE();
   const GeometricFactors *geom = mesh->GetGeometricFactors(*ir, GeometricFactors::JACOBIANS);
   auto W = ir->GetWeights().Read();

   if ((space_dims == 1) || (space_dims == 2)) {
      MFEM_ABORT("Dimensions of 1 or 2 not supported.");
   }
   else {
      const int dim = 3;

      if (grad.Size() != (nqpts * dim * nnodes)) {
         grad.SetSize(nqpts * dim * nnodes);
         {
            DenseMatrix DSh(nnodes, space_dims);
            const int offset = nnodes * dim;
            double *qpts_dshape_data = grad.GetData();
            for (int i = 0; i < nqpts; i++) {
               const IntegrationPoint &ip = ir->IntPoint(i);
               DSh.UseExternalData(&qpts_dshape_data[offset * i], nnodes, dim);
               el.CalcDShape(ip, DSh);
            }
         }
      }

      // geom->J really isn't going to work for us as of right now. We could just reorder it
      // to the version that we want it to be in instead...
      Vector jacobian(dim * dim * nqpts * nelems);
      pa_dmat.SetSize(dim * dim * dim * dim * nqpts * nelems);
      pa_dmat = 0.0;

      const int DIM2 = 2;
      const int DIM4 = 4;
      const int DIM6 = 6;
      std::array<RAJA::idx_t, DIM6> perm6 {{ 5, 4, 3, 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM4> perm4 {{ 3, 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };

      // bunch of helper RAJA views to make dealing with data easier down below in our kernel.

      RAJA::Layout<DIM6> layout_4Dtensor = RAJA::make_permuted_layout({{ dim, dim, dim, dim, nqpts, nelems } }, perm6);
      RAJA::View<const double, RAJA::Layout<DIM6, RAJA::Index_type, 0> > C(model->GetMTanData(), layout_4Dtensor);
      // Swapped over to row order since it makes sense in later applications...
      // Should make C row order as well for PA operations
      RAJA::View<double, RAJA::Layout<DIM6> > D(pa_dmat.GetData(), nelems, nqpts, dim, dim, dim, dim);

      RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{ dim, dim, nqpts, nelems } }, perm4);
      RAJA::View<double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian.GetData(), layout_jacob);

      RAJA::Layout<DIM4> layout_geom = RAJA::make_permuted_layout({{ nqpts, dim, dim, nelems } }, perm4);
      RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > geom_j_view(geom->J.GetData(), layout_geom);

      RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{ dim, dim } }, perm2);
      // Should replace these with RAJA foralls or kernels at some point in time.
      // fix me
      for (int i = 0; i < nelems; i++) {
         for (int j = 0; j < nqpts; j++) {
            for (int k = 0; k < dim; k++) {
               for (int l = 0; l < dim; l++) {
                  J(l, k, j, i) = geom_j_view(j, l, k, i);
               }
            }
         }
      }

      // This loop we'll want to parallelize the rest are all serial for now.
      for (int i_elems = 0; i_elems < nelems; i_elems++) {
         double adj[dim * dim];
         double c_detJ;
         // So, we're going to say this view is constant however we're going to mutate the values only in
         // that one scoped section for the quadrature points.
         RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > A(&adj[0], layout_adj);
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            // If we scope this then we only need to carry half the number of variables around with us for
            // the adjugate term.
            {
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
               c_detJ = 1.0 / detJ * W[j_qpts];
               // adj(J)
               adj[0] = (J22 * J33) - (J23 * J32); // 0,0
               adj[1] = (J32 * J13) - (J12 * J33); // 0,1
               adj[2] = (J12 * J23) - (J22 * J13); // 0,2
               adj[3] = (J31 * J23) - (J21 * J33); // 1,0
               adj[4] = (J11 * J33) - (J13 * J31); // 1,1
               adj[5] = (J21 * J13) - (J11 * J23); // 1,2
               adj[6] = (J21 * J32) - (J31 * J22); // 2,0
               adj[7] = (J31 * J12) - (J11 * J32); // 2,1
               adj[8] = (J11 * J22) - (J12 * J21); // 2,2
            }
            // Unrolled part of the loops just so we wouldn't have so many nested ones.
            // If we were to get really ambitious we could eliminate also the m indexed
            // loop...
            for (int n = 0; n < dim; n++) {
               for (int m = 0; m < dim; m++) {
                  for (int l = 0; l < dim; l++) {
                     D(i_elems, j_qpts, 0, 0, l, n) += (A(0, 0) * C(0, 0, l, m, j_qpts, i_elems) +
                                                        A(1, 0) * C(1, 0, l, m, j_qpts, i_elems) +
                                                        A(2, 0) * C(2, 0, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 0, 1, l, n) += (A(0, 0) * C(0, 1, l, m, j_qpts, i_elems) +
                                                        A(1, 0) * C(1, 1, l, m, j_qpts, i_elems) +
                                                        A(2, 0) * C(2, 1, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 0, 2, l, n) += (A(0, 0) * C(0, 2, l, m, j_qpts, i_elems) +
                                                        A(1, 0) * C(1, 2, l, m, j_qpts, i_elems) +
                                                        A(2, 0) * C(2, 2, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 1, 0, l, n) += (A(0, 1) * C(0, 0, l, m, j_qpts, i_elems) +
                                                        A(1, 1) * C(1, 0, l, m, j_qpts, i_elems) +
                                                        A(2, 1) * C(2, 0, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 1, 1, l, n) += (A(0, 1) * C(0, 1, l, m, j_qpts, i_elems) +
                                                        A(1, 1) * C(1, 1, l, m, j_qpts, i_elems) +
                                                        A(2, 1) * C(2, 1, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 1, 2, l, n) += (A(0, 1) * C(0, 2, l, m, j_qpts, i_elems) +
                                                        A(1, 1) * C(1, 2, l, m, j_qpts, i_elems) +
                                                        A(2, 1) * C(2, 2, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 2, 0, l, n) += (A(0, 2) * C(0, 0, l, m, j_qpts, i_elems) +
                                                        A(1, 2) * C(1, 0, l, m, j_qpts, i_elems) +
                                                        A(2, 2) * C(2, 0, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 2, 1, l, n) += (A(0, 2) * C(0, 1, l, m, j_qpts, i_elems) +
                                                        A(1, 2) * C(1, 1, l, m, j_qpts, i_elems) +
                                                        A(2, 2) * C(2, 1, l, m, j_qpts, i_elems)) * A(m, n);
                     D(i_elems, j_qpts, 2, 2, l, n) += (A(0, 2) * C(0, 2, l, m, j_qpts, i_elems) +
                                                        A(1, 2) * C(1, 2, l, m, j_qpts, i_elems) +
                                                        A(2, 2) * C(2, 2, l, m, j_qpts, i_elems)) * A(m, n);
                  }
               }
            } // End of Dikln = adj(J)_{ji} C_{jklm} adj(J)_{mn} loop

            // Unrolled part of the loops just so we wouldn't have so many nested ones.
            for (int n = 0; n < dim; n++) {
               for (int l = 0; l < dim; l++) {
                  D(i_elems, j_qpts, l, n, 0, 0) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 0, 1) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 0, 2) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 1, 0) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 1, 1) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 1, 2) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 2, 0) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 2, 1) *= c_detJ;
                  D(i_elems, j_qpts, l, n, 2, 2) *= c_detJ;
               }
            } // End of D_{ijkl} *= 1/det(J) * w_{qpt} loop
         } // End of quadrature loop
      } // End of Elements loop
   } // End of else statement
}

// Here we're applying the following action operation using the assembled "D" 4th order
// tensor found above:
// y_{ik} = \nabla_{ij}\phi^T_{\epsilon} D_{jklm} \nabla_{mn}\phi_{\epsilon} x_{nl}
void ExaNLFIntegrator::AddMultPAGrad(const mfem::Vector &x, mfem::Vector &y)
{
   if ((space_dims == 1) || (space_dims == 2)) {
      MFEM_ABORT("Dimensions of 1 or 2 not supported.");
   }
   else {
      const int dim = 3;
      const int DIM2 = 2;
      const int DIM3 = 3;
      const int DIM6 = 6;

      std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };
      std::array<RAJA::idx_t, DIM2> perm2 {{ 1, 0 } };
      // Swapped over to row order since it makes sense in later applications...
      // Should make C row order as well for PA operations
      RAJA::View<double, RAJA::Layout<DIM6> > D(pa_dmat.GetData(), nelems, nqpts, dim, dim, dim, dim);
      // Our field variables that are inputs and outputs
      RAJA::Layout<DIM3> layout_field = RAJA::make_permuted_layout({{ nnodes, dim, nelems } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > X(x.GetData(), layout_field);
      RAJA::View<double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Y(y.GetData(), layout_field);
      // Transpose of the local gradient variable
      RAJA::Layout<DIM3> layout_grads = RAJA::make_permuted_layout({{ nnodes, dim, nqpts } }, perm3);
      RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > Gt(grad.GetData(), layout_grads);

      // View for our temporary 2d array
      RAJA::Layout<DIM2> layout_adj = RAJA::make_permuted_layout({{ dim, dim } }, perm2);
      for (int i_elems = 0; i_elems < nelems; i_elems++) {
         for (int j_qpts = 0; j_qpts < nqpts; j_qpts++) {
            double T[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            for (int i = 0; i < dim; i++) {
               for (int j = 0; j < dim; j++) {
                  for (int k = 0; k < nnodes; k++) {
                     T[0] += D(i_elems, j_qpts, 0, 0, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[1] += D(i_elems, j_qpts, 1, 0, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[2] += D(i_elems, j_qpts, 2, 0, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[3] += D(i_elems, j_qpts, 0, 1, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[4] += D(i_elems, j_qpts, 1, 1, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[5] += D(i_elems, j_qpts, 2, 1, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[6] += D(i_elems, j_qpts, 0, 2, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[7] += D(i_elems, j_qpts, 1, 2, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                     T[8] += D(i_elems, j_qpts, 2, 2, i, j) * Gt(k, j, j_qpts) * X(k, i, i_elems);
                  }
               }
            } // End of doing tensor contraction of D_{jkmo}G_{op}X_{pm}

            RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > Tview(&T[0], layout_adj);
            for (int k = 0; k < dim; k++) {
               for (int j = 0; j < dim; j++) {
                  for (int i = 0; i < nnodes; i++) {
                     Y(i, k, i_elems) += Gt(i, j, j_qpts) * Tview(j, k);
                  }
               }
            } // End of the final action of Y_{ik} += Gt_{ij} T_{jk}
         } // End of npts
      } // End of nelems
   } // End of if statement
}