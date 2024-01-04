#include "mfem.hpp"
#include "mfem/general/forall.hpp"
#include "mechanics_model.hpp"
#include "mechanics_log.hpp"
#include "BCManager.hpp"
#include <math.h> // log
#include <algorithm>
#include <iostream> // cerr
#include "RAJA/RAJA.hpp"

using namespace mfem;
using namespace std;

void computeDefGrad(QuadratureFunction *qf, ParFiniteElementSpace *fes,
                    Vector &x0)
{
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->ReadWrite();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpaceBase* qspace = qf->GetSpace();

   ParGridFunction x_gf;

   double* vals = x0.ReadWrite();

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
      // PMatI.UseExternalData(el_x.ReadWrite(), dof, dim);

      // get element physical coordinates
      Array<int> vdofs(dof * dim);
      Vector el_x(PMatI.Data(), dof * dim);
      fes->GetElementVDofs(i, vdofs);

      x_gf.GetSubVector(vdofs, el_x);

      ir = &(qspace->GetIntRule(i));
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

ExaModel::ExaModel(mfem::QuadratureFunction *q_stress0, mfem::QuadratureFunction *q_stress1,
                   mfem::QuadratureFunction *q_matGrad, mfem::QuadratureFunction *q_matVars0,
                   mfem::QuadratureFunction *q_matVars1,
                   mfem::ParGridFunction* _beg_coords, mfem::ParGridFunction* _end_coords,
                   mfem::Vector *props, int nProps, int nStateVars, Assembly _assembly) :
         numProps(nProps), numStateVars(nStateVars),
         beg_coords(_beg_coords),
         end_coords(_end_coords),
         stress0(q_stress0),
         stress1(q_stress1),
         matGrad(q_matGrad),
         matVars0(q_matVars0),
         matVars1(q_matVars1),
         matProps(props),
         assembly(_assembly)
      {
         if (assembly == Assembly::PA) {
            int npts = q_matGrad->Size() / q_matGrad->GetVDim();
            matGradPA.SetSize(81 * npts, mfem::Device::GetMemoryType());
            matGradPA.UseDevice(true);
         }
      }

// This method sets the end time step stress to the beginning step
// and then returns the internal data pointer of the end time step
// array.
double* ExaModel::StressSetup()
{
   const double *stress_beg = stress0->Read();
   double *stress_end = stress1->ReadWrite();
   const int N = stress0->Size();
   MFEM_FORALL(i, N, stress_end[i] = stress_beg[i]; );

   return stress_end;
}

// This methods set the end time step state variable array to the
// beginning time step values and then returns the internal data pointer
// of the end time step array.
double* ExaModel::StateVarsSetup()
{
   const double *state_vars_beg = matVars0->Read();
   double *state_vars_end = matVars1->ReadWrite();

   const int N = matVars0->Size();
   MFEM_FORALL(i, N, state_vars_end[i] = state_vars_beg[i]; );

   return state_vars_end;
}

// the getter simply returns the beginning step stress
void ExaModel::GetElementStress(const int elID, const int ipNum,
                                bool beginStep, double* stress, int numComps)
{
   const IntegrationRule *ir = NULL;
   double* qf_data = NULL;
   int qf_offset = 0;
   QuadratureFunction* qf = NULL;
   QuadratureSpaceBase* qspace = NULL;

   if (beginStep) {
      qf = stress0;
   }
   else {
      qf = stress1;
   }

   qf_data = qf->HostReadWrite();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nGetElementStress: number of components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetIntRule(elID));
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
   QuadratureSpaceBase* qspace;

   if (beginStep) {
      qf = stress0;
   }
   else {
      qf = stress1;
   }

   qf_data = qf->HostReadWrite();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nSetElementStress: number of components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetIntRule(elID));
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
   QuadratureSpaceBase* qspace;

   if (beginStep) {
      qf = matVars0;
   }
   else {
      qf = matVars1;
   }

   qf_data = qf->ReadWrite();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nGetElementStateVars: num. components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetIntRule(elID));
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
   QuadratureSpaceBase* qspace;

   if (beginStep) {
      qf = matVars0;
   }
   else {
      qf = matVars1;
   }

   qf_data = qf->ReadWrite();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nSetElementStateVars: num. components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetIntRule(elID));
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
   QuadratureSpaceBase* qspace;

   qf = matGrad;

   qf_data = qf->HostReadWrite();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nGetElementMatGrad: num. components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetIntRule(elID));
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
   QuadratureSpaceBase* qspace;

   qf = matGrad;

   qf_data = qf->ReadWrite();
   qf_offset = qf->GetVDim();
   qspace = qf->GetSpace();

   // check offset to input number of components
   if (qf_offset != numComps) {
      cerr << "\nSetElementMatGrad: num. components does not match quad func offset"
           << endl;
   }

   ir = &(qspace->GetIntRule(elID));
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
   double* mpdata = matProps->ReadWrite();
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
   stress0->Swap(*stress1);
}

void ExaModel::UpdateStateVars()
{
   matVars0->Swap(*matVars1);
}

void ExaModel::UpdateEndCoords(const Vector& vels)
{
   int size;

   size = vels.Size();

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

   const double* bcrd = bcrds.Read();
   const double* vel = vels.Read();
   double* end_crd = end_crds.ReadWrite();
   const double dt_ = this->dt;
   // Perform a simple time integration to get our new end time step coordinates
   MFEM_FORALL(i, size, {
      end_crd[i] = vel[i] * dt_ + bcrd[i];
   });

   // Now make sure the update gets sent to all the other processors that have ghost copies
   // of our data.
   end_coords->Distribute(end_crds);

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

void ExaModel::GenerateGradBarMatrix(const mfem::DenseMatrix& DS, const mfem::DenseMatrix& eDS, mfem::DenseMatrix& B)
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

// This takes in the material gradient matrix that's being used in most models as the 2D
// version and saves off the 4D space version
void ExaModel::TransformMatGradTo4D()
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
   MFEM_FORALL(i, npts, {
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
