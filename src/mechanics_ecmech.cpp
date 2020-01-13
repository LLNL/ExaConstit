#include "mfem.hpp"
#include "ECMech_cases.h"
#include "ECMech_evptnWrap.h"
#include "mechanics_integrators.hpp"
#include "mechanics_ecmech.hpp"
#include "BCManager.hpp"
#include <math.h> // log
#include <algorithm>
#include <iostream> // cerr
#include "RAJA/RAJA.hpp"
//#include "exacmech.hpp" //Will need to export all of the various header files into here as well

//namespace mfem
//{
using namespace mfem;
using namespace std;
using namespace ecmech;

void ExaCMechModel::UpdateModelVars(){}

//For ExaCMechModel definitions the ecmech namespace is useful
void ExaCMechModel::ModelSetup(const int nqpts, const int nelems, const int space_dim,
                     const int nnodes, const Vector &jacobian,
                     const Vector &loc_grad, const Vector &vel) {

   // set properties and state variables length (hard code for now);
   int nstatv = numStateVars;
   int ntens = 6; 

   double statev[nstatv]; // populate from the state variables associated with this element/ip

   double stress[ecmech::nsvec]; // Cauchy stress at ip
   double stress_svec_p[ecmech::nsvp]; //Cauchy stress at ip layed out as deviatoric portion first then pressure term

   double d_svec_p[ecmech::nsvp];
   double w_vec[ecmech::nwvec];

   double ddsdde[ecmech::nsvp * ecmech::nsvp]; // output Jacobian matrix of the constitutive model.
                     // ddsdde(i,j) defines the change in the ith stress component 
                     // due to an incremental perturbation in the jth deformation rate component

   double sdd[ecmech::nsdd]; //not really used for our purposes as a quasi-static type code
   double vol_ratio[4];
   double eng_int[ecmech::ne];
   double d_mean;
   double stress_mean;

   const int space_dim2 = space_dim * space_dim;
   const double *jacobian_data = jacobian.GetData();
   const double *loc_grad_data = loc_grad.GetData(); 
   const double *vel_data = vel.GetData();
   double vgrad[space_dim2];

   const int DIM4 = 4;
   const int DIM3 = 3;
   const int DIM2 = 2;
   std::array<RAJA::idx_t, DIM4> perm4 {{3, 2, 1, 0 }};
   std::array<RAJA::idx_t, DIM3> perm3{{2, 1, 0 }};
   std::array<RAJA::idx_t, DIM2> perm2{{1, 0 }};
   //bunch of helper RAJA views to make dealing with data easier down below in our kernel.
   RAJA::Layout<DIM4> layout_jacob = RAJA::make_permuted_layout({{space_dim, space_dim, nqpts, nelems}}, perm4);
   RAJA::View<const double, RAJA::Layout<DIM4, RAJA::Index_type, 0> > J(jacobian_data, layout_jacob);

   RAJA::Layout<DIM3> layout_vel = RAJA::make_permuted_layout({{nnodes, space_dim, nelems}}, perm3);
   RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > vel_view(vel_data, layout_vel);

   RAJA::Layout<DIM3> layout_loc_grad = RAJA::make_permuted_layout({{nnodes, space_dim, nqpts}}, perm3);
   RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > loc_grad_view(loc_grad_data, layout_loc_grad);

   RAJA::Layout<DIM2> layout_vgrad = RAJA::make_permuted_layout({{space_dim, space_dim}}, perm2);
   RAJA::View<double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > vgrad_view(&vgrad[0], layout_vgrad);

   for(int i = 0; i < nelems; i++){
      for(int j = 0; j < nqpts; j++){
         //initialize vgrad to 0
         for(int q = 0; q < space_dim2; q++){
            vgrad[q] = 0.0;
         }
         //scoping the next set of work related to calculating the velocity gradient.
         {
            const double J11 = J(0, 0, j, i); //0,0
            const double J21 = J(1, 0, j, i); //1,0
            const double J31 = J(2, 0, j, i); //2,0
            const double J12 = J(0, 1, j, i); //0,1
            const double J22 = J(1, 1, j, i); //1,1
            const double J32 = J(2, 1, j, i); //2,1
            const double J13 = J(0, 2, j, i); //0,2
            const double J23 = J(1, 2, j, i); //1,2
            const double J33 = J(2, 2, j, i); //2,2
            const double detJ = J11 * (J22 * J33 - J32 * J23) -
            /* */               J21 * (J12 * J33 - J32 * J13) +
            /* */               J31 * (J12 * J23 - J22 * J13);
            const double c_detJ = 1.0 / detJ;
            // adj(J)
            const double A11 = c_detJ * ((J22 * J33) - (J23 * J32));
            const double A12 = c_detJ * ((J32 * J13) - (J12 * J33));
            const double A13 = c_detJ * ((J12 * J23) - (J22 * J13));
            const double A21 = c_detJ * ((J31 * J23) - (J21 * J33));
            const double A22 = c_detJ * ((J11 * J33) - (J13 * J31));
            const double A23 = c_detJ * ((J21 * J13) - (J11 * J23));
            const double A31 = c_detJ * ((J21 * J32) - (J31 * J22));
            const double A32 = c_detJ * ((J31 * J12) - (J11 * J32));
            const double A33 = c_detJ * ((J11 * J22) - (J12 * J21));
            const double A[space_dim2] = {A11, A21, A31, A12, A22, A32, A13, A23, A33};
            //Raja view to make things easier again
            
            RAJA::View<const double, RAJA::Layout<DIM2, RAJA::Index_type, 0> > jinv_view(&A[0], layout_vgrad);
            //All of the data down below is ordered in column major order
            for(int t = 0; t < space_dim; t++){
               for(int s = 0; s < space_dim; s++){
                  for(int r = 0; r < nnodes; r++){
                     for(int q = 0; q < space_dim; q++){
                        vgrad_view(q, t) += vel_view(r, q, i) * loc_grad_view(r, s, j) * jinv_view(s, t);
                     }
                  }
               }
            }//End of loop used to calculate velocity gradient
         }//end velocity gradient scope
         
         //Material model code now

         // get state variables and material properties
         GetElementStateVars(i, j, true, statev, nstatv);
         // get element stress and make sure ordering is ok
         GetElementStress(i, j, true, stress, 6);

         // initialize 6x6 2d arrays
         for (int q = 0; q < ecmech::nsvec; ++q) 
         {
            for (int r = 0; r < ecmech::nsvec; ++r) 
            {
               ddsdde[(q * ecmech::nsvec) + r] = 0.0;
            }
         }
         
         for(int q = 0; q < ecmech::ne; ++q)
         {
            eng_int[q] = statev[ind_int_eng + q];
         }

         if(init_step){
            w_vec[0] = 0.0;
            w_vec[1] = 0.0;
            w_vec[2] = 0.0;

            d_svec_p[0] = 0.0;
            d_svec_p[1] = 0.0;
            d_svec_p[2] = 0.0;
            d_svec_p[3] = 0.0;
            d_svec_p[4] = 0.0;
            d_svec_p[5] = 0.0;
            d_svec_p[6] = 0.0;

            vol_ratio[0] = statev[ind_vols];
            vol_ratio[1] = vol_ratio[0] * exp(d_svec_p[ecmech::iSvecP] * dt);
            vol_ratio[3] = vol_ratio[1] - vol_ratio[0];
            vol_ratio[2] = vol_ratio[3] / (dt * 0.5 * (vol_ratio[0] + vol_ratio[1]));

            std::copy(stress, stress + ecmech::nsvec, stress_svec_p);

            stress_mean = -ecmech::onethird * (stress[0] + stress[1] + stress[2]);
            stress_svec_p[0] += stress_mean;
            stress_svec_p[1] += stress_mean;
            stress_svec_p[2] += stress_mean;
            stress_svec_p[ecmech::iSvecP] = stress_mean;

            mat_model_base->getResponse(dt, &d_svec_p[0], &w_vec[0], &vol_ratio[0],
                                       &eng_int[0], &stress_svec_p[0], &statev[0], &temp_k, &sdd[0], &ddsdde[0], 1);

            //ExaCMech saves this in Row major, so we need to get out the transpose.                                                                                                          
            //The good thing is we can do this all in place no problem.                                                                                                                       
            for (int q = 0; q < ecmech::ntvec; ++q)
            {
               for (int r = q + 1; r < ecmech::nsvec; ++r)
               {                                                                                                                                     
                  std::swap(ddsdde[(ecmech::nsvec * r) + q], ddsdde[(ecmech::nsvec * q) + r]);
               }                                                                                                                                          
            }

            //Here we have the skew portion of our velocity gradient as represented as an                                                                                                     
            //axial vector.                                                                                                                                                                   
            w_vec[0] = 0.5 * (vgrad_view(2, 1) - vgrad_view(1, 2));
            w_vec[1] = 0.5 * (vgrad_view(0, 2) - vgrad_view(2, 0));
            w_vec[2] = 0.5 * (vgrad_view(1, 0) - vgrad_view(0, 1));

            //Really we're looking at the negative of J but this will do...                                                                                                                   
            d_mean = -ecmech::onethird * (vgrad_view(0, 0) + vgrad_view(1, 1) + vgrad_view(2, 2));
            //The 1st 6 components are the symmetric deviatoric portion of our                                                                                                                
            //Vgrad or J as seen here                                                                                                                                                         
            //The last value is the minus of hydrostatic term so the "pressure"                                                                                                               
            d_svec_p[0] = vgrad_view(0, 0) + d_mean;
            d_svec_p[1] = vgrad_view(1, 1) + d_mean;
            d_svec_p[2] = vgrad_view(2, 2) + d_mean;
            d_svec_p[3] = 0.5 * (vgrad_view(2, 1) + vgrad_view(1, 2));
            d_svec_p[4] = 0.5 * (vgrad_view(2, 0) + vgrad_view(0, 2));
            d_svec_p[5] = 0.5 * (vgrad_view(1, 0) + vgrad_view(0, 1));
            d_svec_p[6] = -3 * d_mean;

            vol_ratio[0] = statev[ind_vols];
            vol_ratio[1] = vol_ratio[0] * exp(d_svec_p[ecmech::iSvecP] * dt);
            vol_ratio[3] = vol_ratio[1] - vol_ratio[0];
            vol_ratio[2] = vol_ratio[3] / (dt * 0.5 * (vol_ratio[0] + vol_ratio[1]));

            std::copy(stress, stress + ecmech::nsvec, stress_svec_p);

            stress_mean = -ecmech::onethird * (stress[0] + stress[1] + stress[2]);
            stress_svec_p[0] += stress_mean;
            stress_svec_p[1] += stress_mean;
            stress_svec_p[2] += stress_mean;
            stress_svec_p[ecmech::iSvecP] = stress_mean;

            mat_model_base->getResponse(dt, &d_svec_p[0], &w_vec[0], &vol_ratio[0],
                                       &eng_int[0], &stress_svec_p[0], &statev[0], &temp_k, &sdd[0], nullptr, 1);

         }else
         {
            //Here we have the skew portion of our velocity gradient as represented as an
            //axial vector.
            w_vec[0] = 0.5 * (vgrad_view(2, 1) - vgrad_view(1, 2));
            w_vec[1] = 0.5 * (vgrad_view(0, 2) - vgrad_view(2, 0));
            w_vec[2] = 0.5 * (vgrad_view(1, 0) - vgrad_view(0, 1));

            //Really we're looking at the negative of J but this will do...
            d_mean = -ecmech::onethird * (vgrad_view(0, 0) + vgrad_view(1, 1) + vgrad_view(2, 2));
            //The 1st 6 components are the symmetric deviatoric portion of our
            //Vgrad or J as seen here
            //The last value is the minus of hydrostatic term so the "pressure"
            d_svec_p[0] = vgrad_view(0, 0) + d_mean; 
            d_svec_p[1] = vgrad_view(1, 1) + d_mean; 
            d_svec_p[2] = vgrad_view(2, 2) + d_mean; 
            d_svec_p[3] = 0.5 * (vgrad_view(2, 1) + vgrad_view(1, 2));
            d_svec_p[4] = 0.5 * (vgrad_view(2, 0) + vgrad_view(0, 2));
            d_svec_p[5] = 0.5 * (vgrad_view(1, 0) + vgrad_view(0, 1));
            d_svec_p[6] = -3 * d_mean;

            vol_ratio[0] = statev[ind_vols];
            vol_ratio[1] = vol_ratio[0] * exp(d_svec_p[ecmech::iSvecP] * dt);
            vol_ratio[3] = vol_ratio[1] - vol_ratio[0];
            vol_ratio[2] = vol_ratio[3] / (dt * 0.5 * (vol_ratio[0] + vol_ratio[1]));

            std::copy(stress, stress + ecmech::nsvec, stress_svec_p);

            stress_mean = -ecmech::onethird * (stress[0] + stress[1] + stress[2]);
            stress_svec_p[0] += stress_mean;
            stress_svec_p[1] += stress_mean;
            stress_svec_p[2] += stress_mean;
            stress_svec_p[ecmech::iSvecP] = stress_mean;

            mat_model_base->getResponse(dt, &d_svec_p[0], &w_vec[0], &vol_ratio[0],
                           &eng_int[0], &stress_svec_p[0], &statev[0], &temp_k, &sdd[0], &ddsdde[0], 1);

            //ExaCMech saves this in Row major, so we need to get out the transpose.
            //The good thing is we can do this all in place no problem.
            for (int q = 0; q < ecmech::ntvec; ++q)
            {
               for (int r = q + 1; r < ecmech::nsvec; ++r)
               {
                  std::swap(ddsdde[(ecmech::nsvec * r) + q], ddsdde[(ecmech::nsvec * q) + r]);
               }
            }
         }//endif

         //We need to update our state variables to include the volume ratio and 
         //internal energy portions
         statev[ind_vols] = vol_ratio[1];
         for(int q = 0; q < ecmech::ne; ++q)
         {
            statev[ind_int_eng + q] = eng_int[q];
         }
         //Here we're converting back from our deviatoric + pressure representation of our
         //Cauchy stress back to the Voigt notation of stress.
         stress_mean = -stress_svec_p[ecmech::iSvecP];
         std::copy(stress_svec_p, stress_svec_p + ecmech::nsvec, stress);
         stress[0] += stress_mean;
         stress[1] += stress_mean;
         stress[2] += stress_mean;

         // set the material stiffness on the model
         SetElementMatGrad(i, j, ddsdde, ntens * ntens);
         // set the updated stress values
         SetElementStress(i, j, false, stress, ntens);
         // set the updated statevars
         SetElementStateVars(i, j, false, statev, nstatv);

      }



   }


}
void ExaCMechModel::EvalModel(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double qptWeight, const double elemVol, 
                          const int elemID, const int ipID, DenseMatrix &PMatO)
{
   
   double stress[ecmech::nsvec];
   GetElementStress(elemID, ipID, false, stress, 6);
   //This could become a problem when we have this all vectorized to run on the GPU... 
   //Could probably later have this only set once...
   //Would reduce the number mallocs that we're doing and
   //should potentially provide a small speed boost.
   DenseMatrix P(3);

   P(0, 0) = stress[0];
   P(1, 1) = stress[1];
   P(2, 2) = stress[2];
   P(1, 2) = stress[3];
   P(0, 2) = stress[4];
   P(0, 1) = stress[5];

   P(2, 1) = P(1, 2);
   P(2, 0) = P(0, 2);
   P(1, 0) = P(0, 1);

   //The below is letting us just do: Int_{body} B^t sigma dV
   DenseMatrix DSt(DS);
   DSt *= (qptWeight * elemVol);
   
   AddMult(DSt, P, PMatO);
   
   return;
}

//The formulation for this is essentially the exact same as the Abaqus version for now
//Once a newer version of ExaCMech is updated the only difference for this will be that
//we no longer have to account for the dt term.
void ExaCMechModel::AssembleH(const DenseMatrix &DS, 
                              const int elemID, const int ipID,
                              const double weight, DenseMatrix &A)
{  
   //We currently only take into account the material tangent stiffness contribution
   //Generally for the applications that we are examining one doesn't need to also include
   //the geometrical tangent stiffness contributions. If we start looking at problems where our
   //elements are becoming highly distorted then we'll probably want to bring that factor back into
   //tangent stiffness matrix.
   int offset = 36;
   double matGrad[offset];

   int dof = DS.Height(), dim = DS.Width();
   
   GetElementMatGrad(elemID, ipID, matGrad, offset);

   DenseMatrix Cstiff(matGrad, 6, 6);

   //Now time to start assembling stuff
   
   DenseMatrix temp1, Bt;
   DenseMatrix sbar;
   DenseMatrix BtsigB;
   
   //temp1 is now going to become the transpose Bmatrix as seen in
   //[B^t][Cstiff][B]
   Bt.SetSize(dof*dim, 6);
   //We need a temp matrix to store our first matrix results as seen in here
   temp1.SetSize(6, dof*dim);
   //temp1 is B^t as seen above
   GenerateGradMatrix(DS, Bt);
   //We multiple our quadrature wts here to our Cstiff matrix
   //We also include the dt term here. Later on this dt term won't be needed
   Cstiff *= dt * weight;
   //We use kgeom as a temporary matrix
   //[temp1] = [Cstiff][B]
   MultABt(Cstiff, Bt, temp1);
   //We now add our [B^t][temp1] product to our tangent stiffness matrix that
   //we want to output to our material tangent stiffness matrix
   AddMult(Bt, temp1, A);

   return;
}
