#include "mfem.hpp"
#include "ECMech_cases.h"
#include "ECMech_evptnWrap.h"
#include "mechanics_integrators.hpp"
#include "mechanics_ecmech.hpp"
#include "BCManager.hpp"
#include <math.h> // log
#include <algorithm>
#include <iostream> // cerr
//#include "exacmech.hpp" //Will need to export all of the various header files into here as well

//namespace mfem
//{
using namespace mfem;
using namespace std;
using namespace ecmech;

void ExaCMechModel::UpdateModelVars(){}

//For ExaCMechModel definitions the ecmech namespace is useful
void ExaCMechModel::EvalModel(const DenseMatrix &Jpt, const DenseMatrix &DS,
                          const double qptWeight, const double elemVol, 
                          const int elemID, const int ipID, DenseMatrix &PMatO)
{
   
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

   // get state variables and material properties
   GetElementStateVars(elemID, ipID, true, statev, nstatv);
   // get element stress and make sure ordering is ok
   GetElementStress(elemID, ipID, true, stress, 6);

   // initialize 6x6 2d arrays
   for (int i = 0; i < ecmech::nsvec; ++i) 
   {
      for (int j = 0; j < ecmech::nsvec; ++j) 
      {
         ddsdde[(i * ecmech::nsvec) + j] = 0.0;
      }
   }
   
   for(int i = 0; i < ecmech::ne; ++i)
   {
      eng_int[i] = statev[ind_int_eng + i];
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
      for (int i = 0; i < ecmech::ntvec; ++i)
      {
         for (int j = i + 1; j < ecmech::nsvec; ++j)
         {                                                                                                                                     
            std::swap(ddsdde[(ecmech::nsvec * j) + i], ddsdde[(ecmech::nsvec * i) + j]);
         }                                                                                                                                          
      }

      //Here we have the skew portion of our velocity gradient as represented as an                                                                                                     
      //axial vector.                                                                                                                                                                   
      w_vec[0] = 0.5 * (Jpt(2, 1) - Jpt(1, 2));
      w_vec[1] = 0.5 * (Jpt(0, 2) - Jpt(2, 0));
      w_vec[2] = 0.5 * (Jpt(1, 0) - Jpt(0, 1));

      //Really we're looking at the negative of J but this will do...                                                                                                                   
      d_mean = -ecmech::onethird * (Jpt(0, 0) + Jpt(1, 1) + Jpt(2, 2));
      //The 1st 6 components are the symmetric deviatoric portion of our                                                                                                                
      //Vgrad or J as seen here                                                                                                                                                         
      //The last value is the minus of hydrostatic term so the "pressure"                                                                                                               
      d_svec_p[0] = Jpt(0, 0) + d_mean;
      d_svec_p[1] = Jpt(1, 1) + d_mean;
      d_svec_p[2] = Jpt(2, 2) + d_mean;
      d_svec_p[3] = 0.5 * (Jpt(2, 1) + Jpt(1, 2));
      d_svec_p[4] = 0.5 * (Jpt(2, 0) + Jpt(0, 2));
      d_svec_p[5] = 0.5 * (Jpt(1, 0) + Jpt(0, 1));
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
      w_vec[0] = 0.5 * (Jpt(2, 1) - Jpt(1, 2));
      w_vec[1] = 0.5 * (Jpt(0, 2) - Jpt(2, 0));
      w_vec[2] = 0.5 * (Jpt(1, 0) - Jpt(0, 1));

      //Really we're looking at the negative of J but this will do...
      d_mean = -ecmech::onethird * (Jpt(0, 0) + Jpt(1, 1) + Jpt(2, 2));
      //The 1st 6 components are the symmetric deviatoric portion of our
      //Vgrad or J as seen here
      //The last value is the minus of hydrostatic term so the "pressure"
      d_svec_p[0] = Jpt(0, 0) + d_mean; 
      d_svec_p[1] = Jpt(1, 1) + d_mean; 
      d_svec_p[2] = Jpt(2, 2) + d_mean; 
      d_svec_p[3] = 0.5 * (Jpt(2, 1) + Jpt(1, 2));
      d_svec_p[4] = 0.5 * (Jpt(2, 0) + Jpt(0, 2));
      d_svec_p[5] = 0.5 * (Jpt(1, 0) + Jpt(0, 1));
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
      for (int i = 0; i < ecmech::ntvec; ++i)
      {
         for (int j = i + 1; j < ecmech::nsvec; ++j)
         {
            std::swap(ddsdde[(ecmech::nsvec * j) + i], ddsdde[(ecmech::nsvec * i) + j]);
         }
      }
   }//endif

   //We need to update our state variables to include the volume ratio and 
   //internal energy portions
   statev[ind_vols] = vol_ratio[1];
   for(int i = 0; i < ecmech::ne; ++i)
   {
      statev[ind_int_eng + i] = eng_int[i];
   }
   //Here we're converting back from our deviatoric + pressure representation of our
   //Cauchy stress back to the Voigt notation of stress.
   stress_mean = -stress_svec_p[ecmech::iSvecP];
   std::copy(stress_svec_p, stress_svec_p + ecmech::nsvec, stress);
   stress[0] += stress_mean;
   stress[1] += stress_mean;
   stress[2] += stress_mean;

   // set the material stiffness on the model
   SetElementMatGrad(elemID, ipID, ddsdde, ntens * ntens);
   // set the updated stress values
   SetElementStress(elemID, ipID, false, stress, ntens);
   // set the updated statevars
   SetElementStateVars(elemID, ipID, false, statev, nstatv);
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
