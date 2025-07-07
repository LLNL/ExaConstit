#include "models/mechanics_model.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

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
   auto qspace = qf->GetSpaceShared();

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
   }

   return;
}

// NEW CONSTRUCTOR: Much simpler parameter list focused on essential information
// The region parameter is key - it tells this model instance which material region
// it should manage, enabling proper data access through SimulationState
ExaModel::ExaModel(const int region, int nStateVars, SimulationState& sim_state) :
         numStateVars(nStateVars),
         m_region(region),
         assembly(sim_state.getOptions().solvers.assembly),
         m_sim_state(sim_state) {}

// Get material properties for this region from SimulationState
// This replaces direct access to the matProps vector member variable
const std::vector<double>& ExaModel::GetMaterialProperties() const {
    std::string region_name = m_sim_state.GetRegionName(m_region);
    // Note: You'll need to expose this method in SimulationState or make it accessible
    // For now, assuming there's a public getter or friend access
    return m_sim_state.GetMaterialProperties(region_name);
}
