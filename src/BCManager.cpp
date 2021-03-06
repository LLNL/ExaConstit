

#include "mfem.hpp"
#include "BCManager.hpp"
#include <fstream>

using namespace mfem;

void BCManager::updateBCData(mfem::Array<int> & ess_bdr)
{
   m_bcInstances.clear();
   ess_bdr = 0;

   auto ess_vel = map_ess_vel.find(step)->second;
   auto ess_comp = map_ess_comp.find(step)->second;
   auto ess_id = map_ess_id.find(step)->second;

   for (std::uint32_t i = 0; i < ess_id.size(); ++i) {
      // set the boundary condition id based on the attribute id
      int bcID = ess_id[i];

      // instantiate a boundary condition manager instance and
      // create a BCData object
      BCData & bc = this->CreateBCs(bcID);

      // set the velocity component values
      bc.essVel[0] = ess_vel[3 * i];
      bc.essVel[1] = ess_vel[3 * i + 1];
      bc.essVel[2] = ess_vel[3 * i + 2];
      bc.compID = ess_comp[i];

      // set the boundary condition scales
      bc.setScales();

      // set the active boundary attributes
      if (bc.compID != 0) {
         ess_bdr[bcID - 1] = 1;
      }
   }
}

// set partial dof component list for all essential BCs based on my
// custom BC manager and input, srw.
// We probably should move these over to their appropriate location in mfem
// library at some point...
void NonlinearForm::SetEssentialBCPartial(const Array<int> &bdr_attr_is_ess,
                                          Vector *rhs)
{
   Array2D<int> component;
   Array<int> cmp_row;
   // The size here is set explicitly
   component.SetSize(bdr_attr_is_ess.Size(), 3);
   cmp_row.SetSize(3);

   component = 0;
   cmp_row = 0;

   for (int i = 0; i<bdr_attr_is_ess.Size(); ++i) {
      if (bdr_attr_is_ess[i]) {
         BCManager & bcManager = BCManager::getInstance();
         BCData & bc = bcManager.GetBCInstance(i + 1);
         BCData::getComponents(bc.compID, cmp_row);

         component(i, 0) = cmp_row[0];
         component(i, 1) = cmp_row[1];
         component(i, 2) = cmp_row[2];
      }
   }

   fes->GetEssentialTrueDofs(bdr_attr_is_ess, ess_tdof_list, component);

   if (rhs) {
      for (int i = 0; i < ess_tdof_list.Size(); i++) {
         (*rhs)(ess_tdof_list[i]) = 0.0;
      }
   }
}

// We probably should move these over to their appropriate location in mfem
// library at some point...
void GridFunction::ProjectBdrCoefficient(VectorFunctionRestrictedCoefficient &vfcoeff)

{
   int i, j, fdof, d, ind, vdim;
   Vector val;
   const FiniteElement *fe;
   ElementTransformation *transf;
   Array<int> vdofs;
   const Array<int> &active_attr = vfcoeff.GetActiveAttr();

   this->HostReadWrite();
   vdim = fes->GetVDim();
   // loop over boundary elements
   for (i = 0; i < fes->GetNBE(); i++) {
      // if boundary attribute is 1 (Dirichlet)
      if (active_attr[fes->GetBdrAttribute(i) - 1]) {
         // instantiate a BC object
         BCManager & bcManager = BCManager::getInstance();
         BCData & bc = bcManager.GetBCInstance(fes->GetBdrAttribute(i));

         fe = fes->GetBE(i);
         fdof = fe->GetDof();
         transf = fes->GetBdrElementTransformation(i);
         const IntegrationRule &ir = fe->GetNodes();
         fes->GetBdrElementVDofs(i, vdofs);

         // loop over dofs
         for (j = 0; j < fdof; j++) {
            const IntegrationPoint &ip = ir.IntPoint(j);
            transf->SetIntPoint(&ip);

            vfcoeff.Eval(val, *transf, ip);

            // loop over vector dimensions
            for (d = 0; d < vdim; d++) {
               // check if the vector component (i.e. dof) is not constrained by a
               // partial essential BC
               if (bc.scale[d] > 0.0) {
                  ind = vdofs[fdof * d + j];
                  if ( (ind = vdofs[fdof * d + j]) < 0) {
                     val(d) = -val(d), ind = -1 - ind;
                  }
                  (*this)(ind) = val(d); // placing computed value in grid function
               }
            }
         }
      }
   }
}
