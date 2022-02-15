

#include "mfem.hpp"
#include "BCManager.hpp"
#include <fstream>

using namespace mfem;

void BCManager::updateBCData(mfem::Array<int> & ess_bdr, mfem::Array2D<double> & scale, mfem::Array2D<int> & component)
{
   if (!constant_strain_rate) {
      m_bcInstances.clear();
      ess_bdr = 0;
      scale = 0.0;

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
            scale(bcID - 1, 0) = bc.scale[0];
            scale(bcID - 1, 1) = bc.scale[1];
            scale(bcID - 1, 2) = bc.scale[2];
            ess_bdr[bcID - 1] = 1;
         }
      }

      // The size here is set explicitly
      component.SetSize(ess_bdr.Size(), 3);
      Array<int> cmp_row;
      cmp_row.SetSize(3);

      component = 0;
      cmp_row = 0;

      for (int i = 0; i < ess_bdr.Size(); ++i) {
         if (ess_bdr[i]) {
            BCData& bc = this->GetBCInstance(i + 1);
            BCData::getComponents(bc.compID, cmp_row);

            component(i, 0) = cmp_row[0];
            component(i, 1) = cmp_row[1];
            component(i, 2) = cmp_row[2];
         }
      }
   } 
   else {
      MFEM_ABORT("Trying to update BCs with changing boundaries conditions rather than the chosen constant strain condition");
   }
}

void BCManager::updateBCData(mfem::Array<int> & ess_bdr, mfem::Vector & vgrad, mfem::Array2D<int> & component)
{
   if (constant_strain_rate) {
      m_bcInstances.clear();
      ess_bdr = 0;
      vgrad.HostReadWrite();
      vgrad = 0.0;

      auto ess_vgrad = map_ess_vgrad.find(step)->second;
      auto ess_comp = map_ess_comp.find(step)->second;
      auto ess_id = map_ess_id.find(step)->second;

      for (std::uint32_t i = 0; i < ess_vgrad.size(); ++i) {
         vgrad(i) = ess_vgrad.at(i);
      }

      for (std::uint32_t i = 0; i < ess_id.size(); ++i) {
         // set the boundary condition id based on the attribute id
         int bcID = ess_id[i];

         // instantiate a boundary condition manager instance and
         // create a BCData object
         BCData & bc = this->CreateBCs(bcID);

         // set the velocity component values
         bc.essVel[0] = 0.0;
         bc.essVel[1] = 0.0;
         bc.essVel[2] = 0.0;
         bc.compID = ess_comp[i];

         // set the boundary condition scales
         bc.setScales();

         // set the active boundary attributes
         if (bc.compID != 0) {
            ess_bdr[bcID - 1] = 1;
         }
      }

      // The size here is set explicitly
      component.SetSize(ess_bdr.Size(), 3);
      Array<int> cmp_row;
      cmp_row.SetSize(3);

      component = 0;
      cmp_row = 0;

      for (int i = 0; i < ess_bdr.Size(); ++i) {
         if (ess_bdr[i]) {
            BCData& bc = this->GetBCInstance(i + 1);
            BCData::getComponents(bc.compID, cmp_row);

            component(i, 0) = cmp_row[0];
            component(i, 1) = cmp_row[1];
            component(i, 2) = cmp_row[2];
         }
      }
   } 
   else {
      MFEM_ABORT("Trying to update BCs with constant strain condition rather than the changing boundaries conditions chosen");
   }
}

// set partial dof component list for all essential BCs based on my
// custom BC manager and input, srw.
// We probably should move these over to their appropriate location in mfem
// library at some point...

