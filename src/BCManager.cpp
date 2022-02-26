

#include "mfem.hpp"
#include "BCManager.hpp"
#include <fstream>

using namespace mfem;


void BCManager::updateBCData(std::unordered_map<std::string, mfem::Array<int>> & ess_bdr, 
                             mfem::Array2D<double> & scale,
                             mfem::Vector & vgrad, 
                             std::unordered_map<std::string, mfem::Array2D<int>> & component)
{
   ess_bdr["total"] = 0;
   scale = 0.0;

   auto ess_comp = map_ess_comp["total"].find(step)->second;
   auto ess_id = map_ess_id["total"].find(step)->second;

   Array<int> cmp_row;
   cmp_row.SetSize(3);

   component["total"] = 0;
   cmp_row = 0;

   for (std::uint32_t i = 0; i < ess_id.size(); ++i) {
      // set the active boundary attributes
      if (ess_comp[i] != 0) {
         ess_bdr["total"][ess_id[i] - 1] = 1;
         BCData::getComponents(std::abs(ess_comp[i]), cmp_row);

         component["total"](i, 0) = cmp_row[0];
         component["total"](i, 1) = cmp_row[1];
         component["total"](i, 2) = cmp_row[2];
      }
   }

   updateBCData(ess_bdr["ess_vel"], scale, component["ess_vel"]);
   updateBCData(ess_bdr["ess_vgrad"], vgrad, component["ess_vgrad"]);
}

void BCManager::updateBCData(mfem::Array<int> & ess_bdr, mfem::Array2D<double> & scale, mfem::Array2D<int> & component)
{
   m_bcInstances.clear();
   ess_bdr = 0;
   scale = 0.0;

   // The size here is set explicitly
   component.SetSize(ess_bdr.Size(), 3);
   Array<int> cmp_row;
   cmp_row.SetSize(3);

   component = 0;
   cmp_row = 0;

   if (map_ess_vel.find(step) == map_ess_vel.end())
   {
      return;
   }

   auto ess_vel = map_ess_vel.find(step)->second;
   auto ess_comp = map_ess_comp["ess_vel"].find(step)->second;
   auto ess_id = map_ess_id["ess_vel"].find(step)->second;

   for (std::uint32_t i = 0; i < ess_id.size(); ++i) {
      // set the active boundary attributes
      if (ess_comp[i] != 0) {
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

         scale(bcID - 1, 0) = bc.scale[0];
         scale(bcID - 1, 1) = bc.scale[1];
         scale(bcID - 1, 2) = bc.scale[2];
         ess_bdr[bcID - 1] = 1;
      }
   }

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

void BCManager::updateBCData(mfem::Array<int> & ess_bdr, mfem::Vector & vgrad, mfem::Array2D<int> & component)
{
   ess_bdr = 0;
   vgrad.HostReadWrite();
   vgrad = 0.0;

   // The size here is set explicitly
   component.SetSize(ess_bdr.Size(), 3);
   Array<int> cmp_row;
   cmp_row.SetSize(3);

   component = 0;
   cmp_row = 0;

   if (map_ess_vgrad.find(step) == map_ess_vgrad.end())
   {
      return;
   }

   auto ess_vgrad = map_ess_vgrad.find(step)->second;
   auto ess_comp = map_ess_comp["ess_vgrad"].find(step)->second;
   auto ess_id = map_ess_id["ess_vgrad"].find(step)->second;

   for (std::uint32_t i = 0; i < ess_vgrad.size(); ++i) {
      vgrad(i) = ess_vgrad.at(i);
   }

   for (std::uint32_t i = 0; i < ess_id.size(); ++i) {
      // set the active boundary attributes
      if (ess_comp[i] != 0) {
         ess_bdr[ess_id[i] - 1] = 1;
         BCData::getComponents(ess_comp[i], cmp_row);
         component(i, 0) = cmp_row[0];
         component(i, 1) = cmp_row[1];
         component(i, 2) = cmp_row[2];
      }
   }
}

