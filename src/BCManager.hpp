
#ifndef BCMANAGER
#define BCMANAGER

#include "BCData.hpp"

// C/C++ includes
#include <unordered_map> // for std::unordered_map
#include <vector>
#include <algorithm>
#include <mutex>


class BCManager
{
   public:
      static BCManager & getInstance()
      {
         static BCManager bcManager;
         return bcManager;
      }

      void init(const std::vector<int> &uStep,
                const std::unordered_map<int, std::vector<double>> &ess_vel,
                const std::unordered_map<int, std::vector<int>> &ess_comp,
                const std::unordered_map<int, std::vector<int>> &ess_id) {
         std::call_once(init_flag, [&](){
            updateStep = uStep;
            map_ess_vel = ess_vel;
            map_ess_comp = ess_comp;
            map_ess_id = ess_id;
         });
      }

      BCData & GetBCInstance(int bcID)
      {
         return m_bcInstances.find(bcID)->second;
      }

      const BCData & GetBCInstance(int bcID) const
      {
         return m_bcInstances.find(bcID)->second;
      }

      BCData & CreateBCs(int bcID)
      {
         return m_bcInstances[bcID];
      }

      std::unordered_map<int, BCData>&GetBCInstances()
      {
         return m_bcInstances;
      }

      void updateBCData(mfem::Array<int> & ess_bdr, mfem::Array2D<double> & scale, mfem::Array2D<int> & component);

      bool getUpdateStep(int step_)
      {
         if(std::find(updateStep.begin(), updateStep.end(), step_) != updateStep.end()) {
            step = step_;
            return true;
         }
         else {
            return false;
         }
      }
   private:
      BCManager() {}
      BCManager(const BCManager&) = delete;
      BCManager& operator=(const BCManager &) = delete;
      BCManager(BCManager &&) = delete;
      BCManager & operator=(BCManager &&) = delete;

      std::once_flag init_flag;
      int step = 0;
      std::unordered_map<int, BCData> m_bcInstances;
      std::vector<int> updateStep;
      std::unordered_map<int, std::vector<double>> map_ess_vel;
      std::unordered_map<int, std::vector<int>> map_ess_comp;
      std::unordered_map<int, std::vector<int>> map_ess_id;
};

#endif
