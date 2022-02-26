
#ifndef BCMANAGER
#define BCMANAGER

#include "BCData.hpp"
#include "option_parser.hpp"

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
                const std::unordered_map<int, std::vector<double>> &ess_vgrad,
                const map_of_imap &ess_comp,
                const map_of_imap &ess_id) {
         std::call_once(init_flag, [&](){
            updateStep = uStep;
            map_ess_vel = ess_vel;
            map_ess_vgrad = ess_vgrad;
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

      void updateBCData(std::unordered_map<std::string, mfem::Array<int>> & ess_bdr, 
                        mfem::Array2D<double> & scale,
                        mfem::Vector & vgrad, 
                        std::unordered_map<std::string, mfem::Array2D<int>> & component);

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

      void updateBCData(mfem::Array<int> & ess_bdr, mfem::Vector & vgrad, mfem::Array2D<int> & component);
      void updateBCData(mfem::Array<int> & ess_bdr, mfem::Array2D<double> & scale, mfem::Array2D<int> & component);

      std::once_flag init_flag;
      int step = 0;
      std::unordered_map<int, BCData> m_bcInstances;
      std::vector<int> updateStep;
      std::unordered_map<int, std::vector<double>> map_ess_vel;
      std::unordered_map<int, std::vector<double>> map_ess_vgrad;
      map_of_imap map_ess_comp;
      map_of_imap map_ess_id;
};

#endif
