#include "models/mechanics_model.hpp"
#include "utilities/mechanics_log.hpp"

#include "mfem.hpp"
#include "mfem/general/forall.hpp"

#include <math.h> // log
#include <algorithm>
#include <iostream> // cerr
#include "RAJA/RAJA.hpp"

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
