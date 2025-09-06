// Example: How to use the simplified projection system
// and how easy it is to add new projections

#include "simplified_projections.hpp"

namespace ExaConstit {

//=============================================================================
// EXAMPLE: Adding a new stress-based projection
//=============================================================================

/**
 * @brief Example: Triaxiality stress projection
 * Shows how easy it is to add a new stress-based calculation
 */
class TriaxialityProjection : public StressProjection {
public:
    int GetVectorDimension() const override { return 1; } // Scalar quantity
    std::string GetDisplayName() const override { return "Stress Triaxiality"; }
    
protected:
    void ProjectStress(const mfem::PartialQuadratureFunction& stress_qf,
                      mfem::ParGridFunction& grid_function) override {
        const int nelems = grid_function.ParFESpace()->GetNE();
        const int nqpts = stress_qf.GetNQPTs();
        
        mfem::Vector triax_values(nelems);
        double* triax_data = triax_values.ReadWrite();
        const double* stress_data = stress_qf.Read();
        
        // Compute element-averaged triaxiality (hydrostatic / von_mises)
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
            double triax_avg = 0.0;
            
            for (int iq = 0; iq < nqpts; ++iq) {
                const int idx = ie * nqpts * 6 + iq * 6;
                
                // Extract stress components
                const double sxx = stress_data[idx + 0];
                const double syy = stress_data[idx + 1];
                const double szz = stress_data[idx + 2];
                const double sxy = stress_data[idx + 3];
                const double sxz = stress_data[idx + 4];
                const double syz = stress_data[idx + 5];
                
                // Hydrostatic stress
                const double hydro = (sxx + syy + szz) / 3.0;
                
                // Von Mises stress
                const double dev_xx = sxx - hydro;
                const double dev_yy = syy - hydro;
                const double dev_zz = szz - hydro;
                const double vm = sqrt(1.5 * (dev_xx*dev_xx + dev_yy*dev_yy + dev_zz*dev_zz +
                                             2.0*(sxy*sxy + sxz*sxz + syz*syz)));
                
                // Triaxiality = hydrostatic / von_mises
                const double triax = (vm > 1e-12) ? hydro / vm : 0.0;
                triax_avg += triax;
            }
            
            triax_data[ie] = triax_avg / nqpts;
        });
        
        grid_function = triax_values;
    }
};

//=============================================================================
// EXAMPLE: Adding a new state variable projection
//=============================================================================

/**
 * @brief Example: Custom temperature projection from state variables
 */
class TemperatureProjection : public StateVariableProjection {
public:
    TemperatureProjection(int index) : StateVariableProjection("temperature", index, 1) {}
    
    std::string GetDisplayName() const override { return "Temperature"; }
    bool CanAggregateGlobally() const override { return true; }
    
protected:
    void PostProcessStateVariable(mfem::ParGridFunction& grid_function) override {
        // Convert from Kelvin to Celsius if needed
        double* data = grid_function.ReadWrite();
        const int size = grid_function.Size();
        
        mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) {
            data[i] = data[i] - 273.15; // K to C
        });
    }
};

//=============================================================================
// EXAMPLE: Adding a new geometry projection
//=============================================================================

/**
 * @brief Example: Element surface area projection
 */
class SurfaceAreaProjection : public GeometryProjection {
public:
    int GetVectorDimension() const override { return 1; } // Scalar surface area
    std::string GetDisplayName() const override { return "Element Surface Area"; }
    
protected:
    void ProjectGeometry(mfem::ParFiniteElementSpace* fes, 
                        mfem::ParGridFunction& grid_function) override {
        auto* mesh = fes->GetMesh();
        const int nelems = fes->GetNE();
        
        double* area_data = grid_function.ReadWrite();
        
        // Calculate surface area for each element
        // This is a simplified example - actual implementation would depend on element type
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
            // Placeholder: compute actual surface area based on element geometry
            area_data[ie] = 1.0; // Replace with real calculation
        });
    }
};

//=============================================================================
// USAGE EXAMPLES
//=============================================================================

void ExampleUsage() {
    // Assume we have a simulation state
    SimulationState sim_state; // Your existing simulation state
    
    // Create the simplified post-processing driver
    SimplifiedPostProcessingDriver pp_driver(sim_state);
    
    // Set up for 3 regions
    pp_driver.SetNumRegions(3);
    
    // Enable basic projections for all regions
    pp_driver.EnableProjection("stress", true);        // Cauchy stress tensor
    pp_driver.EnableProjection("von_mises", true);     // Von Mises stress
    pp_driver.EnableProjection("centroid", true);      // Element centroids
    pp_driver.EnableProjection("volume", true);        // Element volumes
    
    // Enable state variable projections (only for regions that have ExaCMech)
    pp_driver.EnableProjection("dpeff", 0, true);      // Only region 0
    pp_driver.EnableProjection("orientations", 0, true); // Only region 0
    
    // Register and enable custom projections
    pp_driver.RegisterCustomProjection("triaxiality", std::make_unique<TriaxialityProjection>());
    pp_driver.RegisterCustomProjection("temperature", std::make_unique<TemperatureProjection>(10)); // Index 10 in state vars
    pp_driver.RegisterCustomProjection("surface_area", std::make_unique<SurfaceAreaProjection>());
    
    pp_driver.EnableProjection("triaxiality", true);
    pp_driver.EnableProjection("temperature", 1, true);  // Only region 1 has temperature
    pp_driver.EnableProjection("surface_area", true);
    
    // Execute projections for a time step
    std::unordered_map<std::string, std::unique_ptr<mfem::ParGridFunction>> grid_functions;
    
    for (int region = 0; region < 3; ++region) {
        pp_driver.ExecuteProjections(region, grid_functions);
    }
    
    // Grid functions are now populated and ready for visualization or analysis
    // Access them via grid_functions["projection_name_region_X"]
    
    // List all available projections
    auto available_projections = pp_driver.GetAvailableProjections();
    for (const auto& proj_name : available_projections) {
        std::cout << "Available projection: " << proj_name << std::endl;
    }
}

//=============================================================================
// COMPARISON: OLD vs NEW SYSTEM
//=============================================================================

/*
OLD SYSTEM PROBLEMS:
====================
1. Complex template hierarchy with ProjectionTrait<Derived>
2. Multiple registration methods: RegisterSimpleProjection, RegisterSpecialProjection, RegisterGeometryProjection
3. Complicated trait detection and SFINAE
4. Hard to understand inheritance and static dispatch
5. Difficult to extend - need to understand the trait system
6. Tightly coupled to specific PostProcessingDriver implementation

NEW SYSTEM BENEFITS:
====================
1. Simple base class (ProjectionBase) with virtual methods
2. Three clear categories: GeometryProjection, StressProjection, StateVariableProjection
3. Easy to extend - just inherit from the appropriate base class
4. Clean runtime polymorphism - no template metaprogramming required
5. Simple registration: just RegisterProjection(name, std::make_unique<YourProjection>())
6. Easy to test individual projections
7. Self-contained projections that know their requirements
8. Clear separation of concerns

ADDING A NEW PROJECTION:
========================
OLD: 
- Create trait class inheriting from ProjectionTrait<T>
- Implement static methods: GetModelCompatibility, SelectComponent, PostProcess, CanAggregateGlobally
- Understand which registration method to use
- Deal with template instantiation and compatibility checking

NEW:
- Inherit from appropriate base (GeometryProjection, StressProjection, or StateVariableProjection)
- Implement virtual methods (much clearer interface)
- Register with a single method call
- Done!

EXTENDING FOR NEW CATEGORIES:
=============================
If you need a new category of projections (e.g., time-derivative based):

class TimeDerivativeProjection : public ProjectionBase {
public:
    void Execute(SimulationState& sim_state, 
                mfem::ParGridFunction& grid_function, 
                int region) override {
        // Get current and previous state, compute derivatives
        auto current_qf = sim_state.GetQuadratureFunction(m_var_name, region);
        auto previous_qf = sim_state.GetPreviousQuadratureFunction(m_var_name, region);
        
        // Compute time derivative and project
        ComputeTimeDerivative(*current_qf, *previous_qf, grid_function);
    }
    
protected:
    virtual void ComputeTimeDerivative(const mfem::PartialQuadratureFunction& current,
                                      const mfem::PartialQuadratureFunction& previous,
                                      mfem::ParGridFunction& grid_function) = 0;
    std::string m_var_name;
};

Then implement specific time derivative projections by inheriting from this base.
*/

} // namespace ExaConstit