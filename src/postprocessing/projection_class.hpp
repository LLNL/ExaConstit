#pragma once

#include "sim_state/simulation_state.hpp"
#include "utilities/rotations.hpp"

#include "mfem.hpp"
#include "ECMech_const.h"
#include "SNLS_linalg.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ProjectionTraits {
/**
 * @brief Model compatibility enumeration for projections
 */
enum class ModelCompatibility {
    ALL_MODELS,      ///< Compatible with all material models
    EXACMECH_ONLY,   ///< Only compatible with ExaCMech models
    UMAT_ONLY        ///< Only compatible with UMAT models
};
}

/**
 * @brief Base projection interface for all projection types in ExaConstit
 * 
 * ProjectionBase provides the fundamental interface that all projection classes
 * must implement. It defines the common operations for converting quadrature
 * function data to grid function data suitable for visualization and analysis.
 * 
 * Key responsibilities:
 * - Execute projection operations for specific material regions
 * - Provide vector dimension information for grid function creation
 * - Support material model compatibility checking
 * - Enable global aggregation capabilities when appropriate
 * 
 * The class uses the template method pattern where derived classes implement
 * specific projection algorithms while the base class handles common interface
 * requirements and material model compatibility checking.
 * 
 * Material model compatibility is enforced through the ProjectionTraits::ModelCompatibility
 * enumeration, allowing projections to specify whether they work with all material
 * models, only ExaCMech models, or only UMAT models.
 * 
 * @ingroup ExaConstit_projections
 */
class ProjectionBase {
public:
    /**
     * @brief Model compatibility type alias
     * 
     * Shorthand for ProjectionTraits::ModelCompatibility enumeration,
     * used throughout projection classes to specify material model
     * compatibility requirements.
     */
    using ptmc = ProjectionTraits::ModelCompatibility;
    /**
     * @brief Material model compatibility for this projection
     * 
     * Specifies which material model types are compatible with this projection.
     * Used during registration to ensure projections are only created for
     * appropriate material regions. Defaults to ALL_MODELS for maximum compatibility.
     */
    const ptmc model = ptmc::ALL_MODELS;
public:
    ProjectionBase() = default;
    ProjectionBase(const ptmc mc) : model(mc) {};
    virtual ~ProjectionBase() = default;
    
    /**
     * @brief Execute the projection for a specific region
     * @param sim_state Reference to simulation state
     * @param grid_function Target grid function to populate
     * @param region Region index
     */
    virtual void Execute(std::shared_ptr<SimulationState> sim_state, 
                         std::shared_ptr<mfem::ParGridFunction> grid_function,
                         mfem::Array<int>& qpts2mesh, 
                         int region) = 0;
    
    /**
     * @brief Get the vector dimension for this projection
     */
    virtual int GetVectorDimension() const = 0;
    
    /**
     * @brief Check if this projection can be aggregated globally across regions
     */
    virtual bool CanAggregateGlobally() const { return false; }
    
    /**
     * @brief Get a display name for this projection
     */
    virtual std::string GetDisplayName() const = 0;
};

//=============================================================================
// GEOMETRY PROJECTIONS
//=============================================================================

/**
 * @brief Base class for geometry-based projections that operate directly on mesh data
 * 
 * GeometryProjection specializes ProjectionBase for projections that compute
 * geometric quantities directly from mesh topology and coordinates, without
 * requiring material-specific quadrature function data.
 * 
 * These projections are inherently region-independent since they depend only
 * on mesh geometry rather than material state. Examples include element
 * centroids, volumes, and geometric quality measures.
 * 
 * Key characteristics:
 * - Region-independent operation (same result regardless of material region)
 * - Direct mesh geometry access through finite element spaces
 * - No dependency on material model type or quadrature function data
 * - Automatic global aggregation support for visualization
 * 
 * Derived classes must implement ProjectGeometry() to perform the actual
 * geometric calculations using MFEM's geometric factors and integration rules.
 * 
 * @ingroup ExaConstit_projections_geometry
 */
class GeometryProjection : public ProjectionBase {
public:

    GeometryProjection() = default;
    ~GeometryProjection() {};
    /**
     * @brief Execute geometry projection (region-independent)
     * 
     * @param sim_state Reference to simulation state (unused for geometry)
     * @param grid_function Target grid function to populate
     * @param qpts2mesh Mapping array (unused for geometry)
     * @param region Region index (unused for geometry)
     * 
     * Executes geometry-based projection by calling the pure virtual
     * ProjectGeometry() method. Geometry projections are region-independent
     * since they depend only on mesh topology and element geometry.
     */
    void Execute([[maybe_unused]] std::shared_ptr<SimulationState> sim_state, 
                 std::shared_ptr<mfem::ParGridFunction> grid_function,
                 [[maybe_unused]] mfem::Array<int>& qpts2mesh, 
                 [[maybe_unused]] int region) override {
        // Geometry projections don't depend on region-specific data
        ProjectGeometry(grid_function);
    }

    /**
     * @brief Check if this projection can be aggregated globally across regions
     */
    virtual bool CanAggregateGlobally() const override { return true; }

protected:
    /**
     * @brief Pure virtual method for geometry calculations
     * 
     * @param grid_function Target grid function to populate with geometry data
     * 
     * Derived classes implement this method to compute geometry-based quantities
     * such as element centroids or volumes. The method has direct access to
     * mesh geometry through the grid function's finite element space.
     */
    virtual void ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function) = 0;
};

/**
 * @brief Element centroid calculation projection
 * 
 * Computes geometric centroids of mesh elements by integrating coordinate
 * positions over element volumes. Provides spatial location information
 * for visualization and spatial analysis of simulation results.
 * 
 * The centroid calculation uses numerical integration over each element
 * with proper volume weighting to handle arbitrary element shapes and
 * polynomial orders. Results are stored as 3D coordinate vectors.
 * 
 * @ingroup ExaConstit_projections_geometry
 */
class CentroidProjection final : public GeometryProjection {
public:

    CentroidProjection() = default;
    ~CentroidProjection() {};

    int GetVectorDimension() const override { return 3; } // Always 3D coordinates
    std::string GetDisplayName() const override { return "Element Centroids"; }
    
protected:
    void ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function) override {

        auto* fes = grid_function->ParFESpace();
        auto* mesh = fes->GetMesh();
        const mfem::FiniteElement& el = *fes->GetFE(0);
        const mfem::IntegrationRule* ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
        
        const int nqpts = ir->GetNPoints();
        const int nelems = fes->GetNE();
        const int vdim = mesh->SpaceDimension();

        const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
            *ir, mfem::GeometricFactors::DETERMINANTS | mfem::GeometricFactors::COORDINATES);

        const double* W = ir->GetWeights().Read();
        const double* const detJ = geom->detJ.Read();
        const auto x_coords = mfem::Reshape(geom->X.Read(), nqpts, vdim, nelems);
        
        double* centroid_data = grid_function->ReadWrite();
        
        // Calculate element centroids
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
            double vol = 0.0;
            for (int iv = 0; iv < vdim; ++iv) {
                centroid_data[ie * vdim + iv] = 0.0;
            }
            
            for (int iq = 0; iq < nqpts; ++iq) {
                const double wt = detJ[ie * nqpts + iq] * W[iq];
                vol += wt;
                for (int iv = 0; iv < vdim; ++iv) {
                    const double coord = x_coords(iq, iv, ie);
                    centroid_data[ie * vdim + iv] += coord * wt;
                }
            }
            
            const double inv_vol = 1.0 / vol;
            for (int iv = 0; iv < vdim; ++iv) {
                centroid_data[ie * vdim + iv] *= inv_vol;
            }
        });
    }
};

/**
 * @brief Element volume projection for mesh analysis and visualization
 * 
 * VolumeProjection computes the volume of each mesh element through numerical
 * integration of the Jacobian determinant over the element domain. This provides
 * essential geometric information for volume averaging operations, mesh quality
 * assessment, and visualization scaling.
 * 
 * The volume calculation uses MFEM's geometric factors to access pre-computed
 * Jacobian determinants at integration points, which are then integrated using
 * the appropriate quadrature weights to obtain accurate element volumes for
 * arbitrary element shapes and polynomial orders.
 * 
 * Key features:
 * - Accurate volume calculation for arbitrary element geometries
 * - Support for high-order finite elements through appropriate integration rules
 * - Essential for volume-weighted averaging operations in post-processing
 * - Useful for mesh quality assessment and adaptive refinement criteria
 * 
 * The computed volumes are stored as scalar values (vector dimension = 1) and
 * can be visualized directly or used internally for volume-weighted calculations
 * in other post-processing operations.
 * 
 * @ingroup ExaConstit_projections_geometry
 */
class VolumeProjection final : public GeometryProjection {
public:

    VolumeProjection() = default;
    ~VolumeProjection() {};

    int GetVectorDimension() const override { return 1; } // Scalar volume
    std::string GetDisplayName() const override { return "Element Volumes"; }
    
protected:
    void ProjectGeometry(std::shared_ptr<mfem::ParGridFunction> grid_function) override {

        auto* fes = grid_function->ParFESpace();
        auto* mesh = fes->GetMesh();
        const mfem::FiniteElement& el = *fes->GetFE(0);
        const mfem::IntegrationRule* ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
        
        const int nqpts = ir->GetNPoints();
        const int nelems = fes->GetNE();

        const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
            *ir, mfem::GeometricFactors::DETERMINANTS);

        const double* const W = ir->GetWeights().Read();
        const double* const detJ = geom->detJ.Read();
        
        double* volume_data = grid_function->ReadWrite();
        
        // Calculate element volumes
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
            double vol = 0.0;
            for (int iq = 0; iq < nqpts; ++iq) {
                vol += detJ[ie * nqpts + iq] * W[iq];
            }
            volume_data[ie] = vol;
        });
    }
};

//=============================================================================
// STRESS-BASED PROJECTIONS
//=============================================================================

/**
 * @brief Base class for stress-based projections using Cauchy stress tensor data
 * 
 * StressProjection provides a specialized interface for projections that operate
 * on stress tensor data from material constitutive models. It handles the common
 * pattern of retrieving stress quadrature functions and delegating to derived
 * classes for specific stress calculations.
 * 
 * The class expects stress data in Voigt notation with 6 components representing
 * the symmetric Cauchy stress tensor: [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂].
 * 
 * Key features:
 * - Automatic stress quadrature function retrieval from simulation state
 * - Support for both element-averaged and quadrature point stress data
 * - Global aggregation capabilities for multi-region stress analysis
 * - Compatible with all material model types that provide stress output
 * 
 * Derived classes implement ProjectStress() to perform specific calculations
 * such as equivalent stress measures, stress invariants, or direct component
 * extraction for visualization and post-processing.
 * 
 * @ingroup ExaConstit_projections_stress
 */
class StressProjection : public ProjectionBase {
public:

    StressProjection() = default;
    ~StressProjection() {};

    void Execute(std::shared_ptr<SimulationState> sim_state, 
                 std::shared_ptr<mfem::ParGridFunction> grid_function,
                 mfem::Array<int>& qpts2mesh, 
                 int region) override {
        // Get stress quadrature function for this region
        auto stress_qf = sim_state->GetQuadratureFunction("cauchy_stress_avg", region);
        if (!stress_qf) return; // Region doesn't have stress data
        
        // Project the stress calculation
        ProjectStress(stress_qf, grid_function, qpts2mesh);
    }

    /**
     * @brief Check if this projection can be aggregated globally across regions
     */
    virtual bool CanAggregateGlobally() const override { return true; }

protected:
    /**
     * @brief Pure virtual method for stress-specific calculations
     * 
     * @param stress_qf Partial quadrature function containing stress tensor data
     * @param grid_function Target grid function to populate with processed stress
     * @param qpts2mesh Mapping from local partial space to global element indices
     * 
     * Derived classes implement this method to perform specific stress calculations
     * such as Von Mises equivalent stress, hydrostatic stress, or direct stress
     * component extraction. The stress data is provided in Voigt notation with
     * 6 components: [S11, S22, S33, S23, S13, S12].
     */
    virtual void ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                               std::shared_ptr<mfem::ParGridFunction> grid_function,
                               mfem::Array<int>& qpts2mesh) = 0;
};

/**
 * @brief Full Cauchy stress tensor projection in Voigt notation
 * 
 * CauchyStressProjection extracts and projects the complete Cauchy stress tensor
 * from material constitutive models for visualization and analysis. The stress
 * components are output in Voigt notation for efficient storage and compatibility
 * with standard post-processing workflows.
 * 
 * The projection preserves all six independent components of the symmetric stress
 * tensor: [σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂], enabling detailed stress field analysis
 * and validation of constitutive model predictions.
 * 
 * Key applications:
 * - Complete stress field visualization in finite element post-processors
 * - Stress validation against analytical solutions or experimental data
 * - Input for stress-based failure criteria and damage models
 * - Principal stress calculations and stress invariant analysis
 * - Multi-axial loading analysis and stress path characterization
 * 
 * The projection is compatible with all material model types that provide
 * Cauchy stress output and supports global aggregation for multi-material
 * simulations with consistent stress field representation.
 * 
 * @ingroup ExaConstit_projections_stress
 */
class CauchyStressProjection final : public StressProjection {
public:

    CauchyStressProjection() = default;
    ~CauchyStressProjection() {};

    int GetVectorDimension() const override { return 6; } // Symmetric tensor in Voigt notation
    std::string GetDisplayName() const override { return "Cauchy Stress"; }
    
protected:
    virtual void ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                               std::shared_ptr<mfem::ParGridFunction> stress_gf,
                               mfem::Array<int>& qpts2mesh) override {

        // Get stress data and compute Von Mises
        const int nelems = stress_gf->ParFESpace()->GetNE();
        const auto part_quad_space = stress_qf->GetPartialSpaceShared();
        const int local_nelems = part_quad_space->GetNE();

        const auto l2g = qpts2mesh.Read();
        const auto stress_data = mfem::Reshape(stress_qf->Read(), 6, local_nelems);
        auto stress_gf_data = mfem::Reshape(stress_gf->Write(), 6, nelems);

        // Compute element-averaged Von Mises stress
        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int ie) {
            const int global_idx = l2g[ie];

            stress_gf_data(0, global_idx) = stress_data(0, ie);
            stress_gf_data(1, global_idx) = stress_data(1, ie);
            stress_gf_data(2, global_idx) = stress_data(2, ie);
            stress_gf_data(3, global_idx) = stress_data(3, ie);
            stress_gf_data(4, global_idx) = stress_data(4, ie);
            stress_gf_data(5, global_idx) = stress_data(5, ie);
        });

    }
};

/**
 * @brief Von Mises equivalent stress projection for failure analysis
 * 
 * VonMisesStressProjection computes the Von Mises equivalent stress from the
 * Cauchy stress tensor, providing a scalar measure of stress intensity commonly
 * used in plasticity theory and failure analysis. The Von Mises stress is
 * calculated as the second invariant of the stress deviator tensor.
 * 
 * Mathematical formulation:
 * σᵥₘ = √(3/2 * sᵢⱼsᵢⱼ) = √(1/2 * [(σ₁₁-σ₂₂)² + (σ₂₂-σ₃₃)² + (σ₃₃-σ₁₁)² + 6(σ₁₂² + σ₁₃² + σ₂₃²)])
 * 
 * Key applications:
 * - Yield criterion evaluation in metal plasticity
 * - Fatigue analysis and life prediction
 * - Stress concentration identification
 * - Material failure assessment
 * - Optimization of component design for stress reduction
 * 
 * The Von Mises stress provides a material-independent measure of stress state
 * that can be directly compared with material yield strength and used in
 * plasticity models regardless of the specific loading configuration.
 * 
 * @ingroup ExaConstit_projections_stress
 */
class VonMisesStressProjection final : public StressProjection {
public:

    VonMisesStressProjection() = default;
    ~VonMisesStressProjection() {};

    int GetVectorDimension() const override { return 1; } // Scalar quantity
    std::string GetDisplayName() const override { return "Von Mises Stress"; }
    
protected:
    virtual void ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                               std::shared_ptr<mfem::ParGridFunction> von_mises,
                               mfem::Array<int>& qpts2mesh) override {
        // Get stress data and compute Von Mises
        const int nelems = von_mises->ParFESpace()->GetNE();
        const auto part_quad_space = stress_qf->GetPartialSpaceShared();
        const int local_nelems = part_quad_space->GetNE();

        const auto l2g = qpts2mesh.Read();
        const auto stress_data = mfem::Reshape(stress_qf->Read(), 6, local_nelems);
        auto von_mises_data = mfem::Reshape(von_mises->Write(), nelems);

        // Compute element-averaged Von Mises stress
        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int ie) {
            const int global_idx = l2g[ie];

            double term1 = stress_data(0, ie) - stress_data(1, ie);
            double term2 = stress_data(1, ie) - stress_data(2, ie);
            double term3 = stress_data(2, ie) - stress_data(0, ie);
            double term4 = stress_data(3, ie) * stress_data(3, ie)
                         + stress_data(4, ie) * stress_data(4, ie)
                         + stress_data(5, ie) * stress_data(5, ie);

            term1 *= term1;
            term2 *= term2;
            term3 *= term3;
            term4 *= 6.0;

            von_mises_data(global_idx) = sqrt(0.5 * (term1 + term2 + term3 + term4));
        });
    }
};

/**
 * @brief Hydrostatic stress projection for pressure analysis
 * 
 * HydrostaticStressProjection computes the hydrostatic (mean normal) stress
 * component from the Cauchy stress tensor. The hydrostatic stress represents
 * the volumetric part of the stress state and is crucial for analyzing
 * pressure-dependent material behavior and volumetric deformation.
 * 
 * Mathematical formulation:
 * σₕ = (σ₁₁ + σ₂₂ + σ₃₃) / 3 = (1/3) * tr(σ)
 * 
 * Key applications:
 * - Pressure-dependent plasticity models (Drucker-Prager, Mohr-Coulomb)
 * - Volumetric strain analysis and compressibility studies
 * - Geomechanics and soil mechanics applications
 * - Phase transformation analysis under pressure
 * - Cavitation and void nucleation studies
 * - Bulk modulus validation and material characterization
 * 
 * The hydrostatic stress is particularly important in materials that exhibit
 * pressure-sensitive behavior, such as geological materials, polymers, and
 * porous media, where the volumetric stress component significantly affects
 * material response.
 * 
 * @ingroup ExaConstit_projections_stress
 */
class HydrostaticStressProjection final : public StressProjection {
public:


    HydrostaticStressProjection() = default;
    ~HydrostaticStressProjection() {};

    int GetVectorDimension() const override { return 1; } // Scalar quantity
    std::string GetDisplayName() const override { return "Hydrostatic Stress"; }
    
protected:
    virtual void ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                               std::shared_ptr<mfem::ParGridFunction> hydro_static,
                               mfem::Array<int>& qpts2mesh) override {
        // Get stress data and compute Von Mises
        const int nelems = hydro_static->ParFESpace()->GetNE();
        const auto part_quad_space = stress_qf->GetPartialSpaceShared();
        const int local_nelems = part_quad_space->GetNE();

        const auto l2g = qpts2mesh.Read();
        const auto stress_data = mfem::Reshape(stress_qf->Read(), 6, local_nelems);
        auto hydro_static_data = mfem::Reshape(hydro_static->Write(), nelems);

        // Compute element-averaged Von Mises stress
        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int ie) {
            const int global_idx = l2g[ie];

            hydro_static_data(global_idx) = ecmech::onethird * (stress_data(0, ie) + stress_data(1, ie) + stress_data(2, ie));
        });
    }
};

//=============================================================================
// STATE VARIABLE PROJECTIONS
//=============================================================================

/**
 * @brief Base class for state variable projections from material constitutive models
 * 
 * StateVariableProjection provides a framework for extracting and projecting
 * specific components from material model state variable arrays. This enables
 * visualization and analysis of internal material state evolution including
 * plastic strains, hardening variables, crystal orientations, and other
 * constitutive model-specific quantities.
 * 
 * The class handles the common pattern of:
 * 1. Retrieving state variable quadrature functions from simulation state
 * 2. Extracting specific components based on index and length specifications
 * 3. Copying data to grid functions with proper element mapping
 * 4. Applying optional post-processing for data conditioning
 * 
 * Key features:
 * - Flexible component extraction with configurable start index and length
 * - Automatic dimension validation against available data and target grid functions
 * - Material model compatibility checking (ALL_MODELS, EXACMECH_ONLY, UMAT_ONLY)
 * - Optional post-processing hook for derived classes (normalization, clamping, etc.)
 * - Support for both scalar and vector state variable components
 * 
 * The state variable array organization depends on the specific material model,
 * but typically follows a consistent layout within each model type. ExaCMech
 * models provide well-defined state variable mappings through SimulationState
 * helper methods.
 * 
 * @ingroup ExaConstit_projections_state_variables
 */
class StateVariableProjection : public ProjectionBase {
public:
    StateVariableProjection(const std::string& state_var_name, 
                           int component_index = 0, 
                           int component_length = -1,
                           ptmc mc = ptmc::ALL_MODELS)
        : ProjectionBase(mc)
        , m_state_var_name(state_var_name)
        , m_component_index(component_index)
        , m_component_length(component_length) {}

    ~StateVariableProjection() {};
    
    void Execute(std::shared_ptr<SimulationState> sim_state, 
                std::shared_ptr<mfem::ParGridFunction> state_gf,
                mfem::Array<int>& qpts2mesh, 
                int region) override {
        // Get state variable quadrature function for this region
        auto state_qf = sim_state->GetQuadratureFunction("state_var_avg", region);
        if (!state_qf) return; // Region doesn't have state variables
        
        // Project the specific component(s)
        const int nelems = state_gf->ParFESpace()->GetNE();
        const auto part_quad_space = state_qf->GetPartialSpaceShared();
        const int local_nelems = part_quad_space->GetNE();
        const int vdim = state_qf->GetVDim();
        m_component_length = (m_component_length == -1) ? vdim : m_component_length;

        if ((m_component_length + m_component_index) > vdim) {
            MFEM_ABORT("StateVariableProjection provided a length and index that pushes us past the state variable length");
        };

        if (m_component_length > state_gf->VectorDim()) {
            MFEM_ABORT("StateVariableProjection provided length is greater than the gridfunction vector length");
        };

        const auto l2g = qpts2mesh.Read();
        const auto state_qf_data = mfem::Reshape(state_qf->Read(), vdim, local_nelems);
        auto state_gf_data = mfem::Reshape(state_gf->Write(), state_gf->VectorDim(), nelems);

        // Compute element-averaged Von Mises stress
        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int ie) {
            const int global_idx = l2g[ie];
            for (int j = 0; j < m_component_length; j++) {
                state_gf_data(j, global_idx) = state_qf_data(j + m_component_index, ie);
            }
        });

        // Apply any post-processing
        PostProcessStateVariable(state_gf, part_quad_space, qpts2mesh);
    }
    
    int GetVectorDimension() const override { return m_component_length; }

protected:
    /**
     * @brief Post-processing hook for derived classes
     * 
     * @param grid_function Target grid function containing extracted state data
     * @param qspace Partial quadrature space for the region
     * @param qpts2mesh Mapping from local to global element indices
     * 
     * Virtual method called after state variable extraction to allow derived
     * classes to perform additional processing such as normalization, coordinate
     * transformations, or value clamping. Default implementation does nothing.
     */
    virtual void PostProcessStateVariable([[maybe_unused]] std::shared_ptr<mfem::ParGridFunction> grid_function,
                                          [[maybe_unused]] std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
                                          [[maybe_unused]] mfem::Array<int>& qpts2mesh) const {};

    /**
     * @brief State variable name for SimulationState lookup
     * 
     * Key used to retrieve the appropriate state variable quadrature function
     * from SimulationState. Must match the naming conventions used by the
     * material model for proper data access.
     */
    std::string m_state_var_name;

    /**
     * @brief Starting index of component within state variable vector
     * 
     * Zero-based index indicating the first component to extract from the
     * state variable vector at each quadrature point. For scalar quantities,
     * this is the direct index. For multi-component data, this is the starting index.
     */
    int m_component_index;

    /**
     * @brief Number of components to extract (-1 for automatic detection)
     * 
     * Specifies how many consecutive components to extract starting from
     * m_component_index. When set to -1, the component length is determined
     * automatically based on available data or target grid function dimensions.
     */
    int m_component_length;
};

/**
 * @brief Complete state variable array projection for debugging and analysis
 * 
 * AllStateVariablesProjection extracts and projects the entire state variable
 * array from material constitutive models, providing comprehensive access to
 * all internal material state information. This projection is primarily used
 * for debugging material model implementations and detailed analysis of
 * constitutive model behavior.
 * 
 * Key characteristics:
 * - Projects all available state variables without filtering
 * - Vector dimension determined automatically from material model
 * - Useful for debugging constitutive model implementations
 * - Enables detailed analysis of material state evolution
 * - Cannot be aggregated globally due to variable interpretation complexity
 * 
 * The interpretation of state variable components depends entirely on the
 * specific material model implementation:
 * - ExaCMech: Includes plastic strains, hardening variables, orientations, etc.
 * - UMAT: User-defined state variables with model-specific meanings
 * - Other models: Model-specific internal state representations
 * 
 * This projection is typically used during material model development,
 * validation, and debugging rather than for routine post-processing and
 * visualization of simulation results.
 * 
 * @note The output requires detailed knowledge of the material model's
 *       state variable organization for proper interpretation.
 * 
 * @ingroup ExaConstit_projections_state_variables
 */
class AllStateVariablesProjection final : public StateVariableProjection {
public:
    AllStateVariablesProjection() : StateVariableProjection("all_state_vars", 0, -1) {}

    AllStateVariablesProjection([[maybe_unused]] const std::string& state_var_name,
                                [[maybe_unused]] int component_index,
                                [[maybe_unused]] int component_length)
                                : StateVariableProjection("all_state_vars", 0, -1) {}

    ~AllStateVariablesProjection() {};

    std::string GetDisplayName() const override { return "All State Variables"; }
    virtual bool CanAggregateGlobally() const override { return false; }
};

/**
 * @brief Equivalent plastic strain rate projection for ECMech models
 * 
 * Projects the equivalent plastic strain rate (dpeff) state variable from
 * ExaCMech constitutive models. This quantity represents the rate of
 * plastic strain accumulation and is essential for rate-dependent analysis.
 * 
 * Only compatible with ExaCMech material models that provide the
 * "eq_pl_strain_rate" state variable. Non-ECMech regions are handled
 * with dummy projections that produce no output.
 * 
 * Post-processing applies rate scaling and unit conversions as needed
 * for consistent output formatting across different material models.
 * 
 * @ingroup ExaConstit_projections_state_variables
 */
class DpEffProjection final : public StateVariableProjection {
public:
    DpEffProjection([[maybe_unused]] const std::string& state_var_name,
                    int component_index,
                    [[maybe_unused]] int component_length)
                    : StateVariableProjection("eq_pl_strain_rate", component_index, 1, ptmc::EXACMECH_ONLY) {}
    ~DpEffProjection() = default;

    std::string GetDisplayName() const override { return "Equivalent Plastic Strain Rate"; }
    virtual bool CanAggregateGlobally() const override { return true; }

protected:
    virtual
    void PostProcessStateVariable(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                  std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
                                  mfem::Array<int>& qpts2mesh) const override {
        auto data = grid_function->Write();
        const auto l2g = qpts2mesh.Read();
        const int local_nelems = qspace->GetNE();

        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int i) {
            const int ie = l2g[i];
            data[ie] = fmax(data[ie], 0.0);
        });
    }

};

/**
 * @brief Crystal orientation projection with quaternion normalization
 * 
 * Projects crystal orientation quaternions from ExaCMech models with
 * automatic normalization to ensure unit quaternions. Handles quaternion
 * data extraction and post-processing for texture analysis applications.
 * 
 * The post-processing step normalizes quaternions to correct for numerical
 * drift during simulation and ensure valid rotation representations.
 * Output quaternions follow the convention [q0, q1, q2, q3] where q0
 * is the scalar component.
 * 
 * Only compatible with ExaCMech material models that provide quaternion
 * orientation data in their state variable arrays.
 * 
 * @ingroup ExaConstit_projections_crystal_plasticity
 */
class XtalOrientationProjection final : public StateVariableProjection {
public:
    XtalOrientationProjection([[maybe_unused]] const std::string& state_var_name,
                              int component_index,
                              [[maybe_unused]] int component_length)
                              : StateVariableProjection("quats", component_index, 4, ptmc::EXACMECH_ONLY) {}
    ~XtalOrientationProjection() = default;

    std::string GetDisplayName() const override { return "Crystal Orientations"; }
    virtual bool CanAggregateGlobally() const override { return true; }

protected:
    virtual
    void PostProcessStateVariable(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                  std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
                                  mfem::Array<int>& qpts2mesh) const override {
        const int nelems = grid_function->ParFESpace()->GetNE();
        auto ori = mfem::Reshape(grid_function->Write(), grid_function->VectorDim(), nelems);
        const auto l2g = qpts2mesh.Read();
        const int local_nelems = qspace->GetNE();

        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int i) {
            const int ie = l2g[i];
            const double inv_norm = 1.0 / (ori(0, ie) * ori(0, ie)
                                         + ori(1, ie) * ori(1, ie)
                                         + ori(2, ie) * ori(2, ie)
                                         + ori(3, ie) * ori(3, ie));
            
            ori(0, ie) = ori(0, ie) * inv_norm;
            ori(1, ie) = ori(1, ie) * inv_norm;
            ori(2, ie) = ori(2, ie) * inv_norm;
            ori(3, ie) = ori(3, ie) * inv_norm;
        });
    }
};

/**
 * @brief Elastic strain tensor projection for ExaCMech models
 * 
 * Projects elastic strain tensor components from ExaCMech state variables
 * with coordinate system transformations. Performs conversion from lattice
 * coordinates to sample coordinates using crystal orientation data.
 * 
 * The projection involves:
 * 1. Extraction of deviatoric elastic strain and relative volume
 * 2. Reconstruction of full elastic strain tensor in lattice coordinates
 * 3. Rotation to sample coordinates using quaternion orientations
 * 4. Output in Voigt notation: [E11, E22, E33, E23, E13, E12]
 * 
 * Only compatible with ExaCMech models that provide elastic strain
 * state variables and crystal orientation data.
 * 
 * @ingroup ExaConstit_projections_strain
 */
class ElasticStrainProjection final : public StateVariableProjection {
public:
    ElasticStrainProjection([[maybe_unused]] const std::string& state_var_name,
                            int component_index,
                            [[maybe_unused]] int component_length)
                            : StateVariableProjection("elastic_strain", component_index, 6, ptmc::EXACMECH_ONLY) {}
    /**
     * @brief Execute elastic strain projection with coordinate transformation
     * 
     * @param sim_state Reference to simulation state for data access
     * @param elastic_strain_gf Target grid function for elastic strain output
     * @param qpts2mesh Mapping from local to global element indices
     * @param region Material region identifier
     * 
     * Overrides the base StateVariableProjection::Execute() method to implement
     * specialized processing for elastic strain data. The method:
     * 
     * 1. Retrieves state variables including elastic strain, orientations, and volume
     * 2. Reconstructs full 3x3 elastic strain tensor from deviatoric components
     * 3. Applies coordinate transformation from crystal to sample coordinates
     * 4. Outputs strain components in Voigt notation
     * 
     * The coordinate transformation accounts for crystal orientation evolution
     * and ensures strain components are expressed in the global reference frame
     * for visualization and post-processing compatibility.
     * 
     * @note This method bypasses the standard StateVariableProjection data flow
     *       due to the complex coordinate transformations required.
     */
    void Execute(std::shared_ptr<SimulationState> sim_state, 
                std::shared_ptr<mfem::ParGridFunction> elastic_strain_gf,
                mfem::Array<int>& qpts2mesh, 
                int region) override {

        // Get state variable quadrature function for this region
        auto state_qf = sim_state->GetQuadratureFunction("state_var_avg", region);
        if (!state_qf) return; // Region doesn't have state variables

        const int nelems = elastic_strain_gf->ParFESpace()->GetNE();
        const auto part_quad_space = state_qf->GetPartialSpaceShared();

        const auto l2g = qpts2mesh.Read();
        const int local_nelems = part_quad_space->GetNE();
        const int vdim = state_qf->GetVDim();
        const int gf_vdim = elastic_strain_gf->VectorDim();
        m_component_length = (m_component_length == -1) ? vdim : m_component_length;

        if ((m_component_length + m_component_index) > vdim) {
            MFEM_ABORT("ElasticStrainProjection provided a length and index that pushes us past the state variable length");
        };

        if (m_component_length > elastic_strain_gf->VectorDim()) {
            MFEM_ABORT("ElasticStrainProjection provided length is greater than the gridfunction vector length");
        };

        const int estrain_ind = sim_state->GetQuadratureFunctionStatePair("elastic_strain", region).first;
        const int quats_ind = sim_state->GetQuadratureFunctionStatePair("quats", region).first;
        const int rel_vol_ind = sim_state->GetQuadratureFunctionStatePair("relative_volume", region).first;

        auto state_vars = sim_state->GetQuadratureFunction("state_var_end", region)->Read();
        auto strain = mfem::Reshape(elastic_strain_gf->Write(), gf_vdim, nelems);
        
        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int ie) {
            const int global_idx = l2g[ie];

            const auto strain_lat = &state_vars[ie * vdim + estrain_ind];
            const auto quats = &state_vars[ie * vdim + quats_ind];
            const auto rel_vol = state_vars[ie * vdim + rel_vol_ind];
            {
                double strainm[3 * 3] = {};
                double* strain_m[3] = {&strainm[0], &strainm[3], &strainm[6]};
                const double t1 = ecmech::sqr2i * strain_lat[0];
                const double t2 = ecmech::sqr6i * strain_lat[1];
                //
                // Volume strain is ln(V^e_mean) term aka ln(relative volume)
                // Our plastic deformation has a det(1) aka no change in volume change
                const double elas_vol_strain = log(rel_vol);
                // We output elastic strain formulation such that the relationship
                // between V^e and \varepsilon is just V^e = I + \varepsilon
                strain_m[0][0] = (t1 - t2) + elas_vol_strain; // 11
                strain_m[1][1] = (-t1 - t2) + elas_vol_strain ; // 22
                strain_m[2][2] = ecmech::sqr2b3 * strain_lat[1] + elas_vol_strain; // 33
                strain_m[1][2] = ecmech::sqr2i * strain_lat[4]; // 23
                strain_m[2][0] = ecmech::sqr2i * strain_lat[3]; // 31
                strain_m[0][1] = ecmech::sqr2i * strain_lat[2]; // 12

                strain_m[2][1] = strain_m[1][2];
                strain_m[0][2] = strain_m[2][0];
                strain_m[1][0] = strain_m[0][1];

                double rmat[3 * 3] = {};
                double strain_samp[3 * 3] = {};            

                quat2rmat(quats, rmat);
                snls::linalg::rotMatrix<3, false>(strainm, rmat, strain_samp);

                strain_m[0] = &strain_samp[0];
                strain_m[1] = &strain_samp[3];
                strain_m[2] = &strain_samp[6];

                strain(0, global_idx) = strain_m[0][0];
                strain(1, global_idx) = strain_m[1][1];
                strain(2, global_idx) = strain_m[2][2];
                strain(3, global_idx) = strain_m[1][2];
                strain(4, global_idx) = strain_m[0][2];
                strain(5, global_idx) = strain_m[0][1];
            }
        });
    }

    std::string GetDisplayName() const override { return "Elastic Strains"; }
    virtual bool CanAggregateGlobally() const override { return true; }
};

/**
 * @brief Hardness projection with non-negative value enforcement
 * 
 * Projects hardness values from ExaCMech state variables with post-processing
 * to ensure non-negative values. Useful for visualizing material hardening
 * evolution in crystal plasticity simulations.
 * 
 * Post-processing applies fmax(value, 0.0) to prevent negative hardness
 * values that may arise from numerical issues or specific hardening models.
 * 
 * Only compatible with ExaCMech material models that include hardness
 * in their state variable definitions.
 * 
 * @ingroup ExaConstit_projections_material_properties
 */
class HardnessProjection final : public StateVariableProjection {
public:
    HardnessProjection([[maybe_unused]] const std::string& state_var_name,
                       int component_index,
                       int component_length)
                       : StateVariableProjection("hardness", component_index, component_length, ptmc::EXACMECH_ONLY) {}
    
    std::string GetDisplayName() const override { return "Hardness"; }

    virtual bool CanAggregateGlobally() const override { return false; }
    
protected:

    virtual
    void PostProcessStateVariable(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                  std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace,
                                  mfem::Array<int>& qpts2mesh) const override {
        // Ensure non-negative values
        double* data = grid_function->ReadWrite();
        const auto l2g = qpts2mesh.Read();
        const int local_nelems = qspace->GetNE();

        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int i) {
            const int ie = l2g[i];
            data[ie] = fmax(data[ie], 0.0);
        });
    }
};

/**
 * @brief Shear rate projection for crystal plasticity analysis
 * 
 * Projects macroscopic shear rate data from ExaCMech state variables.
 * Provides access to overall plastic deformation rates for rate-dependent
 * analysis and comparison with experimental strain rate measurements.
 * 
 * This projection extracts aggregate shear rate information rather than
 * individual slip system rates, making it suitable for macroscopic
 * deformation analysis and rate sensitivity studies.
 * 
 * Only compatible with ExaCMech material models that compute and store
 * macroscopic shear rate data.
 * 
 * @ingroup ExaConstit_projections_crystal_plasticity
 */
class ShearingRateProjection final : public StateVariableProjection {
public:
    ShearingRateProjection([[maybe_unused]] const std::string& state_var_name,
                           int component_index,
                           int component_length)
                           : StateVariableProjection("shear_rate", component_index, component_length, ptmc::EXACMECH_ONLY) {}

    std::string GetDisplayName() const override { return "Shearing Rate"; }
    virtual bool CanAggregateGlobally() const override { return false; }
};