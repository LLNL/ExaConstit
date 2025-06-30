#pragma once

#include "mfem.hpp"
#include "sim_state/simulation_state.hpp"

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
    ALL_MODELS,      // Compatible with all material models
    EXACMECH_ONLY,   // Only compatible with ExaCMech models
    UMAT_ONLY        // Only compatible with UMAT models
};
}

__ecmech_hdev__
inline
void 
quat2rmat_v2(const double* const quat,
          double* const rmats) 
{
    double qbar =  quat[0] * quat[0] - (quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

    double* rmat[3] = {&rmats[0], &rmats[3], &rmats[6]};

    rmat[0][0] = qbar + 2.0 * quat[1] * quat[1];
    rmat[1][0] = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
    rmat[2][0] = 2.0 * (quat[1] * quat[3] - quat[0] * quat[2]);

    rmat[0][1] = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
    rmat[1][1] = qbar + 2.0 * quat[2] * quat[2];
    rmat[2][1] = 2.0 * (quat[2] * quat[3] + quat[0] * quat[1]);

    rmat[0][2] = 2.0 * (quat[1] * quat[3] + quat[0] * quat[2]);
    rmat[1][2] = 2.0 * (quat[2] * quat[3] - quat[0] * quat[1]);
    rmat[2][2] = qbar + 2.0 * quat[3] * quat[3];
}

/**
 * @brief Base projection interface - all projections derive from this
 */
class ProjectionBase {
public:
    using ptmc = ProjectionTraits::ModelCompatibility;
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
    virtual void Execute(SimulationState& sim_state, 
                        std::shared_ptr<mfem::ParGridFunction> grid_function, 
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
 * @brief Base class for geometry-based projections that work directly with mesh
 */
class GeometryProjection : public ProjectionBase {
public:

    GeometryProjection() = default;
    ~GeometryProjection() {};

    void Execute(SimulationState& sim_state, 
                std::shared_ptr<mfem::ParGridFunction> grid_function, 
                int region) override {
        // Geometry projections don't depend on region-specific data
        auto fes = sim_state.GetMeshParFiniteElementSpace();
        auto partial_qspace = sim_state.GetQuadratureFunction("cauchy_stress_avg", region)->GetPartialSpaceShared();

        ProjectGeometry(fes, grid_function, partial_qspace);
    }

    /**
     * @brief Check if this projection can be aggregated globally across regions
     */
    virtual bool CanAggregateGlobally() const override { return true; }

protected:
    /**
     * @brief Pure virtual method for specific geometry calculations
     */
    virtual void ProjectGeometry(std::shared_ptr<mfem::ParFiniteElementSpace> fes, 
                                 std::shared_ptr<mfem::ParGridFunction> grid_function,
                                 std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace) = 0;
};

/**
 * @brief Element centroid projection
 */
class CentroidProjection final : public GeometryProjection {
public:

    CentroidProjection() = default;
    ~CentroidProjection() {};

    int GetVectorDimension() const override { return 3; } // Always 3D coordinates
    std::string GetDisplayName() const override { return "Element Centroids"; }
    
protected:
    void ProjectGeometry(std::shared_ptr<mfem::ParFiniteElementSpace> fes, 
                         std::shared_ptr<mfem::ParGridFunction> grid_function,
                         std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace) override {
        auto* mesh = fes->GetMesh();
        const mfem::FiniteElement& el = *fes->GetFE(0);
        const mfem::IntegrationRule* ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
        
        const int nqpts = ir->GetNPoints();
        const int nelems = fes->GetNE();
        const int vdim = mesh->SpaceDimension();

        const int local_nelems = qspace->GetNE();
        const auto l2g = qspace->getLocal2Global().Read();

        const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
            *ir, mfem::GeometricFactors::DETERMINANTS | mfem::GeometricFactors::COORDINATES);

        const double* W = ir->GetWeights().Read();
        const double* const detJ = geom->detJ.Read();
        const auto x_coords = mfem::Reshape(geom->X.Read(), nqpts, vdim, nelems);
        
        double* centroid_data = grid_function->ReadWrite();
        
        // Calculate element centroids
        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int i) {
            const int ie = l2g[i];
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
 * @brief Element volume projection
 */
class VolumeProjection final : public GeometryProjection {
public:

    VolumeProjection() = default;
    ~VolumeProjection() {};

    int GetVectorDimension() const override { return 1; } // Scalar volume
    std::string GetDisplayName() const override { return "Element Volumes"; }
    
protected:
    void ProjectGeometry(std::shared_ptr<mfem::ParFiniteElementSpace> fes, 
                        std::shared_ptr<mfem::ParGridFunction> grid_function, std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace) override {
        auto* mesh = fes->GetMesh();
        const mfem::FiniteElement& el = *fes->GetFE(0);
        const mfem::IntegrationRule* ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
        
        const int nqpts = ir->GetNPoints();
        const int nelems = fes->GetNE();

        const int local_nelems = qspace->GetNE();
        const auto l2g = qspace->getLocal2Global().Read();

        const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
            *ir, mfem::GeometricFactors::DETERMINANTS);

        const double* const W = ir->GetWeights().Read();
        const double* const detJ = geom->detJ.Read();
        
        double* volume_data = grid_function->ReadWrite();
        
        // Calculate element volumes
        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int i) {
            const int ie = l2g[i];
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
 * @brief Base class for stress-based projections
 */
class StressProjection : public ProjectionBase {
public:

    StressProjection() = default;
    ~StressProjection() {};

    void Execute(SimulationState& sim_state, 
                std::shared_ptr<mfem::ParGridFunction> grid_function, 
                int region) override {
        // Get stress quadrature function for this region
        auto stress_qf = sim_state.GetQuadratureFunction("cauchy_stress_avg", region);
        if (!stress_qf) return; // Region doesn't have stress data
        
        // Project the stress calculation
        ProjectStress(stress_qf, grid_function);
    }

    /**
     * @brief Check if this projection can be aggregated globally across regions
     */
    virtual bool CanAggregateGlobally() const override { return true; }

protected:
    /**
     * @brief Pure virtual method for specific stress calculations
     * @param stress_qf Quadrature function containing stress tensor (6 components in Voigt notation)
     * @param grid_function Target grid function to populate
     */
    virtual void ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                               std::shared_ptr<mfem::ParGridFunction> grid_function) = 0;
};

/**
 * @brief Full Cauchy stress tensor projection
 */
class CauchyStressProjection final : public StressProjection {
public:

    CauchyStressProjection() = default;
    ~CauchyStressProjection() {};

    int GetVectorDimension() const override { return 6; } // Symmetric tensor in Voigt notation
    std::string GetDisplayName() const override { return "Cauchy Stress"; }
    
protected:
    virtual void ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                      std::shared_ptr<mfem::ParGridFunction> stress_gf) override {

        // Get stress data and compute Von Mises
        const int nelems = stress_gf->ParFESpace()->GetNE();
        const auto part_quad_space = stress_qf->GetPartialSpaceShared();
        const int local_nelems = part_quad_space->GetNE();

        const auto l2g = part_quad_space->getLocal2Global().Read();
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
 * @brief Von Mises stress projection
 */
class VonMisesStressProjection final : public StressProjection {
public:

    VonMisesStressProjection() = default;
    ~VonMisesStressProjection() {};

    int GetVectorDimension() const override { return 1; } // Scalar quantity
    std::string GetDisplayName() const override { return "Von Mises Stress"; }
    
protected:
    virtual void ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                      std::shared_ptr<mfem::ParGridFunction> von_mises) override {
        // Get stress data and compute Von Mises
        const int nelems = von_mises->ParFESpace()->GetNE();
        const auto part_quad_space = stress_qf->GetPartialSpaceShared();
        const int local_nelems = part_quad_space->GetNE();

        const auto l2g = part_quad_space->getLocal2Global().Read();
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
 * @brief Hydrostatic stress projection
 */
class HydrostaticStressProjection final : public StressProjection {
public:


    HydrostaticStressProjection() = default;
    ~HydrostaticStressProjection() {};

    int GetVectorDimension() const override { return 1; } // Scalar quantity
    std::string GetDisplayName() const override { return "Hydrostatic Stress"; }
    
protected:
    virtual void ProjectStress(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> stress_qf,
                      std::shared_ptr<mfem::ParGridFunction> hydro_static) override {
        // Get stress data and compute Von Mises
        const int nelems = hydro_static->ParFESpace()->GetNE();
        const auto part_quad_space = stress_qf->GetPartialSpaceShared();
        const int local_nelems = part_quad_space->GetNE();

        const auto l2g = part_quad_space->getLocal2Global().Read();
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
 * @brief Base class for state variable projections
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
    
    void Execute(SimulationState& sim_state, 
                std::shared_ptr<mfem::ParGridFunction> state_gf, 
                int region) override {
        // Get state variable quadrature function for this region
        auto state_qf = sim_state.GetQuadratureFunction("state_var_avg", region);
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

        const auto l2g = part_quad_space->getLocal2Global().Read();
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
        PostProcessStateVariable(state_gf, part_quad_space);
    }
    
    int GetVectorDimension() const override { return m_component_length; }

protected:

    virtual void PostProcessStateVariable(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                          std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace) const {};

    std::string m_state_var_name;
    int m_component_index;
    int m_component_length;
};

/**
 * @brief Generic state variable projection - extracts all state variables
 */
class AllStateVariablesProjection final : public StateVariableProjection {
public:
    AllStateVariablesProjection() : StateVariableProjection("all_state_vars", 0, -1) {}

    AllStateVariablesProjection(const std::string& state_var_name,
                                int component_index,
                                int component_length)
                                : StateVariableProjection("all_state_vars", 0, -1) {}

    ~AllStateVariablesProjection() {};

    std::string GetDisplayName() const override { return "All State Variables"; }
    virtual bool CanAggregateGlobally() const override { return false; }
};

/**
 * @brief Equivalent plastic strain rate (eq_pl_strain_rate) projection
 */
class DpEffProjection final : public StateVariableProjection {
public:
    DpEffProjection(const std::string& state_var_name,
                    int component_index,
                    int component_length)
                    : StateVariableProjection("eq_pl_strain_rate", component_index, 1, ptmc::EXACMECH_ONLY) {}
    ~DpEffProjection() = default;

    std::string GetDisplayName() const override { return "Equivalent Plastic Strain Rate"; }
    virtual bool CanAggregateGlobally() const override { return true; }

protected:
    virtual
    void PostProcessStateVariable(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                  std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace) const override {
        const int nelems = grid_function->ParFESpace()->GetNE();
        auto data = grid_function->Write();
        const auto l2g = qspace->getLocal2Global().Read();
        const int local_nelems = qspace->GetNE();

        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int i) {
            const int ie = l2g[i];
            data[ie] = fmax(data[ie], 0.0);
        });
    }

};

/**
 * @brief Crystal orientation projection
 */
class XtalOrientationProjection final : public StateVariableProjection {
public:
    XtalOrientationProjection(const std::string& state_var_name,
                              int component_index,
                              int component_length)
                              : StateVariableProjection("quats", component_index, 4, ptmc::EXACMECH_ONLY) {}
    ~XtalOrientationProjection() = default;

    std::string GetDisplayName() const override { return "Crystal Orientations"; }
    virtual bool CanAggregateGlobally() const override { return true; }

protected:
    virtual
    void PostProcessStateVariable(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                  std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace) const override {
        const int nelems = grid_function->ParFESpace()->GetNE();
        auto ori = mfem::Reshape(grid_function->Write(), grid_function->VectorDim(), nelems);
        const auto l2g = qspace->getLocal2Global().Read();
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
 * @brief Elastic strain projection
 */
class ElasticStrainProjection final : public StateVariableProjection {
public:
    ElasticStrainProjection(const std::string& state_var_name,
                            int component_index,
                            int component_length)
                            : StateVariableProjection("elastic_strain", component_index, 6, ptmc::EXACMECH_ONLY) {}

    void Execute(SimulationState& sim_state, 
                std::shared_ptr<mfem::ParGridFunction> elastic_strain_gf, 
                int region) override {

        // Get state variable quadrature function for this region
        auto state_qf = sim_state.GetQuadratureFunction("state_var_avg", region);
        if (!state_qf) return; // Region doesn't have state variables

        const int nelems = elastic_strain_gf->ParFESpace()->GetNE();
        const auto part_quad_space = state_qf->GetPartialSpaceShared();

        const auto l2g = part_quad_space->getLocal2Global().Read();
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

        const int estrain_ind = sim_state.GetQuadratureFunctionStatePair("elastic_strain", region).first;
        const int quats_ind = sim_state.GetQuadratureFunctionStatePair("quats", region).first;
        const int rel_vol_ind = sim_state.GetQuadratureFunctionStatePair("relative_volume", region).first;

        auto state_vars = sim_state.GetQuadratureFunction("state_var_end", region)->Read();
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

                quat2rmat_v2(quats, rmat);
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
 * @brief Hardness projection
 */
class HardnessProjection final : public StateVariableProjection {
public:
    HardnessProjection(const std::string& state_var_name,
                       int component_index,
                       int component_length)
                       : StateVariableProjection("hardness", component_index, component_length, ptmc::EXACMECH_ONLY) {}
    
    std::string GetDisplayName() const override { return "Hardness"; }

    virtual bool CanAggregateGlobally() const override { return false; }
    
protected:

    virtual
    void PostProcessStateVariable(std::shared_ptr<mfem::ParGridFunction> grid_function,
                                  std::shared_ptr<mfem::expt::PartialQuadratureSpace> qspace) const override {
        // Ensure non-negative values
        double* data = grid_function->ReadWrite();
        const int size = grid_function->Size();
        const auto l2g = qspace->getLocal2Global().Read();
        const int local_nelems = qspace->GetNE();

        mfem::forall(local_nelems, [=] MFEM_HOST_DEVICE (int i) {
            const int ie = l2g[i];
            data[ie] = fmax(data[ie], 0.0);
        });
    }
};

/**
 * @brief Macroscopic Shear Rate projection
 */
class ShearingRateProjection final : public StateVariableProjection {
public:
    ShearingRateProjection(const std::string& state_var_name,
                       int component_index,
                       int component_length)
                       : StateVariableProjection("shear_rate", component_index, component_length, ptmc::EXACMECH_ONLY) {}

    std::string GetDisplayName() const override { return "Shearing Rate"; }
    virtual bool CanAggregateGlobally() const override { return false; }
};