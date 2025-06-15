#pragma once

#include "mfem.hpp"
#include "mfem_expt/partial_qfunc.hpp"
#include "options/option_parser_v2.hpp"

namespace ProjectionTraits {

/**
 * @brief Model compatibility enumeration for projections
 */
enum class ModelCompatibility {
    ALL_MODELS,      // Compatible with all material models
    EXACMECH_ONLY,   // Only compatible with ExaCMech models
    UMAT_ONLY        // Only compatible with UMAT models
};

/**
 * @brief Base projection trait template
 * 
 * This provides the interface that all projection traits must implement.
 * The multi-material system relies on these traits to determine compatibility
 * and execute appropriate projections for each region.
 */
template<typename Derived>
class ProjectionTrait {
public:
    /**
     * @brief Get model compatibility for this projection
     */
    static ModelCompatibility GetModelCompatibility() {
        return Derived::GetModelCompatibility();
    }
    
    /**
     * @brief Select component from quadrature function coefficient
     * 
     * This method extracts the appropriate component(s) from a partial
     * quadrature function for projection to a grid function.
     * 
     * @param qfvc Quadrature function coefficient to modify
     * @param pqf Source partial quadrature function
     */
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                              const int index = 0, const int length = 1) {
        Derived::SelectComponent(qfvc, index, length);
    }
    
    /**
     * @brief Apply post-processing to grid function
     * 
     * This method can modify the grid function after projection,
     * for example to apply scaling, coordinate transformations, etc.
     * 
     * @param gf Grid function to post-process
     */
    static void PostProcess(mfem::ParGridFunction& gf) {
        Derived::PostProcess(gf);
    }
    
    /**
     * @brief Check if this projection can be aggregated globally
     * 
     * Some projections (like orientations) don't make sense when
     * combined across material regions.
     */
    static bool CanAggregateGlobally() {
        return Derived::CanAggregateGlobally();
    }
};

/**
 * @brief Stress tensor projection
 * 
 * Projects the full Cauchy stress tensor. Compatible with all material models
 * since all models compute stress.
 */
class ModelStressProjection : public ProjectionTrait<ModelStressProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
    
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                               [[maybe_unused]] const int index = 0, [[maybe_unused]] const int length = 1) {
        // The stress quadrature function should already have the correct format
        // Just ensure the coefficient is using the right data
        qfvc.SetComponent(0, 6);
    }
    
    static void PostProcess([[maybe_unused]] mfem::ParGridFunction& gf) {
        // No post-processing needed for stress tensor
    }
    
    static bool CanAggregateGlobally() {
        return true;
    }
};

/**
 * @brief Von Mises stress projection
 * 
 * Computes Von Mises equivalent stress from the stress tensor.
 * This is a special projection that transforms one quadrature function
 * into another before projection.
 */
class VonMisesProjection : public ProjectionTrait<VonMisesProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
    
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                [[maybe_unused]] const int index = 0, [[maybe_unused]] const int length = 1) {
        // This should be called after the Von Mises calculation is already done
        // and stored in a separate quadrature function
        qfvc.SetComponent(0, 1);
    }
    
    static void PostProcess(mfem::ParGridFunction& gf) {
        // Ensure non-negative values
        double* data = gf.ReadWrite();
        const int size = gf.Size();
        
        mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) {
            data[i] = fmax(data[i], 0.0);
        });
    }
    
    static bool CanAggregateGlobally() {
        return true;
    }
    
    /**
     * @brief Compute Von Mises stress from stress tensor
     * 
     * This static method can be used to compute Von Mises stress
     * from a stress tensor quadrature function.
     * 
     * @param stress_gf Source stress grid function (6-component)
     * @param vm_gf Target Von Mises grid function (scalar)
     */
    static void PostProcess(mfem::ParGridFunction& stress_gf, mfem::ParGridFunction& vm_gf) {
        const int nelems = stress_gf.ParFESpace()->GetNE();
        const double* stress_data = stress_gf.Read();
        double* vm_data = vm_gf.ReadWrite();
        
        // Compute Von Mises stress: sqrt(3/2 * dev_stress : dev_stress)
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
            // Extract stress components (assuming Voigt notation: xx, yy, zz, xy, xz, yz)
            const double sxx = stress_data[ie * 6 + 0];
            const double syy = stress_data[ie * 6 + 1];
            const double szz = stress_data[ie * 6 + 2];
            const double sxy = stress_data[ie * 6 + 3];
            const double sxz = stress_data[ie * 6 + 4];
            const double syz = stress_data[ie * 6 + 5];
            
            // Compute hydrostatic stress
            const double hydro = (sxx + syy + szz) / 3.0;
            
            // Compute deviatoric stress components
            const double dev_xx = sxx - hydro;
            const double dev_yy = syy - hydro;
            const double dev_zz = szz - hydro;
            
            // Von Mises stress
            const double vm = sqrt(1.5 * (dev_xx*dev_xx + dev_yy*dev_yy + dev_zz*dev_zz +
                                         2.0*(sxy*sxy + sxz*sxz + syz*syz)));
            
            vm_data[ie] = vm;
        });
    }
};

/**
 * @brief Hydrostatic stress projection
 */
class HydroStressProjection : public ProjectionTrait<HydroStressProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
    
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                [[maybe_unused]] const int index = 0, [[maybe_unused]] const int length = 1) {
        qfvc.SetComponent(0, 1);
    }
    
    static void PostProcess([[maybe_unused]] mfem::ParGridFunction& gf) {
        // No special post-processing needed
    }
    
    static bool CanAggregateGlobally() {
        return true;
    }
    
    /**
     * @brief Compute hydrostatic stress from stress tensor
     */
    static void PostProcess(mfem::ParGridFunction& stress_gf, mfem::ParGridFunction& hydro_gf) {
        const int nelems = stress_gf.ParFESpace()->GetNE();
        const double* stress_data = stress_gf.Read();
        double* hydro_data = hydro_gf.ReadWrite();
        
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
            const double sxx = stress_data[ie * 6 + 0];
            const double syy = stress_data[ie * 6 + 1];
            const double szz = stress_data[ie * 6 + 2];
            
            hydro_data[ie] = (sxx + syy + szz) / 3.0;
        });
    }
};

/**
 * @brief Effective plastic strain rate projection (ExaCMech only)
 */
class DpEffProjection : public ProjectionTrait<DpEffProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
    
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                const int index = 0, const int length = 1) {
        qfvc.SetComponent(index, length);
    }
    
    static void PostProcess(mfem::ParGridFunction& gf) {
        // Ensure non-negative values
        double* data = gf.ReadWrite();
        const int size = gf.Size();
        
        mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) {
            data[i] = fmax(data[i], 0.0);
        });
    }
    
    static bool CanAggregateGlobally() {
        return true;
    }
};

/**
 * @brief Effective plastic strain projection (ExaCMech only)
 */
class EffPlasticStrainProjection : public ProjectionTrait<EffPlasticStrainProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
    
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                const int index = 0, const int length = 1) {
        qfvc.SetComponent(index, length);
    }
    
    static void PostProcess(mfem::ParGridFunction& gf) {
        // Ensure non-negative values
        double* data = gf.ReadWrite();
        const int size = gf.Size();
        
        mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) {
            data[i] = fmax(data[i], 0.0);
        });
    }
    
    static bool CanAggregateGlobally() {
        return true;
    }
};

/**
 * @brief Shear rate projection (ExaCMech only)
 */
class ShearRateProjection : public ProjectionTrait<ShearRateProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
    
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                const int index = 0, const int length = 1) {
        qfvc.SetComponent(index, length);
    }
    
    static void PostProcess(mfem::ParGridFunction& gf) {
        // Ensure non-negative values
        double* data = gf.ReadWrite();
        const int size = gf.Size();
        
        mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) {
            data[i] = fmax(data[i], 0.0);
        });
    }
    
    static bool CanAggregateGlobally() {
        return true;
    }
};

/**
 * @brief Crystal orientation projection (ExaCMech only)
 * 
 * This projection handles quaternion-based crystal orientations.
 * Note: Orientations generally should not be aggregated globally
 * as averaging quaternions requires special handling.
 */
class OrientationProjection : public ProjectionTrait<OrientationProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
    
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                const int index = 0, const int length = 1) {
        qfvc.SetComponent(index, length);
    }
    
    static void PostProcess(mfem::ParGridFunction& gf) {
        // Normalize quaternions to unit length
        const int nelems = gf.ParFESpace()->GetNE();
        const int vdim = gf.VectorDim(); // Should be 4 for quaternions
        double* data = gf.ReadWrite();
        
        if (vdim == 4) {
            mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
                double norm_sq = 0.0;
                for (int i = 0; i < 4; ++i) {
                    const double val = data[ie * 4 + i];
                    norm_sq += val * val;
                }
                
                const double norm = sqrt(norm_sq);
                if (norm > 1e-12) {
                    const double inv_norm = 1.0 / norm;
                    for (int i = 0; i < 4; ++i) {
                        data[ie * 4 + i] *= inv_norm;
                    }
                }
            });
        }
    }
    
    static bool CanAggregateGlobally() {
        return false; // Quaternion averaging requires special handling
    }
};

/**
 * @brief Hardness parameter projection (ExaCMech only)
 */
class HProjection : public ProjectionTrait<HProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
    
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                const int index = 0, const int length = 1) {
        qfvc.SetComponent(index, length);
    }
    
    static void PostProcess(mfem::ParGridFunction& gf) {
        // Ensure non-negative values
        double* data = gf.ReadWrite();
        const int size = gf.Size();
        
        mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) {
            data[i] = fmax(data[i], 0.0);
        });
    }
    
    static bool CanAggregateGlobally() {
        return true;
    }
};

/**
 * @brief Geometry-based projections (always available)
 * 
 * These projections work directly with mesh geometry rather than
 * quadrature function data.
 */
class CentroidProjection : public ProjectionTrait<CentroidProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
    
    static void SelectComponent([[maybe_unused]] mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                [[maybe_unused]] const int index = 0, [[maybe_unused]] const int length = 1) {
        // Not used for geometry projections
    }
    
    static void PostProcess([[maybe_unused]] mfem::ParGridFunction& gf) {
        // No post-processing needed
    }
    
    static bool CanAggregateGlobally() {
        return true;
    }
    
    /**
     * @brief Project element centroids directly to grid function
     */
    static void Project(mfem::ParFiniteElementSpace* fes, mfem::ParGridFunction& gf) {
        auto mesh = fes->GetMesh();
        const mfem::FiniteElement &el = *fes->GetFE(0);
        const mfem::IntegrationRule *ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
        
        const int nqpts = ir->GetNPoints();
        const int nelems = fes->GetNE();
        const int vdim = mesh->SpaceDimension();
        
        const double* W = ir->GetWeights().Read();
        const mfem::GeometricFactors *geom = mesh->GetGeometricFactors(
            *ir, mfem::GeometricFactors::DETERMINANTS | mfem::GeometricFactors::COORDINATES);
        
        double* centroid_data = gf.ReadWrite();
        
        // Calculate element centroids
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
            double vol = 0.0;
            for (int iv = 0; iv < vdim; ++iv) {
                centroid_data[ie * vdim + iv] = 0.0;
            }
            
            for (int iq = 0; iq < nqpts; ++iq) {
                const double wt = geom->detJ.Read()[ie * nqpts + iq] * W[iq];
                vol += wt;
                
                for (int iv = 0; iv < vdim; ++iv) {
                    const double coord = geom->X.Read()[iq * vdim * nelems + iv * nelems + ie];
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

class VolumeProjection : public ProjectionTrait<VolumeProjection> {
public:
    static ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
    
    static void SelectComponent([[maybe_unused]] mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                [[maybe_unused]] const int index = 0, [[maybe_unused]] const int length = 1) {
        // Not used for geometry projections
    }
    
    static void PostProcess(mfem::ParGridFunction& gf) {
        // Ensure non-negative volumes
        double* data = gf.ReadWrite();
        const int size = gf.Size();
        
        mfem::forall(size, [=] MFEM_HOST_DEVICE (int i) {
            data[i] = fmax(data[i], 0.0);
        });
    }
    
    static bool CanAggregateGlobally() {
        return true;
    }
    
    /**
     * @brief Project element volumes directly to grid function
     */
    static void Project(mfem::ParFiniteElementSpace* fes, mfem::ParGridFunction& gf) {
        auto mesh = fes->GetMesh();
        const mfem::FiniteElement &el = *fes->GetFE(0);
        const mfem::IntegrationRule *ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));
        
        const int nqpts = ir->GetNPoints();
        const int nelems = fes->GetNE();
        
        const double* W = ir->GetWeights().Read();
        const mfem::GeometricFactors *geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::DETERMINANTS);
        
        double* vol_data = gf.ReadWrite();
        
        // Calculate element volumes
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int ie) {
            vol_data[ie] = 0.0;
            for (int iq = 0; iq < nqpts; ++iq) {
                vol_data[ie] += geom->detJ.Read()[ie * nqpts + iq] * W[iq];
            }
        });
    }
};

} // namespace ProjectionTraits