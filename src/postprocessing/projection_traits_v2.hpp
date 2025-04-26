#pragma once

#include "mfem.hpp"
#include "ECMech_const.h"
#include "option_types.hpp"
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <vector>

// Enhanced projection traits system
namespace ProjectionTraits {

// Model compatibility enumeration
enum class ModelCompatibility {
    ALL_MODELS,
    EXACMECH_ONLY,
    UMAT_ONLY
};

// Base trait struct for common projection behavior
template<typename ProjectionType>
struct ProjectionTrait {
    // Default implementation for simple projections
    static void PreProcess(const mfem::QuadratureFunction* qf, mfem::ParGridFunction& gf, 
                           const std::pair<int, int>& component_info) {}
                           
    static void PostProcess(mfem::ParGridFunction& gf) {}
    
    // Default component selection method
    static void SelectComponent(mfem::VectorQuadratureFunctionCoefficient& qfvc,
                                const std::pair<int, int>& component_info) {
        qfvc.SetComponent(component_info.first, component_info.second);
    }
    
    // Default element averaging implementation
    static void CalcElementAvg(mfem::Vector& elemVal, const mfem::QuadratureFunction& qf, const mfem::FiniteElementSpace* fes) {
        mfem::Mesh* mesh = fes->GetMesh();
        const mfem::FiniteElement& el = *fes->GetFE(0);
        const mfem::IntegrationRule* ir = &(mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1));

        const int nqpts = ir->GetNPoints();
        const int nelems = fes->GetNE();
        const int vdim = qf.GetVDim();

        const double* W = ir->GetWeights().Read();
        const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(*ir, mfem::GeometricFactors::DETERMINANTS);

        elemVal = 0.0;

        // Use mfem::Reshape for modern, cleaner view creation
        auto j_view = mfem::Reshape(geom->detJ.Read(), nqpts, nelems);
        auto qf_view = mfem::Reshape(qf.Read(), vdim, nqpts, nelems);
        auto ev_view = mfem::Reshape(elemVal.ReadWrite(), vdim, nelems);

        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int i) {
            double vol = 0.0;
            for (int j = 0; j < nqpts; j++) {
                const double wts = j_view(j, i) * W[j];
                vol += wts;
                for (int k = 0; k < vdim; k++) {
                    ev_view(k, i) += qf_view(k, j, i) * wts;
                }
            }
            const double ivol = 1.0 / vol;
            for (int k = 0; k < vdim; k++) {
                ev_view(k, i) *= ivol;
            }
        });
    }
    
    // Generic projection from QuadratureFunction to GridFunction via element averaging
    static void ProjectQFToGF(
        const mfem::QuadratureFunction& qf,
        mfem::ParGridFunction& gf,
        mfem::Vector& elem_val,
    ) {
        CalcElementAvg(elem_val, qf, gf->FESpace());
        gf = elem_val;
    }
    
    // Project component from element-averaged vector
    static void ProjectComponent(
        const mfem::Vector& elem_val,
        mfem::ParGridFunction& gf, 
        const std::pair<int, int>& component_info
    ) {
        mfem::VectorQuadratureFunctionCoefficient qfvc(*elem_val);
        SelectComponent(qfvc, component_info);
        gf.ProjectDiscCoefficient(qfvc, mfem::GridFunction::ARITHMETIC);
    }
    
    // Model compatibility check
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
};

// Specialized trait for von Mises stress calculation
struct VonMisesStressTrait : public ProjectionTrait<VonMisesStressTrait> { {
    static void PostProcess(const mfem::ParGridFunction& stress, mfem::ParGridFunction& vonMises) {
        const int npts = vonMises.Size();
        auto stress_view = mfem::Reshape(stress.Read(), 6, npts);
        auto vm_data = vonMises.ReadWrite();
        
        mfem::forall(npts, [=] MFEM_HOST_DEVICE (int i) {
            double term1 = stress_view(0, i) - stress_view(1, i);
            double term2 = stress_view(1, i) - stress_view(2, i);
            double term3 = stress_view(2, i) - stress_view(0, i);
            double term4 = stress_view(3, i) * stress_view(3, i)
                          + stress_view(4, i) * stress_view(4, i)
                          + stress_view(5, i) * stress_view(5, i);
            
            term1 *= term1;
            term2 *= term2;
            term3 *= term3;
            term4 *= 6.0;
            
            vm_data[i] = sqrt(0.5 * (term1 + term2 + term3 + term4));
        });
    }
    
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
};

// Specialized trait for hydrostatic stress calculation
struct HydroStressTrait : public ProjectionTrait<HydroStressTrait> { {
    static void PostProcess(const mfem::ParGridFunction& stress, mfem::ParGridFunction& hydroStress) {
        const int npts = hydroStress.Size();
        auto stress_view = mfem::Reshape(stress.Read(), 6, npts);
        auto hydro = hydroStress.ReadWrite();
        
        constexpr double one_third = 1.0 / 3.0;
        
        mfem::forall(npts, [=] MFEM_HOST_DEVICE (int i) {
            hydro[i] = one_third * (stress_view(0, i) + stress_view(1, i) + stress_view(2, i));
        });
    }
    
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
};

// Specialized trait for quaternion orientation normalization
struct OrientationTrait : public ProjectionTrait<OrientationTrait> {
    static void PostProcess(mfem::ParGridFunction& quats) {
        const int npts = quats.Size() / 4;
        auto quats_view = mfem::Reshape(quats.ReadWrite(), 4, npts);

        mfem::forall(npts, [=] MFEM_HOST_DEVICE (int i) {
            double norm = 0.0;
            norm = quats_view(0, i) * quats_view(0, i);
            norm += quats_view(1, i) * quats_view(1, i);
            norm += quats_view(2, i) * quats_view(2, i);
            norm += quats_view(3, i) * quats_view(3, i);

            const double inv_norm = 1.0 / sqrt(norm);

            for (int j = 0; j < 4; j++) {
                quats_view(j, i) *= inv_norm;
            }
        });
    }
    
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
};

// Specialized trait for elastic strain transformation
struct ElasticStrainTrait : public ProjectionTrait<ElasticStrainTrait> {
    static void PostProcess(
        mfem::ParGridFunction& estrain, 
        const mfem::Vector& evec, 
        const std::pair<int, int>& strain_info,
        const std::pair<int, int>& vol_info
    ) {
        const int e_offset = strain_info.first;
        const int rv_offset = vol_info.first;
        
        const int nelems = estrain.Size() / 6;
        
        auto data_estrain = mfem::Reshape(estrain.ReadWrite(), 6, nelems);
        auto data_evec = mfem::Reshape(evec.Read(), evec.Size()/nelems, nelems);
        
        // Use device kernel when possible
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int i) {
            const double t1 = ecmech::sqr2i * data_evec(e_offset, i);
            const double t2 = ecmech::sqr6i * data_evec(e_offset+1, i);
            
            const double elas_vol_strain = log(data_evec(rv_offset, i));
            
            data_estrain(0, i) = (t1 - t2) + elas_vol_strain; 
            data_estrain(1, i) = (-t1 - t2) + elas_vol_strain;
            data_estrain(2, i) = ecmech::sqr2b3 * data_evec(e_offset+1, i) + elas_vol_strain;
            data_estrain(3, i) = ecmech::sqr2i * data_evec(e_offset+4, i); 
            data_estrain(4, i) = ecmech::sqr2i * data_evec(e_offset+3, i);
            data_estrain(5, i) = ecmech::sqr2i * data_evec(e_offset+2, i);
        });
    }
    
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
};

// Specialized trait for centroid calculation
struct CentroidTrait : public ProjectionTrait<CentroidTrait> {
    static void Project(
        const mfem::ParFiniteElementSpace* fes,
        mfem::ParGridFunction& centroid
    ) {
        mfem::Mesh* mesh = fes->GetMesh();
        const mfem::FiniteElement& el = *fes->GetFE(0);
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1);
        
        const int nqpts = ir.GetNPoints();
        const int nelems = fes->GetNE();
        const int vdim = mesh->SpaceDimension();
        
        const double* W = ir.GetWeights().Read();
        const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(
            ir, mfem::GeometricFactors::DETERMINANTS | mfem::GeometricFactors::COORDINATES);

        centroid = 0.0;

        auto j_view = mfem::Reshape(geom->detJ.Read(), nqpts, nelems);
        auto x_view = mfem::Reshape(geom->X.Read(), nqpts, vdim, nelems);
        auto cent_view = mfem::Reshape(centroid.ReadWrite(), vdim, nelems);

        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int i) {
            double vol = 0.0;
            for (int j = 0; j < nqpts; j++) {
                const double wts = j_view(j, i) * W[j];
                vol += wts;
                for (int k = 0; k < vdim; k++) {
                    cent_view(k, i) += x_view(j, k, i) * wts;
                }
            }
            const double ivol = 1.0 / vol;
            for (int k = 0; k < vdim; k++) {
                cent_view(k, i) *= ivol;
            }
        });
    }
    
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
};

// Specialized trait for volume calculation
struct VolumeTrait : public ProjectionTrait<VolumeTrait> {
    static void Project(
        const mfem::ParFiniteElementSpace* fes,
        mfem::ParGridFunction& vol
    ) {
        mfem::Mesh* mesh = fes->GetMesh();
        const mfem::FiniteElement& el = *fes->GetFE(0);
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 1);
        
        const int nqpts = ir.GetNPoints();
        const int nelems = fes->GetNE();
        
        const double* W = ir.GetWeights().Read();
        const mfem::GeometricFactors* geom = mesh->GetGeometricFactors(ir, mfem::GeometricFactors::DETERMINANTS);

        auto j_view = mfem::Reshape(geom->detJ.Read(), nqpts, nelems);
        auto vol_data = vol.ReadWrite();
        
        mfem::forall(nelems, [=] MFEM_HOST_DEVICE (int i) {
            vol_data[i] = 0.0;
            for (int j = 0; j < nqpts; j++) {
                vol_data[i] += j_view(j, i) * W[j];
            }
        });
    }
    
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::ALL_MODELS;
    }
};

// Trait for effective plastic deformation rate
struct DpEffTrait : public ProjectionTrait<DpEffTrait> {
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
};

// Trait for effective plastic strain
struct EffPlasticStrainTrait : public ProjectionTrait<EffPlasticStrainTrait> {
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
};

// Trait for shear rate
struct ShearRateTrait : public ProjectionTrait<ShearRateTrait> {
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
};

// Trait for hardness
struct HardnessTrait : public ProjectionTrait<HardnessTrait> {
    static constexpr ModelCompatibility GetModelCompatibility() {
        return ModelCompatibility::EXACMECH_ONLY;
    }
};

} // namespace ProjectionTraits