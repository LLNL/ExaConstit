#pragma once

#include "mfem/config/config.hpp"
#include "mfem/fem/fespace.hpp"
#include <unordered_map>
#include <memory>
#include <optional>
#include <array>
#include <string_view>

namespace mfem::expt
{

/// Class representing a subset of a QuadratureSpace, for efficient operations on subdomains.
class PartialQuadratureSpace : public mfem::QuadratureSpaceBase {
protected:
    friend class PartialQuadratureFunction; // Uses the offsets.
    // Maps local indices to global mesh element indices
    mfem::Array<int> local2global;
    mfem::Array<int> global2local;
    // Maps global mesh element indices to local indices (-1 if not in partial set)
    mfem::Array<int> global_offsets;

protected:
    // Implementation of GetGeometricFactorWeights required by the base class
    const mfem::Vector &GetGeometricFactorWeights() const override;
    void ConstructOffsets();
    void Construct();
    void ConstructMappings(std::shared_ptr<mfem::Mesh> mesh, mfem::Array<bool>& partial_index);

public:
    /// Create a PartialQuadratureSpace based on the global rules from #IntRules.
    PartialQuadratureSpace(std::shared_ptr<mfem::Mesh> mesh_, int order_, mfem::Array<bool>& partial_index);

    /// Create a PartialQuadratureSpace based on the global rules from #IntRules (deprecated).
    [[deprecated("Use constructor with std::shared_ptr<mfem::Mesh> instead")]]
    PartialQuadratureSpace(mfem::Mesh* mesh_, int order_, mfem::Array<bool>& partial_index)
        : PartialQuadratureSpace(ptr_utils::borrow_ptr(mesh_), order_, partial_index) { }

    /// Create a PartialQuadratureSpace with an mfem::IntegrationRule, valid only when
    /// the mesh has one element type.
    PartialQuadratureSpace(std::shared_ptr<mfem::Mesh> mesh_, const mfem::IntegrationRule &ir, 
                            mfem::Array<bool>& partial_index);

    /// Create a PartialQuadratureSpace with an mfem::IntegrationRule (deprecated).
    [[deprecated("Use constructor with std::shared_ptr<mfem::Mesh> instead")]]
    PartialQuadratureSpace(mfem::Mesh* mesh_, const mfem::IntegrationRule &ir, mfem::Array<bool>& partial_index)
        : PartialQuadratureSpace(ptr_utils::borrow_ptr(mesh_), ir, partial_index) { }

    /// Read a PartialQuadratureSpace from the stream @a in.
    PartialQuadratureSpace(std::shared_ptr<mfem::Mesh> mesh_, std::istream &in);

    /// Read a PartialQuadratureSpace from the stream @a in (deprecated).
    [[deprecated("Use constructor with std::shared_ptr<mfem::Mesh> instead")]]
    PartialQuadratureSpace(mfem::Mesh* mesh_, std::istream &in)
        : PartialQuadratureSpace(ptr_utils::borrow_ptr(mesh_), in) { }

    // Mapping functions
    [[nodiscard]]
    int LocalToGlobal(int local_idx) const {
        if (local_idx >= 0 && local_idx < local2global.Size()) {
            return local2global[local_idx];
        }
        return -1;
    }
    
    [[nodiscard]] 
    int GlobalToLocal(int global_idx) const {
        if (global_idx >= 0 && global_idx < global2local.Size()) {
            return global2local(global_idx, 0);
        }
        else if (global_idx >= 0 && global2local.Size() == 1) {
            return global_idx;
        }
        return -1;
    }
    
    const mfem::Array<int>& getGlobal2Local() const { return global2local; }
    const mfem::Array<int>& getGlobalOffset() const { return global_offset; }


    // Implementation of QuadratureSpaceBase methods
    
    /// Get the element transformation for a local entity index
    [[nodiscard]] mfem::ElementTransformation *GetTransformation(int idx) override
    {
        int global_idx = LocalToGlobal(idx);
        return mesh->GetElementTransformation(global_idx);
    }

    /// Return the geometry type of the entity with local index idx
    [[nodiscard]] mfem::Geometry::Type GetGeometry(int idx) const override
    {
        int global_idx = LocalToGlobal(idx);
        return mesh->GetElementGeometry(global_idx);
    }

    /// For element quadrature spaces, the permutation is trivial
    [[nodiscard]] int GetPermutedIndex(int idx, int iq) const override
    {
        // For element quadrature spaces, the permutation is trivial
        return iq;
    }

    /// Save the PartialQuadratureSpace to a stream
    void Save(std::ostream &out) const override;

    /// Returns the element index in our partial space for the given mfem::ElementTransformation
    [[nodiscard]] int GetEntityIndex(const mfem::ElementTransformation &T) const override
    {
        return T.ElementNo;
    }

    // Factory methods
    
    /// Create a shared_ptr PartialQuadratureSpace from a mfem::Mesh shared_ptr
    static std::shared_ptr<PartialQuadratureSpace> Create(std::shared_ptr<mfem::Mesh> mesh, 
                                                            int order, 
                                                            mfem::Array<bool> partial_index) {
        return std::make_shared<PartialQuadratureSpace>(std::move(mesh), order, partial_index);
    }

    /// Create a shared_ptr PartialQuadratureSpace from a raw mfem::Mesh pointer (deprecated)
    [[deprecated("Use Create() with std::shared_ptr<mfem::Mesh> instead")]]
    static std::shared_ptr<PartialQuadratureSpace> Create(mfem::Mesh* mesh, 
                                                            int order, 
                                                            mfem::Array<bool> partial_index) {
        return std::make_shared<PartialQuadratureSpace>(
            ptr_utils::borrow_ptr(mesh), order, partial_index);
    }
};


}