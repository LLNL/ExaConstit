#pragma once

#include "partial_qspace.hpp"

#include "mfem/config/config.hpp"
#include "mfem/general/forall.hpp"
#include "mfem/fem/qspace.hpp"
#include "mfem/fem/qfunction.hpp"

#include <unordered_map>
#include <memory>
#include <optional>
#include <array>
#include <string_view>

namespace mfem::expt
{

/// Class for representing quadrature functions on a subset of mesh elements
class PartialQuadratureFunction : public QuadratureFunction {
private:
    // Reference to the specialized QuadratureSpace
    std::shared_ptr<PartialQuadratureSpace> part_quad_space;
    
    // Default value for elements not in our partial set
    double default_value;

public:
    /// Constructor with shared_ptr to PartialQuadratureSpace
    PartialQuadratureFunction(std::shared_ptr<PartialQuadratureSpace> qspace_, int vdim_ = 1, double default_val = -1.0)
        : QuadratureFunction(std::static_pointer_cast<QuadratureSpaceBase>(qspace_), vdim_), 
            part_quad_space(std::move(qspace_)), default_value(default_val)
    { }

    /// Constructor with raw pointer to PartialQuadratureSpace (deprecated)
    [[deprecated("Use constructor with std::shared_ptr<PartialQuadratureSpace> instead")]]
    PartialQuadratureFunction(PartialQuadratureSpace* qspace_, int vdim_ = 1, double default_val = -1.0)
        : PartialQuadratureFunction(ptr_utils::borrow_ptr(qspace_), vdim_, default_val)
    { }

    // Get the specialized PartialQuadratureSpace as shared_ptr
    [[nodiscard]]
    std::shared_ptr<PartialQuadratureSpace>
    GetPartialSpaceShared() const { 
        return part_quad_space; 
    }

    // Get the specialized PartialQuadratureSpace as raw pointer (deprecated)
    [[deprecated("Use GetPartialSpaceShared() instead")]]
    [[nodiscard]]
    PartialQuadratureSpace*
    GetPartialSpace() const { 
        return part_quad_space.get(); 
    }

    /// Set this equal to a constant value.
    PartialQuadratureFunction &
    operator=(double value) override
    {
        QuadratureFunction::operator=(value);
        return *this;
    }

    /// Copy the data from @a vec.
    PartialQuadratureFunction &
    operator=(const Vector &vec) override
    {
        MFEM_ASSERT(part_quad_space && vec.Size() == this->Size(), "");
        QuadratureFunction::operator=(vec);
        return *this;
    }

    /// Copy the data from @a qf.
    // this is wrong we need to check and see if first the sizes are equal if so it's a simple
    // copy. If not then we need to want to check and see if the meshes are equal,
    // integration rules are same or integration rule are same then we can fill things up easy
    // peasy
    PartialQuadratureFunction &operator=(const QuadratureFunction &qf);

    /// Takes in a quadrature function and fill with either the values contained in this
    /// class or the default value provided by users.
    // Note might want to allow the user to decide if we should fill things or not
    // aka when we might be doing things like setting the global mtan or stress vecs
    void FillQuadratureFunction(QuadratureFunction &qf, const bool fill = false);

    /// Override ProjectGridFunction to project only onto the partial space
    /// Currently unsupported but something we can look at in the future.
    void ProjectGridFunction([[maybe_unused]] const GridFunction &gf) override
    {
        MFEM_ABORT("Unsupported case.");
    }

    /// Return all values associated with mesh element @a idx in a Vector.
    /** The result is stored in the Vector @a values as a reference to the
         global values. Although, if idx is not a valid index for the PQF
         then the vector will be set to the appropriate global size and have
         the user default values assigned to it.

        Inside the Vector @a values, the index `i+vdim*j` corresponds to the
        `i`-th vector component at the `j`-th quadrature point.
    */
    virtual void GetValues(int idx, Vector &values) override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = part_quad_space->offsets[local_index];
            const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
            values.MakeRef(*this, vdim * s_offset, vdim * sl_size);
        } else {
            const int s_offset = part_quad_space->global_offsets[idx];
            const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
            values.Destroy();
            values.SetSize(vdim * sl_size);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++)
            {
               values(i) = default_value;
            }
        }
    }

    /// Return all values associated with mesh element @a idx in a Vector.
    /** The result is stored in the Vector @a values as a copy of the
         global values.

        Inside the Vector @a values, the index `i+vdim*j` corresponds to the
        `i`-th vector component at the `j`-th quadrature point.
    */
    virtual void GetValues(int idx, Vector &values) const override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = part_quad_space->offsets[local_index];
            const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
            values.SetSize(vdim * sl_size);
            values.HostWrite();
            const real_t *q = HostRead() + vdim * s_offset;
            for (int i = 0; i < values.Size(); i++)
            {
               values(i) = *(q++);
            }
        } else {
            const int s_offset = part_quad_space->global_offsets[idx];
            const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
            values.SetSize(vdim * sl_size);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++)
            {
                values(i) = default_value;
            }
        }
    }


    /// Return the quadrature function values at an integration point.
    /** The result is stored in the Vector @a values as a reference to the
         global values. */
    virtual void GetValues(int idx, const int ip_num, Vector &values) override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = (part_quad_space->offsets[local_index] + ip_num) * vdim;
            values.MakeRef(*this, s_offset, vdim);
        } else {
            values.Destroy();
            values.SetSize(vdim);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++)
            {
               values(i) = default_value;
            }
        }
    }

    /// Return the quadrature function values at an integration point.
    /** The result is stored in the Vector @a values as a copy to the
         global values. */
    virtual void GetValues(int idx, const int ip_num, Vector &values) const override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = (part_quad_space->offsets[local_index] + ip_num) * vdim;
            const real_t *q = HostRead() + s_offset;
            values.SetSize(vdim);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++)
            {
               values(i) = *(q++);
            }
        } else {
            values.Destroy();
            values.SetSize(vdim);
            values.HostWrite();
            for (int i = 0; i < values.Size(); i++)
            {
               values(i) = default_value;
            }
        }
    }

    /// Return all values associated with mesh element @a idx in a DenseMatrix.
    /** The result is stored in the DenseMatrix @a values as a reference to the
         global values.

        Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
        `i`-th vector component at the `j`-th quadrature point.
    */
    virtual void GetValues(int idx, DenseMatrix &values) override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = part_quad_space->offsets[local_index];
            const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
            // Make the values matrix memory an alias of the quadrature function memory
            Memory<real_t> &values_mem = values.GetMemory();
            values_mem.Delete();
            values_mem.MakeAlias(GetMemory(), vdim * s_offset, vdim * sl_size);
            values.SetSize(vdim, sl_size);
        } else {
            const int s_offset = part_quad_space->global_offsets[idx];
            const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
            values.Clear();
            values.SetSize(vdim, sl_size);
            values.HostWrite();
            for (int j = 0; j < sl_size; j++)
            {
               for (int i = 0; i < vdim; i++)
               {
                  values(i, j) = default_value;
               }
            }
        }
    }


    /// Return all values associated with mesh element @a idx in a const DenseMatrix.
    /** The result is stored in the DenseMatrix @a values as a copy of the
         global values.

        Inside the DenseMatrix @a values, the `(i,j)` entry corresponds to the
        `i`-th vector component at the `j`-th quadrature point.
    */
    virtual void GetValues(int idx, DenseMatrix &values) const override {
        const int local_index = part_quad_space->GlobalToLocal(idx);
        // If global_offsets.Size() == 1 then we'll always
        // go down this path
        if (local_index > -1) {
            const int s_offset = part_quad_space->offsets[local_index];
            const int sl_size = part_quad_space->offsets[local_index + 1] - s_offset;
            values.SetSize(vdim, sl_size);
            values.HostWrite();
            const real_t *q = HostRead() + vdim * s_offset;
            for (int j = 0; j<sl_size; j++)
            {
               for (int i = 0; i<vdim; i++)
               {
                  values(i, j) = *(q++);
               }
            }
        } else {
            // Make the values matrix memory an alias of the quadrature function memory
            const int s_offset = part_quad_space->global_offsets[idx];
            const int sl_size = part_quad_space->global_offsets[idx + 1] - s_offset;
            values.Clear();
            values.SetSize(vdim, sl_size);
            values.HostWrite();
            for (int j = 0; j < sl_size; j++)
            {
               for (int i = 0; i < vdim; i++)
               {
                  values(i, j) = default_value;
               }
            }
        }
    }

    /// Get the IntegrationRule associated with entity (element or face) @a idx.
    using QuadratureFunction::GetIntRule;

    /// Write the QuadratureFunction to the stream @a out.
    virtual void Save(std::ostream &out) const override {
        if (part_quad_space->global_offsets.Size() == 1) {
            QuadratureFunction::Save(out);
            return;
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
    }

    /// @brief Write the QuadratureFunction to @a out in VTU (ParaView) format.
    ///
    /// The data will be uncompressed if @a compression_level is zero, or if the
    /// format is VTKFormat::ASCII. Otherwise, zlib compression will be used for
    /// binary data.
    virtual void SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0, const std::string &field_name="u") const override
    {
        if (part_quad_space->global_offsets.Size() == 1) {
            QuadratureFunction::SaveVTU(out, format, compression_level, field_name);
            return;
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
    }
                

    /// @brief Save the QuadratureFunction to a VTU (ParaView) file.
    ///
    /// The extension ".vtu" will be appended to @a filename.
    /// @sa SaveVTU(std::ostream &out, VTKFormat format=VTKFormat::ASCII,
    ///             int compression_level=0)
    virtual void SaveVTU(const std::string &filename, VTKFormat format=VTKFormat::ASCII,
                int compression_level=0, const std::string &field_name="u") const override
    {
        if (part_quad_space->global_offsets.Size() == 1) {
            QuadratureFunction::SaveVTU(filename, format, compression_level, field_name);
            return;
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
    }

    /// Return the integral of the quadrature function (vdim = 1 only).
    [[nodiscard]] virtual real_t Integrate() const override
    {
        if (part_quad_space->global_offsets.Size() == 1) {
            return QuadratureFunction::Integrate();
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
        return default_value;
    }

    /// @brief Integrate the (potentially vector-valued) quadrature function,
    /// storing the results in @a integrals (length @a vdim).
    virtual void Integrate(Vector &integrals) const override
    {
        if (part_quad_space->global_offsets.Size() == 1) {
            QuadratureFunction::Integrate(integrals);
            return;
        }
        MFEM_ABORT("Currently not supported for PartialQuadratureFunctions");
    }

    // Factory methods for creating PartialQuadratureFunction instances

    /// Create a shared_ptr PartialQuadratureFunction from a PartialQuadratureSpace shared_ptr
    static std::shared_ptr<PartialQuadratureFunction> Create(
        std::shared_ptr<PartialQuadratureSpace> qspace, int vdim = 1, double default_val = -1.0) {
        return std::make_shared<PartialQuadratureFunction>(std::move(qspace), vdim, default_val);
    }

    /// Create a shared_ptr PartialQuadratureFunction from a raw PartialQuadratureSpace pointer (deprecated)
    [[deprecated("Use Create() with std::shared_ptr<PartialQuadratureSpace> instead")]]
    static std::shared_ptr<PartialQuadratureFunction> Create(
        PartialQuadratureSpace* qspace, int vdim = 1, double default_val = -1.0) {
        return std::make_shared<PartialQuadratureFunction>(
            ptr_utils::borrow_ptr(qspace), vdim, default_val);
    }
};


}