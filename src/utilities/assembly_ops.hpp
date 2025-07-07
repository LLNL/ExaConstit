#pragma once

#include "mfem_expt/partial_qfunc.hpp"

#include "mfem.hpp"

// This function is used in generating the B matrix commonly seen in the formation of
// the material tangent stiffness matrix in mechanics [B^t][Cstiff][B]
// Although we're goint to return really B^t here since it better matches up
// with how DS is set up memory wise
// The B matrix should have dimensions equal to (dof*dim, 6).
// We assume it hasn't been initialized ahead of time or it's already
// been written in, so we rewrite over everything in the below.
inline
void
GenerateGradMatrix(const mfem::DenseMatrix& DS, mfem::DenseMatrix& B)
{
    int dof = DS.Height();


    // The B matrix generally has the following structure that is
    // repeated for the number of dofs if we're dealing with something
    // that results in a symmetric Cstiff. If we aren't then it's a different
    // structure
    // [DS(i,0) 0 0]
    // [0 DS(i, 1) 0]
    // [0 0 DS(i, 2)]
    // [0 DS(i,2) DS(i,1)]
    // [DS(i,2) 0 DS(i,0)]
    // [DS(i,1) DS(i,0) 0]

    // Just going to go ahead and make the assumption that
    // this is for a 3D space. Should put either an assert
    // or an error here if it isn't
    // We should also put an assert if B doesn't have dimensions of
    // (dim*dof, 6)
    // fix_me
    // We've rolled out the above B matrix in the comments
    // This is definitely not the most efficient way of doing this memory wise.
    // However, it might be fine for our needs.
    // The ordering has now changed such that B matches up with mfem's internal
    // ordering of vectors such that it's [x0...xn, y0...yn, z0...zn] ordering

    // The previous single loop has been split into 3 so the B matrix
    // is constructed in chunks now instead of performing multiple striding
    // operations in a single loop.
    // x dofs
    for (int i = 0; i < dof; i++) {
        B(i, 0) = DS(i, 0);
        B(i, 1) = 0.0;
        B(i, 2) = 0.0;
        B(i, 3) = 0.0;
        B(i, 4) = DS(i, 2);
        B(i, 5) = DS(i, 1);
    }

    // y dofs
    for (int i = 0; i < dof; i++) {
        B(i + dof, 0) = 0.0;
        B(i + dof, 1) = DS(i, 1);
        B(i + dof, 2) = 0.0;
        B(i + dof, 3) = DS(i, 2);
        B(i + dof, 4) = 0.0;
        B(i + dof, 5) = DS(i, 0);
    }

    // z dofs
    for (int i = 0; i < dof; i++) {
        B(i + 2 * dof, 0) = 0.0;
        B(i + 2 * dof, 1) = 0.0;
        B(i + 2 * dof, 2) = DS(i, 2);
        B(i + 2 * dof, 3) = DS(i, 1);
        B(i + 2 * dof, 4) = DS(i, 0);
        B(i + 2 * dof, 5) = 0.0;
    }
}

inline
void
GenerateGradBarMatrix(const mfem::DenseMatrix& DS, const mfem::DenseMatrix& eDS, mfem::DenseMatrix& B)
{
    int dof = DS.Height();

    for (int i = 0; i < dof; i++) {
        const double B1 = (eDS(i, 0) - DS(i, 0)) / 3.0;
        B(i, 0) = B1 + DS(i, 0);
        B(i, 1) = B1;
        B(i, 2) = B1;
        B(i, 3) = 0.0;
        B(i, 4) = DS(i, 2);
        B(i, 5) = DS(i, 1);
    }

    // y dofs
    for (int i = 0; i < dof; i++) {
        const double B2 = (eDS(i, 1) - DS(i, 1)) / 3.0;
        B(i + dof, 0) = B2;
        B(i + dof, 1) = B2 + DS(i, 1);
        B(i + dof, 2) = B2;
        B(i + dof, 3) = DS(i, 2);
        B(i + dof, 4) = 0.0;
        B(i + dof, 5) = DS(i, 0);
    }

    // z dofs
    for (int i = 0; i < dof; i++) {
        const double B3 = (eDS(i, 2) - DS(i, 2)) / 3.0;
        B(i + 2 * dof, 0) = B3;
        B(i + 2 * dof, 1) = B3;
        B(i + 2 * dof, 2) = B3 + DS(i, 2);
        B(i + 2 * dof, 3) = DS(i, 1);
        B(i + 2 * dof, 4) = DS(i, 0);
        B(i + 2 * dof, 5) = 0.0;
    }
}

inline
void
GenerateGradGeomMatrix(const mfem::DenseMatrix& DS, mfem::DenseMatrix& Bgeom)
{
    int dof = DS.Height();
    // For a 3D mesh Bgeom has the following shape:
    // [DS(i, 0), 0, 0]
    // [DS(i, 0), 0, 0]
    // [DS(i, 0), 0, 0]
    // [0, DS(i, 1), 0]
    // [0, DS(i, 1), 0]
    // [0, DS(i, 1), 0]
    // [0, 0, DS(i, 2)]
    // [0, 0, DS(i, 2)]
    // [0, 0, DS(i, 2)]
    // We'll be returning the transpose of this.
    // It turns out the Bilinear operator can't have this created using
    // the dense gradient matrix, DS.
    // It can be used in the following: Bgeom^T Sigma_bar Bgeom
    // where Sigma_bar is a block diagonal version of sigma repeated 3 times in 3D.

    // I'm assumming we're in 3D and have just unrolled the loop
    // The ordering has now changed such that Bgeom matches up with mfem's internal
    // ordering of vectors such that it's [x0...xn, y0...yn, z0...zn] ordering

    // The previous single loop has been split into 3 so the B matrix
    // is constructed in chunks now instead of performing multiple striding
    // operations in a single loop.

    // x dofs
    for (int i = 0; i < dof; i++) {
        Bgeom(i, 0) = DS(i, 0);
        Bgeom(i, 1) = DS(i, 1);
        Bgeom(i, 2) = DS(i, 2);
        Bgeom(i, 3) = 0.0;
        Bgeom(i, 4) = 0.0;
        Bgeom(i, 5) = 0.0;
        Bgeom(i, 6) = 0.0;
        Bgeom(i, 7) = 0.0;
        Bgeom(i, 8) = 0.0;
    }

    // y dofs
    for (int i = 0; i < dof; i++) {
        Bgeom(i + dof, 0) = 0.0;
        Bgeom(i + dof, 1) = 0.0;
        Bgeom(i + dof, 2) = 0.0;
        Bgeom(i + dof, 3) = DS(i, 0);
        Bgeom(i + dof, 4) = DS(i, 1);
        Bgeom(i + dof, 5) = DS(i, 2);
        Bgeom(i + dof, 6) = 0.0;
        Bgeom(i + dof, 7) = 0.0;
        Bgeom(i + dof, 8) = 0.0;
    }

    // z dofs
    for (int i = 0; i < dof; i++) {
        Bgeom(i + 2 * dof, 0) = 0.0;
        Bgeom(i + 2 * dof, 1) = 0.0;
        Bgeom(i + 2 * dof, 2) = 0.0;
        Bgeom(i + 2 * dof, 3) = 0.0;
        Bgeom(i + 2 * dof, 4) = 0.0;
        Bgeom(i + 2 * dof, 5) = 0.0;
        Bgeom(i + 2 * dof, 6) = DS(i, 0);
        Bgeom(i + 2 * dof, 7) = DS(i, 1);
        Bgeom(i + 2 * dof, 8) = DS(i, 2);
    }
}

inline
void
GetQFData(const int elID, const int ipNum, double* qfdata, std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf)
{   
    const auto data = qf->HostRead();
    const int qf_offset = qf->GetVDim();
    auto qspace = qf->GetSpaceShared();
 
    const mfem::IntegrationRule *ir = &(qf->GetSpaceShared()->GetIntRule(elID));
    int elem_offset = qf_offset * ir->GetNPoints();
 
    for (int i = 0; i < qf_offset; ++i) {
        qfdata[i] = data[elID * elem_offset + ipNum * qf_offset + i];
    }
}

inline
void
SetQFData(const int elID, const int ipNum, double* qfdata, std::shared_ptr<mfem::expt::PartialQuadratureFunction> qf)
{   
    auto data = qf->HostReadWrite();
    const int qf_offset = qf->GetVDim();
    auto qspace = qf->GetSpaceShared();
 
    const mfem::IntegrationRule *ir = &(qf->GetSpaceShared()->GetIntRule(elID));
    int elem_offset = qf_offset * ir->GetNPoints();
 
    for (int i = 0; i < qf_offset; ++i) {
        data[elID * elem_offset + ipNum * qf_offset + i] = qfdata[i];
    }
}

inline
void
TransformMatGradTo4D(const std::shared_ptr<mfem::expt::PartialQuadratureFunction> matGrad, mfem::Vector& matGradPA)
{
    const int npts = matGrad->Size() / matGrad->GetVDim();
 
    const int dim = 3;
    const int dim2 = 6;
 
    const int DIM5 = 5;
    const int DIM3 = 3;
    std::array<RAJA::idx_t, DIM5> perm5 {{ 4, 3, 2, 1, 0 } };
    std::array<RAJA::idx_t, DIM3> perm3 {{ 2, 1, 0 } };
 
    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
    RAJA::Layout<DIM5> layout_4Dtensor = RAJA::make_permuted_layout({{ dim, dim, dim, dim, npts } }, perm5);
    RAJA::View<double, RAJA::Layout<DIM5, RAJA::Index_type, 0> > cmat_4d(matGradPA.ReadWrite(), layout_4Dtensor);
 
    // bunch of helper RAJA views to make dealing with data easier down below in our kernel.
    RAJA::Layout<DIM3> layout_2Dtensor = RAJA::make_permuted_layout({{ dim2, dim2, npts } }, perm3);
    RAJA::View<const double, RAJA::Layout<DIM3, RAJA::Index_type, 0> > cmat(matGrad->Read(), layout_2Dtensor);
 
    // This sets up our 4D tensor to be the same as the 2D tensor which takes advantage of symmetry operations
    mfem::forall(npts, [=] MFEM_HOST_DEVICE (int i) {
       cmat_4d(0, 0, 0, 0, i) = cmat(0, 0, i);
       cmat_4d(1, 1, 0, 0, i) = cmat(1, 0, i);
       cmat_4d(2, 2, 0, 0, i) = cmat(2, 0, i);
       cmat_4d(1, 2, 0, 0, i) = cmat(3, 0, i);
       cmat_4d(2, 1, 0, 0, i) = cmat_4d(1, 2, 0, 0, i);
       cmat_4d(2, 0, 0, 0, i) = cmat(4, 0, i);
       cmat_4d(0, 2, 0, 0, i) = cmat_4d(2, 0, 0, 0, i);
       cmat_4d(0, 1, 0, 0, i) = cmat(5, 0, i);
       cmat_4d(1, 0, 0, 0, i) = cmat_4d(0, 1, 0, 0, i);
 
       cmat_4d(0, 0, 1, 1, i) = cmat(0, 1, i);
       cmat_4d(1, 1, 1, 1, i) = cmat(1, 1, i);
       cmat_4d(2, 2, 1, 1, i) = cmat(2, 1, i);
       cmat_4d(1, 2, 1, 1, i) = cmat(3, 1, i);
       cmat_4d(2, 1, 1, 1, i) = cmat_4d(1, 2, 1, 1, i);
       cmat_4d(2, 0, 1, 1, i) = cmat(4, 1, i);
       cmat_4d(0, 2, 1, 1, i) = cmat_4d(2, 0, 1, 1, i);
       cmat_4d(0, 1, 1, 1, i) = cmat(5, 1, i);
       cmat_4d(1, 0, 1, 1, i) = cmat_4d(0, 1, 1, 1, i);
 
       cmat_4d(0, 0, 2, 2, i) = cmat(0, 2, i);
       cmat_4d(1, 1, 2, 2, i) = cmat(1, 2, i);
       cmat_4d(2, 2, 2, 2, i) = cmat(2, 2, i);
       cmat_4d(1, 2, 2, 2, i) = cmat(3, 2, i);
       cmat_4d(2, 1, 2, 2, i) = cmat_4d(1, 2, 2, 2, i);
       cmat_4d(2, 0, 2, 2, i) = cmat(4, 2, i);
       cmat_4d(0, 2, 2, 2, i) = cmat_4d(2, 0, 2, 2, i);
       cmat_4d(0, 1, 2, 2, i) = cmat(5, 2, i);
       cmat_4d(1, 0, 2, 2, i) = cmat_4d(0, 1, 2, 2, i);
 
       cmat_4d(0, 0, 1, 2, i) = cmat(0, 3, i);
       cmat_4d(1, 1, 1, 2, i) = cmat(1, 3, i);
       cmat_4d(2, 2, 1, 2, i) = cmat(2, 3, i);
       cmat_4d(1, 2, 1, 2, i) = cmat(3, 3, i);
       cmat_4d(2, 1, 1, 2, i) = cmat_4d(1, 2, 1, 2, i);
       cmat_4d(2, 0, 1, 2, i) = cmat(4, 3, i);
       cmat_4d(0, 2, 1, 2, i) = cmat_4d(2, 0, 1, 2, i);
       cmat_4d(0, 1, 1, 2, i) = cmat(5, 3, i);
       cmat_4d(1, 0, 1, 2, i) = cmat_4d(0, 1, 1, 2, i);
 
       cmat_4d(0, 0, 2, 1, i) = cmat(0, 3, i);
       cmat_4d(1, 1, 2, 1, i) = cmat(1, 3, i);
       cmat_4d(2, 2, 2, 1, i) = cmat(2, 3, i);
       cmat_4d(1, 2, 2, 1, i) = cmat(3, 3, i);
       cmat_4d(2, 1, 2, 1, i) = cmat_4d(1, 2, 1, 2, i);
       cmat_4d(2, 0, 2, 1, i) = cmat(4, 3, i);
       cmat_4d(0, 2, 2, 1, i) = cmat_4d(2, 0, 1, 2, i);
       cmat_4d(0, 1, 2, 1, i) = cmat(5, 3, i);
       cmat_4d(1, 0, 2, 1, i) = cmat_4d(0, 1, 1, 2, i);
 
       cmat_4d(0, 0, 2, 0, i) = cmat(0, 4, i);
       cmat_4d(1, 1, 2, 0, i) = cmat(1, 4, i);
       cmat_4d(2, 2, 2, 0, i) = cmat(2, 4, i);
       cmat_4d(1, 2, 2, 0, i) = cmat(3, 4, i);
       cmat_4d(2, 1, 2, 0, i) = cmat_4d(1, 2, 2, 0, i);
       cmat_4d(2, 0, 2, 0, i) = cmat(4, 4, i);
       cmat_4d(0, 2, 2, 0, i) = cmat_4d(2, 0, 2, 0, i);
       cmat_4d(0, 1, 2, 0, i) = cmat(5, 4, i);
       cmat_4d(1, 0, 2, 0, i) = cmat_4d(0, 1, 2, 0, i);
 
       cmat_4d(0, 0, 0, 2, i) = cmat(0, 4, i);
       cmat_4d(1, 1, 0, 2, i) = cmat(1, 4, i);
       cmat_4d(2, 2, 0, 2, i) = cmat(2, 4, i);
       cmat_4d(1, 2, 0, 2, i) = cmat(3, 4, i);
       cmat_4d(2, 1, 0, 2, i) = cmat_4d(1, 2, 2, 0, i);
       cmat_4d(2, 0, 0, 2, i) = cmat(4, 4, i);
       cmat_4d(0, 2, 0, 2, i) = cmat_4d(2, 0, 2, 0, i);
       cmat_4d(0, 1, 0, 2, i) = cmat(5, 4, i);
       cmat_4d(1, 0, 0, 2, i) = cmat_4d(0, 1, 2, 0, i);
 
       cmat_4d(0, 0, 0, 1, i) = cmat(0, 5, i);
       cmat_4d(1, 1, 0, 1, i) = cmat(1, 5, i);
       cmat_4d(2, 2, 0, 1, i) = cmat(2, 5, i);
       cmat_4d(1, 2, 0, 1, i) = cmat(3, 5, i);
       cmat_4d(2, 1, 0, 1, i) = cmat_4d(1, 2, 0, 1, i);
       cmat_4d(2, 0, 0, 1, i) = cmat(4, 5, i);
       cmat_4d(0, 2, 0, 1, i) = cmat_4d(2, 0, 0, 1, i);
       cmat_4d(0, 1, 0, 1, i) = cmat(5, 5, i);
       cmat_4d(1, 0, 0, 1, i) = cmat_4d(0, 1, 0, 1, i);
 
       cmat_4d(0, 0, 1, 0, i) = cmat(0, 5, i);
       cmat_4d(1, 1, 1, 0, i) = cmat(1, 5, i);
       cmat_4d(2, 2, 1, 0, i) = cmat(2, 5, i);
       cmat_4d(1, 2, 1, 0, i) = cmat(3, 5, i);
       cmat_4d(2, 1, 1, 0, i) = cmat_4d(1, 2, 0, 1, i);
       cmat_4d(2, 0, 1, 0, i) = cmat(4, 5, i);
       cmat_4d(0, 2, 1, 0, i) = cmat_4d(2, 0, 0, 1, i);
       cmat_4d(0, 1, 1, 0, i) = cmat(5, 5, i);
       cmat_4d(1, 0, 1, 0, i) = cmat_4d(0, 1, 0, 1, i);
    });
 }
