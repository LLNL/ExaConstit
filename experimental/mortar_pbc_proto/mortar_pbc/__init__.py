"""Mortar-method periodic boundary conditions for non-conforming RVE meshes.

This package implements the dual-basis SPS (saddle-point) variant of the
mortar method as described in:

    Lopes, I.A.R.; Ferreira, B.P.; Andrade Pires, F.M.
    "On the efficient enforcement of uniform traction and mortar periodic
     boundary conditions in computational homogenisation"
    CMAME 384 (2021) 113930.

It is a precursor / prototype for an eventual MFEM C++ implementation
that will be integrated into ExaConstit (LLNL crystal-plasticity FE code).

Phase 1 scope (this prototype)
------------------------------
    * 2D rectangular RVEs
    * H1 vector-linear elements (Q4 quadrilaterals or T3 triangles, both
      yielding line-2 elements on the interface)
    * pyMFEM ParMesh / ParFiniteElementSpace
    * Saddle-point Newton step solved by scipy.sparse.linalg.spsolve
      (gather-to-root for the K block; mortar matrices assembled
      AllGather-globally on each rank)
    * Periodic BC only (uniform traction is intentionally deferred --
      see ``constraint_builder.py`` for the extension hook)

Future phases (in order)
------------------------
    * Phase 2: heterogeneous RVE + neo-Hookean + Newton iteration
    * Phase 3: MPI -- gather-to-root first, then properly distributed
    * Phase 4: 3D (wirebaskets + Wohlmuth corner modifications, §C of paper)
    * Phase 5: MPI 3D
    * Phase 6: port to MFEM C++; integrate with ExaConstit

Module layout
-------------
    types_2d            : dataclasses (no MPI / MFEM deps)
    mortar_2d           : mortar matrix assembly (no MPI / MFEM deps)
    constraint_builder  : global C from per-edge mortar blocks
    saddle_point        : the [[K, C^T], [C, 0]] block solve
    boundary_2d         : MFEM-dependent classifier (lazy-imported)

The lazy import of ``BoundaryClassifier2D`` is deliberate: it lets the
unit tests of the dual basis and mortar matrices run in environments
where pyMFEM/mpi4py are not installed.  All ExaConstit-developer-facing
math lives in the lazy-import-safe modules.
"""

from .types_2d            import EdgeNodes2D, CornerInfo
from .types_3d            import (
    CornerInfo3D, EdgeInfo3D, FaceInfo3D,
    QuadFaceElement, TriFaceElement, FaceMortarPairBlock,
)
from .mortar_2d           import MortarAssembler2D, MortarBlock2D
from .mortar_3d           import (
    # shape functions
    N_line2 as N_line2_3d,    # alias to avoid shadowing mortar_2d.N_line2
    N_line3,
    N_tri3, N_tri6,
    N_quad4, N_quad8, N_quad9,
    N_tet4, N_tet10,
    # dual bases
    M_tri3_dual, M_quad4_dual, M_tet4_dual,
    # Wohlmuth modifications
    M_tri3_dual_modified, M_quad4_dual_modified,
    # quadrature
    gauss_line_3pt, gauss_quad_3x3, gauss_tri_3pt, gauss_tet_4pt,
    # the §4.9.1 criterion
    lumped_positivity,
)
from .face_mortar_3d      import (
    MortarFaceAssembler,
    QuadFaceMortarAssembler,
    TriFaceMortarAssembler,
    match_conforming_face_pairs,
)
from .constraint_builder  import ConstraintBuilder2D
from .constraint_assembler import (
    ConstraintAssembler,
    MortarPbcConstraintAssembler,
    stack_constraints,
)
from .saddle_point        import (
    SaddlePointSolver,
    make_constraint_operators,
    apply_dirichlet_zero_to_C,
)


# BoundaryClassifier2D and write_pbc_visualization need MPI + mfem.par;
# import them lazily so the rest of the package (including unit tests of
# dual basis and mortar matrices) can be imported without those deps.
def __getattr__(name):
    if name == "BoundaryClassifier2D":
        from .boundary_2d import BoundaryClassifier2D
        return BoundaryClassifier2D
    if name == "write_pbc_visualization":
        from .visualization import write_pbc_visualization
        return write_pbc_visualization
    if name == "PbcVisualizationWriter":
        from .visualization import PbcVisualizationWriter
        return PbcVisualizationWriter
    if name in ("MortarPbcDriver2D", "StepResult", "compute_volume_averaged_F"):
        from .multistep_driver import (
            MortarPbcDriver2D,
            StepResult,
            compute_volume_averaged_F,
        )
        return locals()[name]
    if name in (
        "assemble_linear_elastic_K_hypre",
        "apply_linear_part",
        "find_corners_3d",
        "apply_dirichlet_to_distributed_K",
        "newton_residual_at_u_lin",
        "collect_corner_tdofs",
        "find_all_boundary_tdofs",
        "collect_boundary_tdof_values",
    ):
        from .elastic_3d import (
            assemble_linear_elastic_K_hypre,
            apply_linear_part,
            find_corners_3d,
            apply_dirichlet_to_distributed_K,
            newton_residual_at_u_lin,
            collect_corner_tdofs,
            find_all_boundary_tdofs,
            collect_boundary_tdof_values,
        )
        return locals()[name]
    if name == "BoundaryClassifier3D":
        from .boundary_3d import BoundaryClassifier3D
        return BoundaryClassifier3D
    if name == "ConstraintBuilder3D":
        from .constraint_builder_3d import ConstraintBuilder3D
        return ConstraintBuilder3D
    raise AttributeError(f"module 'mortar_pbc' has no attribute {name!r}")


__all__ = [
    # Lazy import (MFEM-dependent)
    "BoundaryClassifier2D",
    "write_pbc_visualization",
    "PbcVisualizationWriter",
    "MortarPbcDriver2D",
    "StepResult",
    "compute_volume_averaged_F",
    # Lazy import: 3D linear-elastic + Dirichlet (Phase 3.1+)
    "assemble_linear_elastic_K_hypre",
    "apply_linear_part",
    "find_corners_3d",
    "apply_dirichlet_to_distributed_K",
    "newton_residual_at_u_lin",
    "collect_corner_tdofs",
    "find_all_boundary_tdofs",
    "collect_boundary_tdof_values",
    # Lazy import: 3D boundary classifier (Phase 3.3.B+)
    "BoundaryClassifier3D",
    # Lazy import: 3D constraint builder (Phase 3.3.C+)
    "ConstraintBuilder3D",
    # Pure-Python data
    "EdgeNodes2D",
    "CornerInfo",
    "CornerInfo3D",
    "EdgeInfo3D",
    "FaceInfo3D",
    "QuadFaceElement",
    "TriFaceElement",
    "FaceMortarPairBlock",
    # Mortar machinery (2D)
    "MortarAssembler2D",
    "MortarBlock2D",
    "ConstraintBuilder2D",
    # Mortar machinery (3D, Phase 3.2.A)
    "N_line2_3d", "N_line3",
    "N_tri3", "N_tri6",
    "N_quad4", "N_quad8", "N_quad9",
    "N_tet4", "N_tet10",
    "M_tri3_dual", "M_quad4_dual", "M_tet4_dual",
    "M_tri3_dual_modified", "M_quad4_dual_modified",
    "gauss_line_3pt", "gauss_quad_3x3", "gauss_tri_3pt", "gauss_tet_4pt",
    "lumped_positivity",
    # Face-mortar assembler (3D, Phase 3.2.B)
    "MortarFaceAssembler",
    "QuadFaceMortarAssembler",
    "TriFaceMortarAssembler",
    "match_conforming_face_pairs",
    # Constraint-assembly interface (extension point for future UT)
    "ConstraintAssembler",
    "MortarPbcConstraintAssembler",
    "stack_constraints",
    # Solver (distributed Krylov)
    "SaddlePointSolver",
    "make_constraint_operators",
    "apply_dirichlet_zero_to_C",
]
