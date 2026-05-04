"""3D face-mortar assembler — Phase 3.2.B of the architecture doc.

WHAT
----
Three things, in dependency order:

1. ``MortarFaceAssembler`` — abstract base class (ABC) holding the
   element-pair assembly LOOP that is element-type-agnostic.
2. ``QuadFaceMortarAssembler`` and ``TriFaceMortarAssembler`` — concrete
   subclasses providing the per-element-type kernels (shape-function
   evaluation, dual-basis evaluation, reference-element quadrature,
   Jacobian).
3. ``match_conforming_face_pairs`` — pure-Python helper that for each
   nonmortar face element finds its 1:1 conforming mortar partner by
   parametric centroid + tolerance match. The result is consumed by
   ``MortarFaceAssembler.assemble_pair_conforming``.

This is the 3D analog of ``mortar_2d.MortarAssembler2D``. The 2D version
operates on 1D edge elements with 1D parametric overlap; the 3D version
operates on 2D face elements with 2D parametric overlap. Phase 3.2.B
covers only the *conforming* case (1:1 element pairing); Phase 3.5 will
add a non-conforming Sutherland-Hodgman polygon-clipping path that
slots into the same ABC via an alternative ``assemble_pair_clipped``
method.

WHY
---
This layer bridges the per-element dual bases (Phase 3.2.A,
``mortar_3d.py``) and the global constraint matrix builder (Phase 3.3,
``constraint_builder_3d.py``). It is pure-Python (no MFEM dependency)
so unit-testable from synthetic face-element data — the same separation
of concerns that has worked for 2D since Phase 1.

WHO CALLS WHOM
--------------
    BoundaryClassifier3D        -->  list of QuadFaceElement / TriFaceElement
                                       per face (one list per face)
    match_conforming_face_pairs -->  list of (nonmortar_idx, mortar_idx, perm)
    *FaceMortarAssembler        -->  FaceMortarPairBlock (D, A_m, gtdofs)
    ConstraintBuilder3D         -->  global C HypreParMatrix

DESIGN NOTES
------------
* The ABC contains the LOOP; subclasses contain the KERNELS. This
  matches ``MortarAssembler2D`` (single class, line-2-specific kernels
  inlined) but generalises naturally to multiple element types in 3D.
  In particular, mixed hex+tet faces (§11.4) require two distinct
  assembler instances at the ConstraintBuilder3D level — one for the
  quad-4 sub-elements and one for the tri-3 sub-elements — combined
  via row stacking before final C build.

* Sentinel-row drop: per the §5.4 wirebasket hierarchy, nonmortar face
  elements with corner-DOF (gtdof = -1) or edge-DOF (gtdof = -2)
  entries have those rows dropped from D and A_m. Likewise mortar-side
  sentinels drop their columns. This matches
  ``MortarAssembler2D._integrate_overlap_segment`` lines 396-414.

* Lumped-positivity guard: the assembler's __init__ runs
  ``lumped_positivity()`` against its own ``_eval_nonmortar_shape`` on the
  reference element and raises ``RuntimeError`` if any s_j ≤ tol. This
  catches misuse if a higher-order element type is plugged in without
  a proper §4.10 basis-transformation. Per §4.9.1 of the architecture
  doc.

* Dual-basis modification dispatch: the nonmortar element's
  ``boundary_tag`` field is translated into the right modifier-arg
  combination by the subclass-specific ``_dual_modifier_args`` helper.

REFERENCES
----------
* MORTAR_PBC_ARCHITECTURE.md §11.6 (face-mortar geometric matching).
* MORTAR_PBC_ARCHITECTURE.md §11.8 Phase 3.2.B (this phase).
* MORTAR_PBC_ARCHITECTURE.md §4.9.1 (lumped-positivity criterion).
* MORTAR_PBC_ARCHITECTURE.md §5 (Wohlmuth modifications, used here).
* mortar_pbc/mortar_2d.py (the 2D pattern this generalises).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Sequence, Tuple

import numpy as np

from .mortar_3d import (
    M_quad4_dual_modified,
    M_tri3_dual_modified,
    N_quad4,
    N_tri3,
    gauss_quad_3x3,
    gauss_tri_3pt,
    lumped_positivity,
)
from .types_3d import (
    FaceMortarPairBlock,
    QuadFaceElement,
    TriFaceElement,
)


__all__ = [
    "MortarFaceAssembler",
    "QuadFaceMortarAssembler",
    "TriFaceMortarAssembler",
    "match_conforming_face_pairs",
]


# =============================================================================
# Lumped-positivity tolerance for the construction guard
# =============================================================================
#
# Per §4.9.1, strict bi-orthogonal locally-supported dual exists iff
# every shape-function lumped integral s_j > 0. Our quadrature on the
# reference element should reproduce these to machine precision; we
# allow a tolerance of 1e-12 to account for floating-point round-off
# but not to mask any genuine sign issues.
_LUMPED_POSITIVITY_TOL: float = 1e-12


# =============================================================================
# Abstract base: per-element-type assembler
# =============================================================================

class MortarFaceAssembler(ABC):
    """Abstract base class for face-mortar block assembly.

    Subclasses provide element-type-specific kernels (quad-4 or tri-3);
    the loop driver and sentinel-handling are defined here.

    Phase 3.2.B scope: ``assemble_pair_conforming`` only — the nonmortar and
    mortar meshes are assumed conforming (1:1 element pairing on the
    periodic face pair). Non-conforming geometric matching (Sutherland-
    Hodgman) is Phase 3.5; it will add ``assemble_pair_clipped`` that
    re-uses the same kernels.

    Parameters
    ----------
    quadrature_order : int, default 4
        Reference-element quadrature degree. Default is exact for
        polynomial integrands of degree ≤ 4 (sufficient for bilinear
        nonmortar × bilinear mortar = degree 2-per-direction = degree 4
        product, plus margin).

    Attributes
    ----------
    _qpts : (Nq, dim) ndarray
        Reference-element quadrature points. dim = 2 for face elements.
    _qwts : (Nq,) ndarray
        Reference-element quadrature weights.
    """

    def __init__(self, *, quadrature_order: int = 4) -> None:
        self.quadrature_order = quadrature_order
        self._qpts, self._qwts = self._build_quadrature(quadrature_order)
        # Lumped-positivity construction guard (§4.9.1).
        self._verify_lumped_positivity()

    # ------------------------------------------------------------ subclass API
    @abstractmethod
    def _eval_nonmortar_dual(
        self, q_pt: np.ndarray, boundary_tag: str,
    ) -> np.ndarray:
        """Evaluate the (possibly modified) nonmortar-side dual basis.

        Parameters
        ----------
        q_pt : (dim,) ndarray
            Reference-element quadrature point on the nonmortar element.
        boundary_tag : str
            Nonmortar element's boundary tag — selects modification.

        Returns
        -------
        (n_nodes,) ndarray of M_i values.
        """
        ...

    @abstractmethod
    def _eval_nonmortar_shape(self, q_pt: np.ndarray) -> np.ndarray:
        """Evaluate the standard (unmodified) nonmortar-side shape functions.

        Used to construct ``D = ∫ N^nonmortar dA``. Same sample location
        as ``_eval_nonmortar_dual``.
        """
        ...

    @abstractmethod
    def _eval_mortar_shape(self, q_pt_mortar: np.ndarray) -> np.ndarray:
        """Evaluate the standard mortar-side shape functions.

        Parameters
        ----------
        q_pt_mortar : (dim,) ndarray
            Reference-element coords on the *mortar* element. For
            conforming matched pairs with same orientation, this is
            identical to the nonmortar-side q_pt.
        """
        ...

    @abstractmethod
    def _build_quadrature(
        self, order: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return reference-element quadrature points and weights."""
        ...

    @abstractmethod
    def _nonmortar_jacobian(self, nonmortar_elem) -> Callable[[np.ndarray], float]:
        """Return a function ``J(q_pt) -> float`` giving |J| at the point.

        For axis-aligned face elements the Jacobian is constant and
        the closure simply returns that value. For non-axis-aligned
        bilinear quads the Jacobian varies and the returned closure
        does the per-point computation.
        """
        ...

    @abstractmethod
    def _n_nodes_per_elem(self) -> int:
        """Number of nodes per element of the kind this assembler handles."""
        ...

    @abstractmethod
    def _n_basis_for_lumped_check(self) -> int:
        """Number of shape functions for the lumped-positivity guard."""
        ...

    @abstractmethod
    def _shape_for_lumped_check(self) -> Callable:
        """Reference shape-function callable for the lumped-positivity guard."""
        ...

    @abstractmethod
    def _ref_quad_for_lumped_check(self) -> Tuple[np.ndarray, np.ndarray]:
        """Quadrature pts / wts for the lumped-positivity guard."""
        ...

    @abstractmethod
    def _mortar_node_permutation_apply(
        self, mortar_node_perm: Sequence[int], q_pt_nonmortar: np.ndarray,
    ) -> np.ndarray:
        """Map a nonmortar-side q_pt to the mortar-side q_pt under a permutation.

        For ``mortar_node_perm = identity`` (typical axis-aligned RVE),
        this is the identity. For permuted/reflected pairings, it
        applies the corresponding affine reference-element map.
        """
        ...

    # ------------------------------------------------------------ helpers
    def _verify_lumped_positivity(self) -> None:
        """Phase 3.2.B construction guard — see §4.9.1.

        Computes s_j = int N_j on the reference element via the
        subclass-supplied quadrature, and raises if any s_j is
        non-positive. This catches misinstantiation (e.g. plugging in
        a tri-6 dual basis without the §4.10 transformation).
        """
        N_func = self._shape_for_lumped_check()
        n_basis = self._n_basis_for_lumped_check()
        qpts, qwts = self._ref_quad_for_lumped_check()
        # Most simplex shape callables in mortar_3d use the
        # tuple-input convention (e.g. N_tri3 takes (l1, l2, l3));
        # tensor-product callables take separate args. The subclass
        # opts in via the calling convention.
        s = lumped_positivity(
            N_func, qpts, qwts, n_basis,
            use_tuple_input=self._lumped_uses_tuple_input(),
        )
        if np.any(s <= _LUMPED_POSITIVITY_TOL):
            raise RuntimeError(
                f"{self.__class__.__name__}: lumped-positivity check failed "
                f"(s = {s}). Per §4.9.1 of the architecture doc, the strict "
                f"bi-orthogonal dual basis does not exist for this element "
                f"type. Use the §4.10 basis-transformation procedure or the "
                f"§4.11 LOR fallback."
            )

    def _lumped_uses_tuple_input(self) -> bool:
        """Whether the lumped-check shape callable takes a tuple or *args.

        Default: True (simplex shape functions in mortar_3d.py take a
        barycentric tuple). Tensor-product subclasses override to
        False.
        """
        return True

    # ------------------------------------------------------------ public API
    def assemble_pair_conforming(
        self,
        nonmortar_elems: Sequence,
        mortar_elems: Sequence,
        pair_matches: Sequence[Tuple[int, int, Tuple[int, ...]]],
        nonmortar_face_name: str = "nonmortar",
        mortar_face_name: str = "mortar",
    ) -> FaceMortarPairBlock:
        """Assemble (D, A_m) for a conforming face pair.

        Parameters
        ----------
        nonmortar_elems : sequence of QuadFaceElement or TriFaceElement
            All nonmortar-side face elements (caller has filtered to the
            element type this assembler handles).
        mortar_elems : sequence of QuadFaceElement or TriFaceElement
            All mortar-side face elements, same kind.
        pair_matches : list of (nonmortar_idx, mortar_idx, mortar_node_perm)
            One entry per nonmortar element. ``mortar_node_perm`` is a
            permutation of (0, 1, ..., n_nodes-1) telling how the
            mortar-element local nodes correspond to the nonmortar element's
            local nodes. For axis-aligned MakeCartesian3D meshes the
            permutation is the identity.
        nonmortar_face_name, mortar_face_name : str
            Labels for the resulting ``FaceMortarPairBlock``.

        Returns
        -------
        FaceMortarPairBlock with row indexing by *kept* nonmortar gtdofs
        and column indexing by *kept* mortar gtdofs (sentinels dropped).
        """
        # First pass: discover the kept-row / kept-col gtdof sets.
        nonmortar_gtdofs_kept, nonmortar_row_of = self._discover_kept_gtdofs(nonmortar_elems)
        mortar_gtdofs_kept, mortar_col_of = self._discover_kept_gtdofs(mortar_elems)

        n_rows = len(nonmortar_gtdofs_kept)
        n_cols = len(mortar_gtdofs_kept)
        D_full = np.zeros(n_rows, dtype=np.float64)
        A_m = np.zeros((n_rows, n_cols), dtype=np.float64)

        # Second pass: integrate per matched pair.
        for nonmortar_idx, mortar_idx, mortar_node_perm in pair_matches:
            s_elem = nonmortar_elems[nonmortar_idx]
            m_elem = mortar_elems[mortar_idx]
            self._integrate_pair(
                D_full, A_m,
                nonmortar_elem=s_elem, mortar_elem=m_elem,
                mortar_node_perm=mortar_node_perm,
                nonmortar_row_of=nonmortar_row_of,
                mortar_col_of=mortar_col_of,
            )

        return FaceMortarPairBlock(
            A_m=A_m,
            D=D_full,
            nonmortar_face_name=nonmortar_face_name,
            mortar_face_name=mortar_face_name,
            nonmortar_gtdofs=np.asarray(nonmortar_gtdofs_kept, dtype=np.int64),
            mortar_gtdofs=np.asarray(mortar_gtdofs_kept, dtype=np.int64),
        )

    # ------------------------------------------------------------ internals
    @staticmethod
    def _discover_kept_gtdofs(
        elems: Sequence,
    ) -> Tuple[List[int], dict]:
        """Walk the elements, gathering the sorted list of unique kept gtdofs.

        Sentinels (gtdof < 0) are dropped. Returns:
            * sorted list of unique kept gtdofs
            * dict mapping gtdof -> row/col index in that sorted list
        """
        seen = set()
        ordered: List[int] = []
        for e in elems:
            for g in e.gtdofs:
                if g < 0:
                    continue
                if g in seen:
                    continue
                seen.add(g)
                ordered.append(g)
        ordered.sort()
        idx_of = {g: i for i, g in enumerate(ordered)}
        return ordered, idx_of

    def _integrate_pair(
        self,
        D_full: np.ndarray,
        A_m: np.ndarray,
        *,
        nonmortar_elem,
        mortar_elem,
        mortar_node_perm: Sequence[int],
        nonmortar_row_of: dict,
        mortar_col_of: dict,
    ) -> None:
        """Integrate one matched (nonmortar, mortar) element pair into D, A_m.

        Conforming-pair shortcut: the mortar-side q_pt equals the
        nonmortar-side q_pt under the mortar_node_perm map. Integration is
        on the nonmortar reference element's quadrature with the mortar
        shape evaluated at the permuted reference coord.
        """
        boundary_tag = getattr(nonmortar_elem, "boundary_tag", "none")
        nonmortar_J_fn = self._nonmortar_jacobian(nonmortar_elem)

        n_loc = self._n_nodes_per_elem()
        # Per-element local D and A_m, before sentinel-aware accumulation.
        D_loc = np.zeros(n_loc, dtype=np.float64)
        A_loc = np.zeros((n_loc, n_loc), dtype=np.float64)

        for q in range(self._qpts.shape[0]):
            q_pt = self._qpts[q]
            w_q = float(self._qwts[q])
            J = float(nonmortar_J_fn(q_pt))
            phys_w = w_q * J

            # Nonmortar-side dual (modified per boundary_tag) and standard shape.
            M_nonmortar = self._eval_nonmortar_dual(q_pt, boundary_tag)
            N_nonmortar = self._eval_nonmortar_shape(q_pt)
            # Mortar-side coords under the matched-pair permutation, shape there.
            q_pt_mortar = self._mortar_node_permutation_apply(mortar_node_perm, q_pt)
            N_mortar = self._eval_mortar_shape(q_pt_mortar)
            # When mortar_node_perm is non-identity, the mortar shape
            # values at the *permuted* point need to be re-ordered to
            # match the mortar-element's local-node convention; we
            # apply the inverse permutation on the shape values.
            N_mortar_in_mortar_local = self._reorder_mortar_shape(
                N_mortar, mortar_node_perm,
            )

            # D_loc[k] += phys_w * N_nonmortar[k]
            D_loc += phys_w * N_nonmortar
            # A_loc[k, l] += phys_w * M_nonmortar[k] * N_mortar[l]
            A_loc += phys_w * np.outer(M_nonmortar, N_mortar_in_mortar_local)

        # Now scatter into the global D and A_m, dropping sentinel rows/cols.
        for k_loc in range(n_loc):
            g_nonmortar = nonmortar_elem.gtdofs[k_loc]
            if g_nonmortar < 0:
                continue
            k_global = nonmortar_row_of[g_nonmortar]
            D_full[k_global] += D_loc[k_loc]
            for l_loc in range(n_loc):
                g_mortar = mortar_elem.gtdofs[l_loc]
                if g_mortar < 0:
                    continue
                l_global = mortar_col_of[g_mortar]
                A_m[k_global, l_global] += A_loc[k_loc, l_loc]

    @staticmethod
    def _reorder_mortar_shape(
        N_mortar_at_q: np.ndarray, mortar_node_perm: Sequence[int],
    ) -> np.ndarray:
        """Reorder mortar-shape values to match mortar-element local-node order.

        ``mortar_node_perm[i]`` = index in mortar-element local-node
        order of the mortar shape function that lives at *nonmortar-element*
        local-node i. Applying the inverse permutation to N_mortar
        therefore lines up mortar shape values with mortar-element
        local-node order, which matches `mortar_elem.gtdofs[l_loc]`
        in the scatter loop.

        For ``mortar_node_perm = identity = (0, 1, ..., n-1)`` (the
        common axis-aligned RVE case), this is a no-op.
        """
        if tuple(mortar_node_perm) == tuple(range(len(mortar_node_perm))):
            return N_mortar_at_q
        # Inverse permutation: where does each mortar-local-node index land.
        inv = [0] * len(mortar_node_perm)
        for nonmortar_local, mortar_local in enumerate(mortar_node_perm):
            inv[mortar_local] = nonmortar_local
        return np.asarray([N_mortar_at_q[i] for i in inv], dtype=np.float64)


# =============================================================================
# Concrete: quad-4 face mortar
# =============================================================================

class QuadFaceMortarAssembler(MortarFaceAssembler):
    """Quad-4 face-mortar assembler.

    Uses ``M_quad4_dual_modified`` and ``N_quad4`` as kernels;
    reference quadrature is 3×3 Gauss-Legendre on [-1, +1]^2 (degree
    5 each direction, exact for quartic integrands).
    """

    # ----------------------------------------------------------- constants
    @staticmethod
    def _quad4_boundary_tag_to_sides(boundary_tag: str) -> Tuple[str, str]:
        """Map a QuadFaceElement.boundary_tag to (side_xi, side_eta).

        Tag conventions (matched against types_3d.QuadFaceElement docstring):
            "none"            -> ("none", "none")
            "edge-xi-low"     -> ("left",  "none")
            "edge-xi-high"    -> ("right", "none")
            "edge-eta-low"    -> ("none",  "bottom")
            "edge-eta-high"   -> ("none",  "top")
            "corner-LL"       -> ("left",  "bottom")
            "corner-LR"       -> ("right", "bottom")
            "corner-UL"       -> ("left",  "top")
            "corner-UR"       -> ("right", "top")
        """
        mapping = {
            "none":            ("none",  "none"),
            "edge-xi-low":     ("left",  "none"),
            "edge-xi-high":    ("right", "none"),
            "edge-eta-low":    ("none",  "bottom"),
            "edge-eta-high":   ("none",  "top"),
            "corner-LL":       ("left",  "bottom"),
            "corner-LR":       ("right", "bottom"),
            "corner-UL":       ("left",  "top"),
            "corner-UR":       ("right", "top"),
        }
        if boundary_tag not in mapping:
            raise ValueError(
                f"QuadFaceMortarAssembler: unrecognised boundary_tag "
                f"{boundary_tag!r}. Expected one of {list(mapping.keys())!r}."
            )
        return mapping[boundary_tag]

    # ----------------------------------------------------------- subclass API
    def _eval_nonmortar_dual(
        self, q_pt: np.ndarray, boundary_tag: str,
    ) -> np.ndarray:
        side_xi, side_eta = self._quad4_boundary_tag_to_sides(boundary_tag)
        xi, eta = float(q_pt[0]), float(q_pt[1])
        return np.asarray(
            M_quad4_dual_modified(xi, eta, side_xi=side_xi, side_eta=side_eta),
            dtype=np.float64,
        )

    def _eval_nonmortar_shape(self, q_pt: np.ndarray) -> np.ndarray:
        return np.asarray(
            N_quad4(float(q_pt[0]), float(q_pt[1])), dtype=np.float64,
        )

    def _eval_mortar_shape(self, q_pt_mortar: np.ndarray) -> np.ndarray:
        return np.asarray(
            N_quad4(float(q_pt_mortar[0]), float(q_pt_mortar[1])),
            dtype=np.float64,
        )

    def _build_quadrature(
        self, order: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 3x3 Gauss-Legendre is degree 5 each direction (exact for any
        # bilinear-bilinear product). Higher-order quads can swap in
        # different rules later.
        return gauss_quad_3x3()

    def _nonmortar_jacobian(self, nonmortar_elem) -> Callable[[np.ndarray], float]:
        # For axis-aligned quad-4 face elements (the RVE case), the
        # Jacobian is constant. The dataclass property handles it; we
        # close over the precomputed value.
        J_const = nonmortar_elem.jacobian_axis_aligned
        if not np.isnan(J_const):
            return lambda q_pt, _J=J_const: _J
        # Non-axis-aligned: bilinear quad Jacobian per point.
        # Restrict to the two parametric axes for the Jacobian
        # determinant (the third axis is constant on the face).
        axis_idx = {"x": 0, "y": 1, "z": 2}
        a_idx = axis_idx[nonmortar_elem.parametric_axes[0]]
        b_idx = axis_idx[nonmortar_elem.parametric_axes[1]]
        # Local-node reference positions for quad-4.
        ref = np.asarray([
            [-1.0, -1.0],
            [+1.0, -1.0],
            [+1.0, +1.0],
            [-1.0, +1.0],
        ])
        coords_2d = nonmortar_elem.coords[:, [a_idx, b_idx]]  # (4, 2)

        def J_fn(q_pt: np.ndarray) -> float:
            xi, eta = float(q_pt[0]), float(q_pt[1])
            # dN/dxi and dN/deta for quad-4.
            dN_dxi = 0.25 * np.asarray([
                -(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta),
            ])
            dN_deta = 0.25 * np.asarray([
                -(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi),
            ])
            J11 = float(dN_dxi @ coords_2d[:, 0])
            J12 = float(dN_dxi @ coords_2d[:, 1])
            J21 = float(dN_deta @ coords_2d[:, 0])
            J22 = float(dN_deta @ coords_2d[:, 1])
            return abs(J11 * J22 - J12 * J21)

        return J_fn

    def _n_nodes_per_elem(self) -> int:
        return 4

    def _n_basis_for_lumped_check(self) -> int:
        return 4

    def _shape_for_lumped_check(self) -> Callable:
        return N_quad4

    def _ref_quad_for_lumped_check(self) -> Tuple[np.ndarray, np.ndarray]:
        return gauss_quad_3x3()

    def _lumped_uses_tuple_input(self) -> bool:
        # N_quad4 takes (xi, eta) as separate args.
        return False

    def _mortar_node_permutation_apply(
        self, mortar_node_perm: Sequence[int], q_pt_nonmortar: np.ndarray,
    ) -> np.ndarray:
        """For Phase 3.2.B conforming-pair, identity permutation = identity map.

        Non-identity quad-4 permutations (rotations / reflections) map
        to corresponding affine maps on (xi, eta). Implemented as a
        small lookup table: for the 8 dihedral-group permutations of a
        quad's 4 corners, the corresponding (xi, eta) -> (xi', eta')
        is a sign-flip / swap.
        """
        if tuple(mortar_node_perm) == (0, 1, 2, 3):
            return q_pt_nonmortar
        # Other permutations: solve for the affine map by examining
        # where local node 0 (-1, -1) and local node 1 (+1, -1) of the
        # nonmortar land in mortar local coords.
        ref_quad4 = np.asarray([
            [-1.0, -1.0],
            [+1.0, -1.0],
            [+1.0, +1.0],
            [-1.0, +1.0],
        ])
        # mortar_node_perm[i] = mortar-local index of the mortar node
        # that is geometrically at nonmortar-local node i.
        # Mortar local coords of node-0-of-nonmortar and node-1-of-nonmortar:
        mortar_at_nonmortar_0 = ref_quad4[mortar_node_perm[0]]
        mortar_at_nonmortar_1 = ref_quad4[mortar_node_perm[1]]
        mortar_at_nonmortar_3 = ref_quad4[mortar_node_perm[3]]
        # The affine map sends nonmortar (-1,-1) -> mortar_at_nonmortar_0,
        # (+1,-1) -> mortar_at_nonmortar_1, (-1,+1) -> mortar_at_nonmortar_3.
        # Two basis vectors in mortar local coords:
        e_xi  = 0.5 * (mortar_at_nonmortar_1 - mortar_at_nonmortar_0)
        e_eta = 0.5 * (mortar_at_nonmortar_3 - mortar_at_nonmortar_0)
        origin = 0.5 * (mortar_at_nonmortar_0 + mortar_at_nonmortar_1) + 0.5 * (
            mortar_at_nonmortar_3 - mortar_at_nonmortar_0
        )
        # We don't actually need the origin here because the affine map
        # is uniquely determined by basis-vector recovery. Simpler form:
        # mortar_q_pt = mortar_at_nonmortar_0 + (xi+1) * e_xi + (eta+1) * e_eta
        xi_s, eta_s = float(q_pt_nonmortar[0]), float(q_pt_nonmortar[1])
        return mortar_at_nonmortar_0 + (xi_s + 1.0) * e_xi + (eta_s + 1.0) * e_eta


# =============================================================================
# Concrete: tri-3 face mortar
# =============================================================================

class TriFaceMortarAssembler(MortarFaceAssembler):
    """Tri-3 face-mortar assembler.

    Uses ``M_tri3_dual_modified`` and ``N_tri3`` as kernels; reference
    quadrature is the 3-point degree-2 Dunavant rule on the simplex
    (sufficient for the bilinear nonmortar × bilinear mortar = degree 2
    integrand).
    """

    # ----------------------------------------------------------- constants
    @staticmethod
    def _tri3_boundary_tag_to_drops(boundary_tag: str) -> Tuple[bool, bool, bool]:
        """Map a TriFaceElement.boundary_tag to a 3-tuple of drop flags.

        Tag conventions (matched against types_3d.TriFaceElement docstring):
            "none"     -> (F, F, F)
            "v0"       -> (T, F, F)
            "v1"       -> (F, T, F)
            "v2"       -> (F, F, T)
            "v0-v1"    -> (T, T, F)
            "v0-v2"    -> (T, F, T)
            "v1-v2"    -> (F, T, T)
            "v0-v1-v2" -> (T, T, T)   # all dropped (rare/edge case)
        """
        mapping = {
            "none":     (False, False, False),
            "v0":       (True,  False, False),
            "v1":       (False, True,  False),
            "v2":       (False, False, True),
            "v0-v1":    (True,  True,  False),
            "v0-v2":    (True,  False, True),
            "v1-v2":    (False, True,  True),
            "v0-v1-v2": (True,  True,  True),
        }
        if boundary_tag not in mapping:
            raise ValueError(
                f"TriFaceMortarAssembler: unrecognised boundary_tag "
                f"{boundary_tag!r}. Expected one of {list(mapping.keys())!r}."
            )
        return mapping[boundary_tag]

    # ----------------------------------------------------------- subclass API
    def _eval_nonmortar_dual(
        self, q_pt: np.ndarray, boundary_tag: str,
    ) -> np.ndarray:
        # gauss_tri_3pt returns (3, 3) where each row is a full
        # barycentric tuple (L1, L2, L3); pass through directly.
        drops = self._tri3_boundary_tag_to_drops(boundary_tag)
        lam = (float(q_pt[0]), float(q_pt[1]), float(q_pt[2]))
        return np.asarray(
            M_tri3_dual_modified(lam, drops), dtype=np.float64,
        )

    def _eval_nonmortar_shape(self, q_pt: np.ndarray) -> np.ndarray:
        lam = (float(q_pt[0]), float(q_pt[1]), float(q_pt[2]))
        return np.asarray(N_tri3(lam), dtype=np.float64)

    def _eval_mortar_shape(self, q_pt_mortar: np.ndarray) -> np.ndarray:
        lam = (float(q_pt_mortar[0]), float(q_pt_mortar[1]), float(q_pt_mortar[2]))
        return np.asarray(N_tri3(lam), dtype=np.float64)

    def _build_quadrature(
        self, order: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # 3-point degree-2 Dunavant on the simplex; exact for any
        # bilinear-shape × bilinear-shape product. Returns (3, 3)
        # barycentric pts and (3,) weights summing to |T_ref| = 1/2.
        return gauss_tri_3pt()

    def _nonmortar_jacobian(self, nonmortar_elem) -> Callable[[np.ndarray], float]:
        # Jacobian of the affine map (reference simplex |T_ref|=1/2 ->
        # physical triangle |T|): J = 2 * |T| / (sum of weights).
        # Since gauss_tri_3pt's weights sum to |T_ref| = 1/2, multiplying
        # the integrand by J = 2 * |T| gives total physical area:
        #     sum_q w_q * J = (1/2) * (2|T|) = |T|.    ✓
        # In other words, J = phys_area / ref_area = phys_area / (1/2) =
        # 2 * phys_area.
        J_const = 2.0 * nonmortar_elem.physical_area
        return lambda q_pt, _J=J_const: _J

    def _n_nodes_per_elem(self) -> int:
        return 3

    def _n_basis_for_lumped_check(self) -> int:
        return 3

    def _shape_for_lumped_check(self) -> Callable:
        return N_tri3

    def _ref_quad_for_lumped_check(self) -> Tuple[np.ndarray, np.ndarray]:
        # gauss_tri_3pt already returns full (L1, L2, L3) tuples; pass
        # through unchanged.
        return gauss_tri_3pt()

    def _lumped_uses_tuple_input(self) -> bool:
        # N_tri3 takes a barycentric tuple.
        return True

    def _mortar_node_permutation_apply(
        self, mortar_node_perm: Sequence[int], q_pt_nonmortar: np.ndarray,
    ) -> np.ndarray:
        """For the conforming-pair case, the 6 dihedral-group permutations
        of the tri's 3 vertices reorder barycentric components.

        ``mortar_node_perm[i]`` = mortar-local index of the mortar node
        at nonmortar-local position i. Under this permutation, the mortar-
        side barycentric coord at the i-th nonmortar-local position is
        simply L_nonmortar[i] re-labelled — the mortar-side q_pt is the
        permuted barycentric tuple with components shuffled to match
        mortar-element local-node order.
        """
        if tuple(mortar_node_perm) == (0, 1, 2):
            return q_pt_nonmortar
        # Permute components: mortar_q_pt[mortar_node_perm[i]] = nonmortar_q_pt[i]
        L_mortar = np.zeros(3, dtype=np.float64)
        for i, m_local in enumerate(mortar_node_perm):
            L_mortar[m_local] = float(q_pt_nonmortar[i])
        return L_mortar


# =============================================================================
# Conforming-pair matching helper
# =============================================================================

def match_conforming_face_pairs(
    nonmortar_elems: Sequence,
    mortar_elems: Sequence,
    perpendicular_axis: str,
    period: float,
    *,
    tol_rel: float = 1e-9,
) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """Pair up nonmortar/mortar face elements by parametric centroid.

    Pure-Python, no MFEM. For each nonmortar element, finds the mortar
    element whose face-plane centroid is closest (after subtracting the
    periodic translation along the perpendicular axis) and returns the
    pairing list.

    This is the conforming case: each nonmortar element matches exactly one
    mortar element with the same parametric extent. Non-conforming
    (Phase 3.5) would require multi-element overlap from polygon
    clipping.

    Parameters
    ----------
    nonmortar_elems : sequence of QuadFaceElement or TriFaceElement
    mortar_elems : sequence of same
    perpendicular_axis : str
        "x", "y", or "z" — the axis the pair is periodic in.
    period : float
        Periodic translation length along ``perpendicular_axis``.
    tol_rel : float
        Tolerance for parametric-centroid match, relative to the nonmortar
        element's characteristic size.

    Returns
    -------
    list of (nonmortar_idx, mortar_idx, mortar_node_perm).

        mortar_node_perm[i] = local-node index in the mortar element
        of the mortar node that is geometrically *at the same parametric
        location* as nonmortar-element local node i.

        For axis-aligned MakeCartesian3D meshes, mortar_node_perm =
        (0, 1, ..., n-1) (identity). The function detects the natural
        permutation from physical-coord matching.
    """
    if len(nonmortar_elems) == 0 or len(mortar_elems) == 0:
        return []

    axis_idx_map = {"x": 0, "y": 1, "z": 2}
    perp_idx = axis_idx_map[perpendicular_axis]

    # Build an array of mortar centroids (in-plane only).
    in_plane_axes = [i for i in range(3) if i != perp_idx]
    n_mortar = len(mortar_elems)
    mortar_centroids = np.zeros((n_mortar, 2), dtype=np.float64)
    for i, m in enumerate(mortar_elems):
        c = m.coords.mean(axis=0)
        mortar_centroids[i] = c[in_plane_axes]

    # Mortar perpendicular-coord (should be nonmortar_perp + period for all
    # mortars, modulo a sign — let the user pass period with the right
    # sign).
    pair_matches: List[Tuple[int, int, Tuple[int, ...]]] = []
    for s_idx, s in enumerate(nonmortar_elems):
        s_centroid_3d = s.coords.mean(axis=0)
        s_centroid_inplane = s_centroid_3d[in_plane_axes]
        # Characteristic length scale of nonmortar element (extent in plane).
        char_len = float(np.linalg.norm(
            s.coords.max(axis=0) - s.coords.min(axis=0)
        ))
        tol = max(tol_rel * char_len, 1e-14)

        # Find mortar(s) within tol of nonmortar centroid.
        diffs = mortar_centroids - s_centroid_inplane
        dists = np.linalg.norm(diffs, axis=1)
        candidates = np.where(dists <= tol)[0]

        if len(candidates) == 0:
            raise RuntimeError(
                f"match_conforming_face_pairs: nonmortar element {s_idx} at "
                f"centroid {s_centroid_inplane} has no mortar partner "
                f"within tol={tol}. Mesh is non-conforming or pairs are "
                f"misordered."
            )
        if len(candidates) > 1:
            # Should not happen for a valid conforming RVE.
            raise RuntimeError(
                f"match_conforming_face_pairs: nonmortar element {s_idx} at "
                f"centroid {s_centroid_inplane} has multiple mortar "
                f"partners ({len(candidates)}) within tol={tol}. Check "
                f"for duplicated mortar elements."
            )
        m_idx = int(candidates[0])
        m = mortar_elems[m_idx]

        # Determine mortar_node_perm by matching nonmortar local-node coords
        # to mortar local-node coords (in-plane).
        mortar_node_perm = _node_perm_by_coord_match(
            s.coords, m.coords, in_plane_axes, tol,
        )
        pair_matches.append((s_idx, m_idx, mortar_node_perm))

    return pair_matches


def _node_perm_by_coord_match(
    nonmortar_coords: np.ndarray,
    mortar_coords: np.ndarray,
    in_plane_axes: List[int],
    tol: float,
) -> Tuple[int, ...]:
    """For each nonmortar local-node, find the mortar local-node at the same
    in-plane physical coords.

    Returns tuple of length n_nodes such that
    ``mortar_coords[perm[i]][in_plane_axes] ≈ nonmortar_coords[i][in_plane_axes]``.
    """
    n = nonmortar_coords.shape[0]
    s_in = nonmortar_coords[:, in_plane_axes]
    m_in = mortar_coords[:, in_plane_axes]
    perm: List[int] = []
    for i in range(n):
        diffs = m_in - s_in[i]
        dists = np.linalg.norm(diffs, axis=1)
        j_candidates = np.where(dists <= tol)[0]
        if len(j_candidates) != 1:
            raise RuntimeError(
                f"_node_perm_by_coord_match: nonmortar node {i} at "
                f"{s_in[i]} matched {len(j_candidates)} mortar nodes; "
                f"expected exactly 1 within tol={tol}."
            )
        perm.append(int(j_candidates[0]))
    return tuple(perm)
