#ifndef MECHANICS_INTEG
#define MECHANICS_INTEG

#include "mfem.hpp"
#include "mechanics_model.hpp"

#include <utility>
#include <unordered_map>
#include <string>

/// A NonlinearForm Integrator specifically built around the ExaModel class
/// and really focused around dealing with your general solid mechanics type
/// problems.
class ExaNLFIntegrator : public mfem::NonlinearFormIntegrator
{
   private:
      ExaModel *model;
      // Will take a look and see what I need and don't need for this.
      mfem::Vector dmat;
      mfem::Vector grad;
      mfem::Vector *tan_mat; // Not owned
      mfem::Vector pa_dmat;
      mfem::Vector jacobian;
      const mfem::GeometricFactors *geom; // Not owned
      int space_dims, nelems, nqpts, nnodes;

   public:
      ExaNLFIntegrator(ExaModel *m) : model(m) { }

      virtual ~ExaNLFIntegrator() { }

      /// This doesn't do anything at this point. We can add the functionality
      /// later on if a use case arises.
      virtual double GetElementEnergy(const mfem::FiniteElement &el,
                                      mfem::ElementTransformation &Ttr,
                                      const mfem::Vector &elfun);

      using mfem::NonlinearFormIntegrator::AssembleElementVector;
      /// Assembles the Div(sigma) term / RHS terms of our linearized system of equations.
      virtual void AssembleElementVector(const mfem::FiniteElement &el,
                                         mfem::ElementTransformation &Ttr,
                                         const mfem::Vector &elfun, mfem::Vector &elvect);

      /// Assembles our gradient matrix (K matrix as seen in typical mechanics FEM formulations)
      virtual void AssembleElementGrad(const mfem::FiniteElement &el,
                                       mfem::ElementTransformation &Ttr,
                                       const mfem::Vector & /*elfun*/, mfem::DenseMatrix &elmat);

      // We currently don't have the AssemblePADiagonal still need to work out what this
      // would look like for the 4D tensor contraction operation

      /** @brief Performs the initial assembly operation on our 4D stiffness tensor
      *   combining the adj(J) terms, quad pt wts, and det(J) terms.
      *
      *   In the below function we'll be applying the below action on our material
      *   tangent matrix C^{tan} at each quadrature point as:
      *   D_{ijkm} = 1 / det(J) * w_{qpt} * adj(J)^T_{ij} C^{tan}_{ijkl} adj(J)_{lm}
      *   where D is our new 4th order tensor, J is our jacobian calculated from the
      *   mesh geometric factors, and adj(J) is the adjugate of J.
      */
      virtual void AssemblePAGrad(const mfem::FiniteElementSpace &fes) override;
      virtual void AddMultPAGrad(const mfem::Vector &x, mfem::Vector &y) override;

      using mfem::NonlinearFormIntegrator::AssemblePA;
      virtual void AssemblePA(const mfem::FiniteElementSpace &fes) override;
      virtual void AddMultPA(const mfem::Vector & /*x*/, mfem::Vector &y) const override;

      virtual void AssembleDiagonalPA(mfem::Vector &y) override;

      /// Method defining element assembly.
      /** The result of the element assembly is added and stored in the @a emat
          Vector. */
      virtual void AssembleEA(const mfem::FiniteElementSpace &fes, mfem::Vector &emat) override;
};

// }

#endif
