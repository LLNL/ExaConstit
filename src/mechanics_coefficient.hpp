
#ifndef MECHANICS_COEF
#define MECHANICS_COEF

#include "mfem.hpp"

// class QuadratureFunction;

/// Quadrature function coefficient
class QuadratureVectorFunctionCoefficient : public mfem::VectorCoefficient
{
   private:
      mfem::QuadratureFunction *QuadF;
      int index;
      int length;

   public:
      // constructor with a quadrature function as input
      QuadratureVectorFunctionCoefficient(mfem::QuadratureFunction *qf)
         : mfem::VectorCoefficient(qf->GetVDim())
      {
         QuadF = qf;
         index = 0;
         length = qf->GetVDim();
      }

      // constructor with a null qf
      QuadratureVectorFunctionCoefficient() : mfem::VectorCoefficient(0) { QuadF = NULL; }

      void SetQuadratureFunction(mfem::QuadratureFunction *qf) { QuadF = qf; }

      void SetIndex(int _index);
      void SetLength(int _length);

      mfem::QuadratureFunction *GetQuadFunction() const { return QuadF; }

      using mfem::VectorCoefficient::Eval;
      virtual void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                        const mfem::IntegrationPoint &ip);

      // virtual void EvalQ(Vector &V, ElementTransformation &T,
      // const int ip_num);

      virtual ~QuadratureVectorFunctionCoefficient() { };
};

/// Generic quadrature function coefficient class for using
/// coefficients which only live at integration points
/// This is based on the same one found in Cardioid
class QuadratureFunctionCoefficient1 : public mfem::Coefficient
{
   private:
      mfem::QuadratureFunction *QuadF;

   public:
      QuadratureFunctionCoefficient1(mfem::QuadratureFunction *qf) { QuadF = qf; }

      QuadratureFunctionCoefficient1() : mfem::Coefficient() { QuadF = NULL; }

      void SetQuadratureFunction(mfem::QuadratureFunction *qf) { QuadF = qf; }

      mfem::QuadratureFunction *GetQuadFunction() const { return QuadF; }

      virtual double Eval(mfem::ElementTransformation &T,
                          const mfem::IntegrationPoint &ip);

      virtual double EvalQ(mfem::ElementTransformation &T,
                           const mfem::IntegrationPoint &ip);
};

#endif
