
// Implementation of Coefficient class

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"

using namespace mfem;

void QuadratureVectorFunctionCoefficient::SetLength(int _length)
{
   int vdim_temp = QuadF->GetVDim();

   MFEM_ASSERT(_length > 0, "Length must be > 0");
   vdim_temp -= index;
   MFEM_ASSERT(_length <= vdim_temp, "Length must be <= (QuadratureFunction length - index)");

   length = _length;
   SetVDim(length);
}

void QuadratureVectorFunctionCoefficient::SetIndex(int _index)
{
   MFEM_ASSERT(_index >= 0, "Index must be >= 0");
   MFEM_ASSERT(_index < QuadF->GetVDim(), "Index must be < the QuadratureFunction length");
   index = _index;
}

void QuadratureVectorFunctionCoefficient::Eval(Vector &V,
                                               ElementTransformation &T,
                                               const IntegrationPoint &ip)
{
   QuadF->HostReadWrite();
   int elem_no = T.ElementNo;
   if (index == 0 && length == QuadF->GetVDim()) {
      QuadF->GetElementValues(elem_no, ip.index, V);
   }
   else {
      // This will need to be improved upon...
      Vector temp;
      QuadF->GetElementValues(elem_no, ip.index, temp);
      double *data = temp.HostReadWrite();
      V.NewDataAndSize(data + index, length);
   }

   return;
}

/// Standard coefficient evaluation is not valid
double QuadratureFunctionCoefficient1::Eval(ElementTransformation &T,
                                           const IntegrationPoint &ip)
{
   // mfem_error ("QuadratureFunctionCoefficient::Eval (...)\n"
   // "   is not implemented for this class.");
   // return 0.0;
   QuadF->HostReadWrite();
   int elem_no = T.ElementNo;
   Vector temp(1);
   QuadF->GetElementValues(elem_no, ip.index, temp);
   return temp[0];
}

/// Evaluate the function coefficient at a specific quadrature point
double QuadratureFunctionCoefficient1::EvalQ(ElementTransformation &T,
                                            const IntegrationPoint &ip)
{
   int elem_no = T.ElementNo;
   Vector temp(1);
   QuadF->GetElementValues(elem_no, ip.index, temp);
   return temp[0];
}