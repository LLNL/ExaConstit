
// Implementation of Coefficient class

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"

using namespace mfem;

void QuadratureVectorFunctionCoefficient::SetLength(int _length)
{
   int vdim = QuadF->GetVDim();

   MFEM_ASSERT(_length > 0, "Length must be > 0");
   vdim -= index;
   MFEM_ASSERT(_length <= vdim, "Length must be <= (QuadratureFunction length - index)");

   length = _length;
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
   int elem_no = T.ElementNo;
   if(index == 0 && length == QuadF->GetVDim()){
      QuadF->GetElementValues(elem_no, ip.ipID, V); 
   }else{
      //This will need to be improved upon...
      Vector temp;
      QuadF->GetElementValues(elem_no, ip.ipID, temp);
      double *data = temp.GetData();
      V.NewDataAndSize(data + index, length);
   }
   
   return;
}


  
//void QuadratureVectorFunctionCoefficient::EvalQ(Vector &V,
//                                                ElementTransformation &T,
//                                                const int ip_num)
//{
//   int elem_no = T.ElementNo;
//   QuadF->GetElementValues(elem_no, ip_num, V);
//   return;
//}

/// Standard coefficient evaluation is not valid
double QuadratureFunctionCoefficient::Eval(ElementTransformation &T,
                                           const IntegrationPoint &ip)
{
  //mfem_error ("QuadratureFunctionCoefficient::Eval (...)\n"
  //             "   is not implemented for this class.");
  // return 0.0;
   int elem_no = T.ElementNo;
   Vector temp(1);
   QuadF->GetElementValues(elem_no, ip.ipID, temp);
   return temp[0];
}

/// Evaluate the function coefficient at a specific quadrature point
double QuadratureFunctionCoefficient::EvalQ(ElementTransformation &T,
                                            const IntegrationPoint &ip)
{
   int elem_no = T.ElementNo;
   Vector temp(1);
   QuadF->GetElementValues(elem_no, ip.ipID, temp);
   return temp[0];
}
