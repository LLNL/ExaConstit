
#ifndef BCDATA
#define BCDATA

#include "mfem.hpp"
#include "mfem/linalg/vector.hpp"
#include <fstream>

class BCData
{
   public:
      BCData();
      ~BCData();

      // scales for nonzero Dirichlet BCs
      double essDisp[3];
      double scale[3];
      int compID;
      double dt, tf;

      void setDirBCs(mfem::Vector& y);

      void setScales();

      static void getComponents(int id, mfem::Array<int> &component);
};
#endif
