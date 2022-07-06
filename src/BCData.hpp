
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
      double essVel[3];
      double scale[3];
      int compID;

      void setDirBCs(mfem::Vector& y);

      void setScales();

      static void getComponents(int id, mfem::Array<bool> &component);
};
#endif
