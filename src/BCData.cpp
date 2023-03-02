
#include "mfem.hpp"
#include "BCData.hpp"

using namespace mfem;

BCData::BCData()
{
   // TODO constructor stub
}

BCData::~BCData()
{
   // TODO destructor stub
}

void BCData::setDirBCs(Vector& y)
{
   // When doing the velocity based methods we only
   // need to do the below.
   y = 0.0;
   y[0] = essVel[0] * scale[0];
   y[1] = essVel[1] * scale[1];
   y[2] = essVel[2] * scale[2];
}

void BCData::setScales()
{
   switch (compID) {
      case 7:
         scale[0] = 1.0;
         scale[1] = 1.0;
         scale[2] = 1.0;
         break;
      case  1:
         scale[0] = 1.0;
         scale[1] = 0.0;
         scale[2] = 0.0;
         break;
      case  2:
         scale[0] = 0.0;
         scale[1] = 1.0;
         scale[2] = 0.0;
         break;
      case  3:
         scale[0] = 0.0;
         scale[1] = 0.0;
         scale[2] = 1.0;
         break;
      case  4:
         scale[0] = 1.0;
         scale[1] = 1.0;
         scale[2] = 0.0;
         break;
      case  5:
         scale[0] = 0.0;
         scale[1] = 1.0;
         scale[2] = 1.0;
         break;
      case  6:
         scale[0] = 1.0;
         scale[1] = 0.0;
         scale[2] = 1.0;
         break;
      case  0:
         scale[0] = 0.0;
         scale[1] = 0.0;
         scale[2] = 0.0;
         break;
   }
}

void BCData::getComponents(int id, Array<bool> &component)
{
   switch (id) {
      case  0:
         component[0] = false;
         component[1] = false;
         component[2] = false;
         break;

      case  1:
         component[0] = true;
         component[1] = false;
         component[2] = false;
         break;
      case  2:
         component[0] = false;
         component[1] = true;
         component[2] = false;
         break;
      case  3:
         component[0] = false;
         component[1] = false;
         component[2] = true;
         break;
      case  4:
         component[0] = true;
         component[1] = true;
         component[2] = false;
         break;
      case  5:
         component[0] = false;
         component[1] = true;
         component[2] = true;
         break;
      case  6:
         component[0] = true;
         component[1] = false;
         component[2] = true;
         break;
      case  7:
         component[0] = true;
         component[1] = true;
         component[2] = true;
         break;
   }
}
