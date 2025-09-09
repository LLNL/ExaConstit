#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>

#define real8 double

#ifdef WIN32
#define UMAT_API __declspec(dllexport)
#elif defined(__clang__)  || defined(__INTEL_LLVM_COMPILER)
#define UMAT_API extern "C"
#define UMAT umat
#else
#define UMAT_API extern "C"
#define UMAT umat_
#endif

UMAT_API
void UMAT(real8 * /* stress */, real8 * /* statev */, real8 *ddsdde,
          real8 *sse, real8 *spd, real8 *scd, real8 *rpl,
          real8 * /* ddsdt */, real8 *drplde, real8 *drpldt,
          real8 * /* stran */, real8 * /* dstran */, real8 * /* time */,
          real8 * /* deltaTime */, real8 * /* tempk */, real8 * /* dtemp */, real8 * /* predef */,
          real8 * /* dpred */, char * /* cmname */, int * /* ndi */, int * /* nshr */, int * ntens,
          int * /* nstatv */, real8 * /* props */, int * /* nprops */, real8 * /* coords */,
          real8 * /* drot */, real8 * /* pnewdt */, real8 * /* celent */,
          real8 * /* dfgrd0 */, real8 * /* dfgrd1 */, int * /* noel */, int * /* npt */,
          int * /* layer */, int * /* kspt */, int * /* kstep */, int * /* kinc */)
{
      sse[0] += 1.0;
      spd[0] += 1.0;
      scd[0] += 1.0;
      rpl[0] += 1.0;
      drpldt[0] += 1.0;
      for (int i = 0; i < ntens[0]; i++){
            drplde[i] += 1.0;
            for (int j = 0; j < ntens[0]; j++) {
                  ddsdde[j * ntens[0] + i] += 1;
            }
      }
}
