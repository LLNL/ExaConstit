#ifndef MECHANICS_LOG
#define MECHANICS_LOG

#ifdef HAVE_CALIPER
#include "caliper/cali.h"
#include "caliper/cali-mpi.h"
#define CALI_INIT \
   cali_mpi_init(); \
   cali_init();
#else
#define CALI_CXX_MARK_FUNCTION
#define CALI_MARK_BEGIN(name)
#define CALI_MARK_END(name)
#define CALI_CXX_MARK_SCOPE(name)
#define CALI_INIT
#endif

#endif