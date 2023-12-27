#ifndef __CGBN_COMMON_CUH__
#define __CGBN_COMMON_CUH__

#include <cuda.h>
#include <gmp.h>
#ifndef __CGBN_H__
#define __CGBN_H__
#include <cgbn/cgbn.h>
#endif
#include <assert.h>
#include "int_types.h"

#define TPI 1
#define BITS 256
#define BATCH 8192
#define TPB 64

typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, BITS> env_t;

#endif // __CGBN_COMMON_CUH__