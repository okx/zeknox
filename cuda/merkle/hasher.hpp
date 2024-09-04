#ifndef _HASHER_CUH_
#define _HASHER_CUH_

#include <stdint.h>
#include <cstring>

#include "types/int_types.h"

#ifdef USE_CUDA
#include "types/gl64_t.cuh"
#else
#include "poseidon/goldilocks.hpp"
#endif

class Hasher {};

#define NUM_HASH_OUT_ELTS 4

#endif // _HASHER_CUH_