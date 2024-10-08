#ifndef __CRYOPTO_UTIL_GPU_T_CUH__
#define __CRYOPTO_UTIL_GPU_T_CUH__

#include <stdint.h>

#define LIMBS_ALIGNMENT(x) ((x) % 4 == 0 ? 16 : ((x) % 2 == 0 ? 8 : 4))

// __align(n)__ enforces that the memory for the struct begins at an address in memory that is a multiple of n bytes.

template <unsigned LIMBS_COUNT>
struct __align__(LIMBS_ALIGNMENT(LIMBS_COUNT)) storage
{
  static constexpr unsigned LC = LIMBS_COUNT;
  uint32_t limbs[LIMBS_COUNT];
};

template <unsigned OMEGAS_COUNT, unsigned LIMBS_COUNT>
struct __align__(LIMBS_ALIGNMENT(LIMBS_COUNT)) storage_array
{
  storage<LIMBS_COUNT> storages[OMEGAS_COUNT];
};
#endif