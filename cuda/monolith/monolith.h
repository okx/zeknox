#ifndef _MONOLITH_H_
#define _MONOLITH_H_

#include <stdint.h>

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void cpu_monolith_hash_one(uint64_t *input, uint32_t size, uint64_t *data);
EXTERNC void cpu_monolith_hash_two(uint64_t *hash1, uint64_t *hash2, uint64_t *hash);

#ifdef USE_CUDA
EXTERNC DEVICE void gpu_monolith_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash);
EXTERNC DEVICE void gpu_monolith_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash);
#endif

#endif // _MONOLITH_H_
