#ifndef __POSEIDON2_CONST_HPP__
#define __POSEIDON2_CONST_HPP__

#include "types/int_types.h"

#ifdef USE_CUDA
extern __device__ __constant__ u64 GPU_HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS[8][8];
#else
extern const u64 HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS[8][8];
#endif

#ifdef USE_CUDA
extern __device__ __constant__ u64 GPU_HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS[22];
#else
extern const u64 HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS[22];
#endif

#ifdef USE_CUDA
extern __device__ __constant__ u64 GPU_MATRIX_DIAG_8_GOLDILOCKS_U64[8];
#else
extern const u64 MATRIX_DIAG_8_GOLDILOCKS_U64[8];
#endif

#ifdef USE_CUDA
extern __device__ __constant__ u64 GPU_MATRIX_DIAG_12_GOLDILOCKS_U64[12];
#else
extern const u64 MATRIX_DIAG_12_GOLDILOCKS_U64[12];
#endif

#ifdef USE_CUDA
extern __device__ __constant__ u64 GPU_MATRIX_DIAG_16_GOLDILOCKS_U64[16];
#else
extern const u64 MATRIX_DIAG_16_GOLDILOCKS_U64[16];
#endif


#ifdef USE_CUDA
extern __device__ __constant__ u64 GPU_MATRIX_DIAG_20_GOLDILOCKS_U64[20];
#else
extern const u64 MATRIX_DIAG_20_GOLDILOCKS_U64[20];
#endif

#ifdef USE_CUDA
extern __device__ __constant__ u32 GPU_BABYBEAR_WIDTH_16_EXT_CONST_P3[8 * 16];
#else
extern const u32 BABYBEAR_WIDTH_16_EXT_CONST_P3[8 * 16];
#endif

#ifdef USE_CUDA
extern __device__ __constant__ u32 GPU_BABYBEAR_WIDTH_16_INT_CONST_P3[13];
#else
extern const u32 BABYBEAR_WIDTH_16_INT_CONST_P3[13];
#endif

#ifdef USE_CUDA
extern __device__ __constant__ u32 GPU_BABYBEAR_WIDTH_24_EXT_CONST_P3[8 * 24];
#else
extern const u32 BABYBEAR_WIDTH_24_EXT_CONST_P3[8 * 24];
#endif

#ifdef USE_CUDA
extern __device__ __constant__ u32 GPU_BABYBEAR_WIDTH_24_INT_CONST_P3[21];
#else
extern const u32 BABYBEAR_WIDTH_24_INT_CONST_P3[21];
#endif

#endif // __POSEIDON2_CONST_HPP__