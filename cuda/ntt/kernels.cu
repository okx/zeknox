#ifndef __CRYPTO_KERNELS_CU__
#define __CRYPTO_KERNELS_CU__


/**
 * \param  i, the integer to be bit reversed, i is in range [0, 1<<nbits)
 * \param nbits, number of bits to represent the integer. e.g Goldilocks Field, log_n_size <=32, most of the time, nbits is <= 32
 * __brev will treat i as 32 bits integer and return the bit reversed integer, whose least (32 - nbits) bits are all zero.
*/
__device__ __forceinline__
index_t bit_rev(index_t i, unsigned int nbits)
{
    if (sizeof(i) == 4 || nbits <= 32)
        return __brev(i) >> (8*sizeof(unsigned int) - nbits);
    else
        return __brevll(i) >> (8*sizeof(unsigned long long) - nbits);
}

#ifdef __CUDA_ARCH__
__device__ __forceinline__
void shfl_bfly(fr_t& r, int laneMask)
{
    #pragma unroll
    for (int iter = 0; iter < r.len(); iter++)
        r[iter] = __shfl_xor_sync(0xFFFFFFFF, r[iter], laneMask);
}
#endif

__device__ __forceinline__
void shfl_bfly(index_t& index, int laneMask)
{
    index = __shfl_xor_sync(0xFFFFFFFF, index, laneMask);
}

// Permutes the data in an array such that data[i] = data[bit_reverse(i)]
// and data[bit_reverse(i)] = data[i]
__launch_bounds__(1024) __global__
void bit_rev_permutation(fr_t* d_out, const fr_t *d_in, uint32_t lg_domain_size)
{
    index_t i = threadIdx.x + blockDim.x * (index_t)blockIdx.x;
    index_t r = bit_rev(i, lg_domain_size);

    if (i < r || (d_out != d_in && i == r)) {
        fr_t t0 = d_in[i];
        fr_t t1 = d_in[r];
        d_out[r] = t0;
        d_out[i] = t1;
    }
}

template<typename T>
static __device__ __host__ constexpr uint32_t lg2(T n)
{   uint32_t ret=0; while (n>>=1) ret++; return ret;   }


/** for goldilocks field
 * if lg_domain_size=14, #blocks = 4, blockDim = 128; 4*128*32 = 1<< (2+5+7) = 1<<14 
*/
__global__
void bit_rev_permutation_aux(fr_t* out, const fr_t* in, uint32_t lg_domain_size)
{
    const size_t Z_COUNT = 256 / sizeof(fr_t);  // 32 for goldilocks field
    const uint32_t LG_Z_COUNT = lg2(Z_COUNT);   // 5 for goldilocks

    extern __shared__ fr_t exchange[]; // dynamically allocated shared memory within a CUDA kernel. Shared memory is a type of memory that is shared among threads within the same thread block and resides on-chip
    fr_t (*xchg)[Z_COUNT][Z_COUNT] = reinterpret_cast<decltype(xchg)>(exchange);

    index_t step = (index_t)1 << (lg_domain_size - LG_Z_COUNT); // if lg_domain_size = 14, it is 1<<9; treat all blocks (and threads) as the column size, 32 is row size
    index_t group_idx = (threadIdx.x + blockDim.x * (index_t)blockIdx.x) >> LG_Z_COUNT;  // col index divided by Z_COUNT, range [[0]*32,[1]*32,[2]*32,[3]*32, ... [15]*32]
    uint32_t brev_limit = lg_domain_size - LG_Z_COUNT * 2;
    index_t brev_mask = ((index_t)1 << brev_limit) - 1;
    index_t group_idx_brev =
        (group_idx & ~brev_mask) | bit_rev(group_idx & brev_mask, brev_limit);
    uint32_t group_thread = threadIdx.x & (Z_COUNT - 1); // group_thread in range [0..32] * 4
    uint32_t group_thread_rev = bit_rev(group_thread, LG_Z_COUNT);
    uint32_t group_in_block_idx = threadIdx.x >> LG_Z_COUNT;  // group_in_block_idx in range [[0]*32,[1]*32,[2]*32,[3]*32]

    #pragma unroll
    for (uint32_t i = 0; i < Z_COUNT; i++) {
        xchg[group_in_block_idx][i][group_thread_rev] =
            in[group_idx * Z_COUNT + i * step + group_thread];
    }

    if (Z_COUNT > WARP_SZ)
        __syncthreads();  // is used to synchronize threads within the same block, ensuring that all shared memory writes are visible to all threads before proceeding with further computation
    else
        __syncwarp();

    #pragma unroll
    for (uint32_t i = 0; i < Z_COUNT; i++) {
        out[group_idx_brev * Z_COUNT + i * step + group_thread] =
            xchg[group_in_block_idx][group_thread_rev][i];
    }
}

__device__ __forceinline__
fr_t get_intermediate_root(index_t pow, const fr_t (*roots)[WINDOW_SIZE],
                           unsigned int nbits = MAX_LG_DOMAIN_SIZE)
{
    unsigned int off = 0;

    fr_t t, root = roots[off][pow % WINDOW_SIZE];
    #pragma unroll 1
    while (pow >>= LG_WINDOW_SIZE)
        root *= (t = roots[++off][pow % WINDOW_SIZE]);

    return root;
}

__device__ __forceinline__
void get_intermediate_roots(fr_t& root0, fr_t& root1,
                            index_t idx0, index_t idx1,
                            const fr_t (*roots)[WINDOW_SIZE])
{
    int win = (WINDOW_NUM - 1) * LG_WINDOW_SIZE;
    int off = (WINDOW_NUM - 1);

    root0 = roots[off][idx0 >> win];
    root1 = roots[off][idx1 >> win];
    #pragma unroll 1
    while (off--) {
        fr_t t;
        win -= LG_WINDOW_SIZE;
        root0 *= (t = roots[off][(idx0 >> win) % WINDOW_SIZE]);
        root1 *= (t = roots[off][(idx1 >> win) % WINDOW_SIZE]);
    }
}

template<unsigned int z_count>
__device__ __forceinline__
void coalesced_load(fr_t r[z_count], const fr_t* inout, index_t idx,
                    const unsigned int stage)
{
    const unsigned int x = threadIdx.x & (z_count - 1);
    idx &= ~((index_t)(z_count - 1) << stage);
    idx += x;

    #pragma unroll
    for (int z = 0; z < z_count; z++, idx += (index_t)1 << stage)
        r[z] = inout[idx];
}

template<unsigned int z_count>
__device__ __forceinline__
void transpose(fr_t r[z_count])
{
    extern __shared__ fr_t shared_exchange[];
    fr_t (*xchg)[z_count] = reinterpret_cast<decltype(xchg)>(shared_exchange);

    const unsigned int x = threadIdx.x & (z_count - 1);
    const unsigned int y = threadIdx.x & ~(z_count - 1);

    #pragma unroll
    for (int z = 0; z < z_count; z++)
        xchg[y + z][x] = r[z];

    __syncwarp();

    #pragma unroll
    for (int z = 0; z < z_count; z++)
        r[z] = xchg[y + x][z];
}

template<unsigned int z_count>
__device__ __forceinline__
void coalesced_store(fr_t* inout, index_t idx, const fr_t r[z_count],
                     const unsigned int stage)
{
    const unsigned int x = threadIdx.x & (z_count - 1);
    idx &= ~((index_t)(z_count - 1) << stage);
    idx += x;

    #pragma unroll
    for (int z = 0; z < z_count; z++, idx += (index_t)1 << stage)
        inout[idx] = r[z];
}

const static int Z_COUNT = 256/8/sizeof(fr_t); // 4 for Goldilocks Field
# include "kernels/ct_mixed_radix_narrow.cu"

#endif /**__CRYPTO_KERNELS_CU__ */