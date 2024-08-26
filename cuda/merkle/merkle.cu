#include "int_types.h"

#include "merkle.h"
#include "merkle_private.h"
#include "merkle_c.h"

#include "hasher.hpp"
#include "keccak.hpp"
#include "poseidon.cuh"
#include "poseidon_bn128.hpp"
#include "poseidon2.hpp"
#include "monolith.hpp"

#include "cuda_utils.cuh"

// 64 threads per block achives the best performance
#define TPB 64

__global__ void init_hasher(Hasher** hasher, uint64_t hash_type)
{
    switch (hash_type) {
        case HashKeccak:
            *hasher = new KeccakHasher();
            break;
        case HashPoseidon2:
            *hasher = new Poseidon2Hasher();
            break;
        case HashPoseidonBN128:
            *hasher = new PoseidonBN128Hasher();
            break;
        case HashMonolith:
            *hasher = new MonolithHasher();
            break;
        default:
            *hasher = new PoseidonHasher();
    }
}

/*
 * Compute only leaves hashes with direct mapping (digest i corresponds to leaf i).
 */
__global__ void compute_leaves_hashes_direct(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, Hasher **hasher)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + (tid * HASH_SIZE_U64);
    (*hasher)->gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);

    // if with stride
    // u64 *lptr = leaves + tid;
    // u64 *dptr = digests_buf + (tid * HASH_SIZE_U64);
    // gpu_poseidon_hash_one_stride((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, leaves_count);
}

/*
 * Compute only leaves hashes with direct mapping per subtree over the entire digest buffer.
 */
__global__ void compute_leaves_hashes_linear_all(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 subtree_leaves_len, u32 subtree_digests_len, Hasher **hasher)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    // tid is the leaf index
    u32 subtree_idx = tid / subtree_leaves_len;
    u32 in_tree_idx = tid % subtree_leaves_len;
    u32 didx = subtree_idx * subtree_digests_len + (subtree_digests_len - subtree_leaves_len) + in_tree_idx;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + didx * HASH_SIZE_U64;

    (*hasher)->gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);

    // if with stride
    // u64 *lptr = leaves + tid;
    // u64 *dptr = digests_buf + didx * HASH_SIZE_U64;
    // gpu_poseidon_hash_one_stride((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, leaves_count);
}

/*
 * Compute only leaves hashes with direct mapping per subtree (one subtree per GPU).
 */
__global__ void compute_leaves_hashes_linear_per_gpu(u64 *leaves, u32 leaf_size, u64 *digests_buf, u32 subtree_leaves_len, u32 subtree_digests_len, Hasher **hasher)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= subtree_leaves_len)
        return;

    int didx = (subtree_digests_len - subtree_leaves_len) + tid;
    // printf("%d\n", didx);
    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + (didx * HASH_SIZE_U64);
    (*hasher)->gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);
}

/*
 * Compute leaves hashes with indirect mapping (digest at digest_idx[i] corresponds to leaf i).
 */
__global__ void compute_leaves_hashes(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 *digest_idx, Hasher **hasher)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + digest_idx[tid] * HASH_SIZE_U64;
    (*hasher)->gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);
}

/*
 * Compute leaves hashes with indirect mapping and offset (digest at digest_idx[i] corresponds to leaf i).
 */
__global__ void compute_leaves_hashes_offset(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 *digest_idx, u64 offset, Hasher **hasher)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + (digest_idx[tid] - offset) * HASH_SIZE_U64;
    (*hasher)->gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);
}

/*
 * Compute internal Merkle tree hashes in linear structure. Only one round.
 */
__global__ void compute_internal_hashes_linear_all(u64 *digests_buf, u32 round_size, u32 last_idx, u32 subtree_count, u32 subtree_digests_len, Hasher **hasher)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= round_size * subtree_count)
        return;

    // compute indexes and pointers
    u32 subtree_idx = tid / round_size;
    u32 idx = tid % round_size;
    u64 *dptrs = digests_buf + (subtree_idx * subtree_digests_len + last_idx) * HASH_SIZE_U64;
    u64 *dptr = dptrs + idx * HASH_SIZE_U64;
    u64 *sptr1 = dptrs + (2 * (idx + 1) + last_idx) * HASH_SIZE_U64;
    u64 *sptr2 = dptrs + (2 * (idx + 1) + 1 + last_idx) * HASH_SIZE_U64;

    // compute hash
    (*hasher)->gpu_hash_two((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr);
}

__global__ void compute_internal_hashes_linear_per_gpu(u64 *digests_buf, u32 round_size, u32 last_idx, Hasher **hasher)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= round_size)
        return;

    // compute indexes and pointers
    u64 *dptrs = digests_buf + last_idx * HASH_SIZE_U64;
    u64 *dptr = dptrs + tid * HASH_SIZE_U64;
    u64 *sptr1 = dptrs + (2 * (tid + 1) + last_idx) * HASH_SIZE_U64;
    u64 *sptr2 = dptrs + (2 * (tid + 1) + 1 + last_idx) * HASH_SIZE_U64;

    // compute hash
    (*hasher)->gpu_hash_two((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr);
}

__global__ void compute_caps_hashes_linear(u64 *caps_buf, u64 *digests_buf, u64 cap_buf_size, u64 subtree_digests_len, Hasher **hasher)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cap_buf_size)
        return;

    // compute hash
    u64 *sptr1 = digests_buf + tid * subtree_digests_len * HASH_SIZE_U64;
    u64 *dptr = caps_buf + tid * HASH_SIZE_U64;
    (*hasher)->gpu_hash_two((gl64_t *)sptr1, (gl64_t *)(sptr1 + HASH_SIZE_U64), (gl64_t *)dptr);
}

void fill_digests_buf_linear_gpu_with_gpu_ptr(
    void *digests_buf_gpu_ptr,
    void *cap_buf_gpu_ptr,
    void *leaves_buf_gpu_ptr,
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height,
    uint64_t hash_type,
    uint64_t gpu_id)
{
    CHECKCUDAERR(cudaSetDevice(gpu_id));

    Hasher **gpu_hasher;
    CHECKCUDAERR(cudaMalloc(&gpu_hasher, sizeof(Hasher*)));
    init_hasher<<<1,1>>>(gpu_hasher, hash_type);

    // (special case) compute leaf hashes on GPU
    if (cap_buf_size == leaves_buf_size)
    {
        compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)cap_buf_gpu_ptr, gpu_hasher);

        CHECKCUDAERR(cudaFree(gpu_hasher));
        return;
    }

    // 2. compute leaf hashes on GPU
    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    // (special cases)
    if (subtree_leaves_len <= 2)
    {
        if (subtree_leaves_len == 1)
        {
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)cap_buf_gpu_ptr, gpu_hasher);
        }
        else
        {
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)digests_buf_gpu_ptr, gpu_hasher);
            if (cap_buf_size <= TPB)
            {
                compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, gpu_hasher);
            }
            else
            {
                compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, gpu_hasher);
            }
        }

        CHECKCUDAERR(cudaFree(gpu_hasher));
        return;
    }

    // (general case) compute leaf hashes on GPU
    compute_leaves_hashes_linear_all<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)digests_buf_gpu_ptr, subtree_leaves_len, subtree_digests_len, gpu_hasher);

    // compute internal hashes on GPU
    u64 r = (u64)log2(subtree_leaves_len) - 1;
    u64 last_index = subtree_digests_len - subtree_leaves_len;

    for (; (1 << r) * cap_buf_size > TPB && r > 0; r--)
    {
        // printf("GPU Round (64) %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<((1 << r) * cap_buf_size) / TPB + 1, TPB>>>((u64 *)digests_buf_gpu_ptr, (1 << r), last_index, cap_buf_size, subtree_digests_len, gpu_hasher);
    }

    for (; r > 0; r--)
    {
        // printf("GPU Round (1) %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<1, TPB>>>((u64 *)digests_buf_gpu_ptr, (1 << r), last_index, cap_buf_size, subtree_digests_len, gpu_hasher);
    }

    // compute cap hashes on GPU
    if (cap_buf_size <= TPB)
    {
        compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, gpu_hasher);
    }
    else
    {
        compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, gpu_hasher);
    }

    CHECKCUDAERR(cudaFree(gpu_hasher));
}

// The provided pointers need to be on GPU 0
void fill_digests_buf_linear_multigpu_with_gpu_ptr(
    void *digests_buf_gpu_ptr,
    void *cap_buf_gpu_ptr,
    void *leaves_buf_gpu_ptr,
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height,
    uint64_t hash_type)
{
    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));

    cudaStream_t gpu_stream[16];
    u64 *gpu_leaves_ptrs[16];
    u64 *gpu_digests_ptrs[16];
    u64 *gpu_caps_ptrs[16];
    gpu_leaves_ptrs[0] = (u64 *)leaves_buf_gpu_ptr;
    Hasher **gpu_hasher[16];

    for (int i = 0; i < nDevices; i++)
    {
        CHECKCUDAERR(cudaSetDevice(i));
        CHECKCUDAERR(cudaMalloc(&gpu_hasher[i], sizeof(Hasher*)));
        init_hasher<<<1,1>>>(gpu_hasher[i], hash_type);
    }

    // (special case) compute leaf hashes on GPU
    if (cap_buf_size == leaves_buf_size)
    {
        CHECKCUDAERR(cudaStreamCreate(gpu_stream));
        gpu_leaves_ptrs[0] = (u64 *)leaves_buf_gpu_ptr;
        gpu_caps_ptrs[0] = (u64 *)cap_buf_gpu_ptr;
        u64 leaves_per_gpu = leaves_buf_size * leaf_size / nDevices;
        u64 leaves_size_bytes = leaves_per_gpu * sizeof(u64);
        u64 caps_per_gpu = cap_buf_size / nDevices;

#pragma omp parallel for num_threads(nDevices)
        for (int i = 1; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));
            CHECKCUDAERR(cudaStreamCreate(gpu_stream + i));
            CHECKCUDAERR(cudaMalloc(&gpu_leaves_ptrs[i], leaves_size_bytes));
            CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_leaves_ptrs[i], i, (u64 *)leaves_buf_gpu_ptr + leaves_per_gpu, 0, leaves_size_bytes, gpu_stream[i]));
            CHECKCUDAERR(cudaMalloc(&gpu_caps_ptrs[i], caps_per_gpu * HASH_SIZE_U64 * sizeof(u64)));
        }
#pragma omp parallel for num_threads(nDevices)
        for (int i = 0; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));
            compute_leaves_hashes_direct<<<leaves_buf_size / (nDevices * TPB) + 1, TPB>>>(gpu_leaves_ptrs[i], leaves_buf_size / nDevices, leaf_size, gpu_caps_ptrs[i], gpu_hasher[i]);
        }
#pragma omp parallel for num_threads(nDevices)
        for (int i = 1; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaMemcpyPeerAsync(cap_buf_gpu_ptr, 0, gpu_caps_ptrs[i], i, caps_per_gpu * HASH_SIZE_U64 * sizeof(u64), gpu_stream[i]));
        }
#pragma omp parallel for num_threads(nDevices)
        for (int i = 0; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[i]));
            CHECKCUDAERR(cudaStreamDestroy(gpu_stream[i]));
            CHECKCUDAERR(cudaFree(gpu_hasher[i]));
        }
#pragma omp parallel for num_threads(nDevices)
        for (int i = 1; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));
            CHECKCUDAERR(cudaFree(gpu_leaves_ptrs[i]));
            CHECKCUDAERR(cudaFree(gpu_caps_ptrs[i]));
        }
        return;
    }

    // 2. compute leaf hashes on GPU
    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    // special case (use only one GPU)
    if (subtree_leaves_len <= 2)
    {
        CHECKCUDAERR(cudaSetDevice(0));
        if (subtree_leaves_len == 1)
        {
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)cap_buf_gpu_ptr, gpu_hasher[0]);
        }
        else
        {
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)digests_buf_gpu_ptr, gpu_hasher[0]);
            if (cap_buf_size <= TPB)
            {
                compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, gpu_hasher[0]);
            }
            else
            {
                compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, gpu_hasher[0]);
            }
        }

        for (int i = 0; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));
            CHECKCUDAERR(cudaFree(gpu_hasher[i]));
        }
        return;
    }

    // (general case) 1 subtree per GPU [TODO: could be further optimized]
    if ((1 << cap_height) < nDevices)
    {
        nDevices = (1 << cap_height);
    }

    u64 leaves_per_gpu = subtree_leaves_len;
    u64 leaves_size_bytes = leaves_per_gpu * leaf_size * sizeof(u64);
    u64 digests_per_gpu = subtree_digests_len;
    u64 digests_size_bytes = digests_per_gpu * HASH_SIZE_U64 * sizeof(u64);

#pragma omp parallel for num_threads(nDevices)
    for (int i = 0; i < nDevices; i++)
    {
        CHECKCUDAERR(cudaSetDevice(i));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + i));
        CHECKCUDAERR(cudaMalloc(&gpu_leaves_ptrs[i], leaves_size_bytes));
        CHECKCUDAERR(cudaMalloc(&gpu_digests_ptrs[i], digests_size_bytes));
    }

    for (int k = 0; k < (1 << cap_height); k += nDevices)
    {
#pragma omp parallel for num_threads(nDevices)
        for (int i = 0; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaMemcpyPeerAsync(gpu_leaves_ptrs[i], i, (u64 *)leaves_buf_gpu_ptr + (k + i) * leaves_per_gpu * leaf_size, 0, leaves_size_bytes, gpu_stream[i]));
        }
#pragma omp parallel for num_threads(nDevices)
        for (int i = 0; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));

            int blocks = (subtree_leaves_len % TPB == 0) ? subtree_leaves_len / TPB : subtree_leaves_len / TPB + 1;
            int threads = (subtree_leaves_len < TPB) ? subtree_leaves_len : TPB;
            compute_leaves_hashes_linear_per_gpu<<<blocks, threads, 0, gpu_stream[i]>>>(gpu_leaves_ptrs[i], leaf_size, gpu_digests_ptrs[i], subtree_leaves_len, subtree_digests_len, gpu_hasher[i]);

            u64 r = (u64)log2(subtree_leaves_len) - 1;
            u64 last_index = subtree_digests_len - subtree_leaves_len;

            for (; (1 << r) > TPB; r--)
            {
                last_index -= (1 << r);
                compute_internal_hashes_linear_per_gpu<<<(1 << r) / TPB + 1, TPB, 0, gpu_stream[i]>>>(gpu_digests_ptrs[i], (1 << r), last_index, gpu_hasher[i]);
            }
            for (; r > 0; r--)
            {
                last_index -= (1 << r);
                compute_internal_hashes_linear_per_gpu<<<1, (1 << r), 0, gpu_stream[i]>>>(gpu_digests_ptrs[i], (1 << r), last_index, gpu_hasher[i]);
            }
            CHECKCUDAERR(cudaMemcpyPeerAsync((u64 *)digests_buf_gpu_ptr + (k + i) * subtree_digests_len * HASH_SIZE_U64, 0, gpu_digests_ptrs[i], i, digests_size_bytes, gpu_stream[i]));
        }
#pragma omp parallel for num_threads(nDevices)
        for (int i = 0; i < nDevices; i++)
        {
            CHECKCUDAERR(cudaSetDevice(i));
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[i]));
        }
    }

    // compute cap hashes on GPU 0
    CHECKCUDAERR(cudaSetDevice(0));
    if (cap_buf_size <= TPB)
    {
        compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, gpu_hasher[0]);
    }
    else
    {
        compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, gpu_hasher[0]);
    }

#pragma omp parallel for num_threads(nDevices)
    for (int i = 0; i < nDevices; i++)
    {
        CHECKCUDAERR(cudaSetDevice(i));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[i]));
        CHECKCUDAERR(cudaFree(gpu_leaves_ptrs[i]));
        CHECKCUDAERR(cudaFree(gpu_digests_ptrs[i]));
        CHECKCUDAERR(cudaFree(gpu_hasher[i]));
    }
}

// #define TESTING
#ifdef TESTING

#define LEAF_SIZE_U64 68

#include <time.h>
#include <sys/time.h>
#include <omp.h>

int compare_results(u64 *digests_buf1, u64 *digests_buf2, u32 n_digests, u64 *cap_buf1, u64 *cap_buf2, u32 n_caps)
{
    int is_diff = 0;
    u64 *ptr1 = digests_buf1;
    u64 *ptr2 = digests_buf2;
    for (int i = 0; i < n_digests * HASH_SIZE_U64; i++, ptr1++, ptr2++)
    {
        if (*ptr1 != *ptr2)
        {
            is_diff = 1;
            break;
        }
    }
    ptr1 = cap_buf1;
    ptr2 = cap_buf2;
    for (int i = 0; is_diff == 0 && i < n_caps * HASH_SIZE_U64; i++, ptr1++, ptr2++)
    {
        if (*ptr1 != *ptr2)
        {
            is_diff = 1;
            break;
        }
    }
    if (is_diff)
    {
        printf("Test failed: outputs are different!\n");
    }
    else
    {
        printf("Test passed: outputs are the same!\n");
    }
    return is_diff;
}

/*
 * Run on GPU and CPU and compare the results. They have to be the same.
 */
void run_gpu_cpu_verify(u32 log_size)
{
    struct timeval t0, t1;

    u64 n_caps = 256;
    u64 n_leaves = (1 << log_size);
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = log2(n_digests) + 1;
    u64 cap_h = log2(n_caps);

    global_digests_buf = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    global_cap_buf = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    global_leaves_buf = (u64 *)malloc(n_leaves * LEAF_SIZE_U64 * sizeof(u64));

    global_cap_buf_end = global_cap_buf + n_caps * HASH_SIZE_U64 + 1;
    global_digests_buf_end = global_digests_buf + n_digests * HASH_SIZE_U64 + 1;
    global_leaves_buf_end = global_leaves_buf + n_leaves * 7 + 1;

    // Generate random leaves
    srand(time(NULL));
    for (int i = 0; i < n_leaves; i++)
    {
        for (int j = 0; j < LEAF_SIZE_U64; j++)
        {
            u32 r = rand();
            global_leaves_buf[i * LEAF_SIZE_U64 + j] = (u64)r << 32 + r * 88958514;
        }
    }
    printf("Leaves count: %ld\n", n_leaves);
    printf("Leaf size: %d\n", LEAF_SIZE_U64);
    printf("Digests count: %ld\n", n_digests);
    printf("Caps count: %ld\n", n_caps);
    printf("Caps height: %ld\n", cap_h);

    // Compute on GPU
    u64 *gpu_leaves;
    u64 *gpu_digests;
    u32 *gpu_caps;

    u64* digests_buf2 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64* cap_buf2 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));

    CHECKCUDAERR(cudaMalloc(&gpu_leaves, n_leaves * LEAF_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_digests, n_digests * HASH_SIZE_U64 * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_caps, n_caps * HASH_SIZE_U64 * sizeof(u64)));

    gettimeofday(&t0, 0);
    CHECKCUDAERR(cudaMemcpy(gpu_leaves, global_leaves_buf, n_leaves * LEAF_SIZE_U64 * sizeof(u64), cudaMemcpyHostToDevice));
    fill_digests_buf_linear_gpu_with_gpu_ptr(
        gpu_digests,
        gpu_caps,
        gpu_leaves,
        n_digests,
        n_caps,
        n_leaves,
        LEAF_SIZE_U64,
        cap_h,
        HashType::HashPoseidon,
        0
    );
    CHECKCUDAERR(cudaMemcpy(digests_buf2, gpu_digests, n_digests * HASH_SIZE_U64 * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(cap_buf2, gpu_caps, n_caps * HASH_SIZE_U64 * sizeof(u64), cudaMemcpyDeviceToHost));
    gettimeofday(&t1, 0);
    long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on GPU: %ld us\n", elapsed);

    printf("%lX %lX %lX %lX\n", cap_buf2[0], cap_buf2[1], cap_buf2[2], cap_buf2[3]);

    CHECKCUDAERR(cudaFree(gpu_leaves));
    CHECKCUDAERR(cudaFree(gpu_digests));
    CHECKCUDAERR(cudaFree(gpu_caps));

    gettimeofday(&t0, 0);
    fill_digests_buf_linear_cpu(n_digests, n_caps, n_leaves, LEAF_SIZE_U64, cap_h, HashType::HashPoseidon);
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);

    compare_results(global_digests_buf, digests_buf2, n_digests, global_cap_buf, cap_buf2, n_caps);
    /*
        printf("After fill...\n");
        for (int i = 0; i < n_digests; i++)
            print_hash(global_digests_buf + i * 4);

        printf("Cap...\n");
        for (int i = 0; i < n_caps; i++)
            print_hash(global_cap_buf + i * 4);
    */

    free(global_digests_buf);
    free(digests_buf2);
    free(global_cap_buf);
    free(cap_buf2);
}

/*
 * Run on GPU, CPU (1 core), CPU (multi-core) and get the execution times.
 */
/*
void run_gpu_cpu_comparison(u32 log_size)
{
    struct timeval t0, t1;

    u64 n_leaves = (1 << log_size);
    u64 n_caps = 2;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = log2(n_digests) + 1;

    global_digests_buf = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    global_cap_buf = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    global_leaves_buf = (u64 *)malloc(n_leaves * LEAF_SIZE_U64 * sizeof(u64));
    assert(global_digests_buf != NULL && global_cap_buf != NULL && global_leaves_buf != NULL);
    global_cap_buf_end = global_cap_buf + (n_caps + 1) * HASH_SIZE_U64;
    global_digests_buf_end = global_digests_buf + (n_digests + 1) * HASH_SIZE_U64;
    global_leaves_buf_end = global_leaves_buf + (n_leaves + 1) * LEAF_SIZE_U64;

    // Generate random leaves
    srand(time(NULL));
    for (int i = 0; i < n_leaves; i++)
    {
        for (int j = 0; j < LEAF_SIZE_U64; j++)
        {
            u32 r = rand();
            global_leaves_buf[i * LEAF_SIZE_U64 + j] = (u64)r << 32 + r * 88958514;
        }
    }
    printf("Number of leaves: %ld\n", n_leaves);

    // 1. Run on GPU
    init_gpu_functions(0);
    fill_init_rounds(n_leaves, rounds);
    gettimeofday(&t0, 0);
    fill_digests_buf_in_rounds_in_c_on_gpu(n_digests, n_caps, n_leaves, LEAF_SIZE_U64, 1);
    // fill_digests_buf_linear_gpu_v1(n_digests, n_caps, n_leaves, LEAF_SIZE_U64, 1);
    gettimeofday(&t1, 0);
    long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on GPU: %ld us\n", elapsed);
    fill_delete_rounds();

    u64 *digests_buf2 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    memcpy(digests_buf2, global_digests_buf, n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf2 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    memcpy(cap_buf2, global_cap_buf, n_caps * HASH_SIZE_U64 * sizeof(u64));

    // 2. Run on CPU (1 core, not using rounds)
    gettimeofday(&t0, 0);
    fill_digests_buf_in_c(n_digests, n_caps, n_leaves, LEAF_SIZE_U64, 1);
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU: %ld us\n", elapsed);

    compare_results(global_digests_buf, digests_buf2, n_digests, global_cap_buf, cap_buf2, n_caps);

    // 3. Run on CPU (nt cores, using rounds)
    fill_init_rounds(n_leaves, rounds);
    int nt = 1;
#pragma omp parallel
    nt = omp_get_num_threads();
    gettimeofday(&t0, 0);
    fill_digests_buf_in_rounds_in_c(n_digests, n_caps, n_leaves, LEAF_SIZE_U64, 1);
    gettimeofday(&t1, 0);
    elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on CPU parallel %d threads: %ld us\n", nt, elapsed);
    fill_delete_rounds();

    // free
    free(global_digests_buf);
    free(global_cap_buf);
    free(global_leaves_buf);
    free(digests_buf2);
    free(cap_buf2);
}
*/

int main(int argc, char **argv)
{
    int size = 10;
    if (argc > 1)
    {
        size = atoi(argv[1]);
    }
    assert(3 <= size);

    run_gpu_cpu_verify(size);

    return 0;
}

#endif // TESTING
