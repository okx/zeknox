#include "types/int_types.h"

#include "merkle/merkle.h"
#include "merkle/merkle_c.h"

#include "utils/cuda_utils.cuh"
#include "poseidon/poseidon.cuh"
#include "poseidon/poseidon.h"
#include "poseidon2/poseidon2.h"
#include "poseidon/poseidon_bn128.h"
#include "keccak/keccak.h"
#include "monolith/monolith.h"

// TODO - benchmark and select best TPB
#define TPB 64

/*
 * Selectors of GPU hash functions.
 */
__device__ void gpu_hash_one(gl64_t *lptr, u32 leaf_size, gl64_t *dptr, u64 hash_type)
{
    switch (hash_type)
    {
    case HashPoseidon:
        return gpu_poseidon_hash_one(lptr, leaf_size, dptr);
    case HashKeccak:
        return gpu_keccak_hash_one(lptr, leaf_size, dptr);
    case HashPoseidonBN128:
        return gpu_poseidon_bn128_hash_one(lptr, leaf_size, dptr);
    case HashPoseidon2:
        return gpu_poseidon2_hash_one(lptr, leaf_size, dptr);
    case HashMonolith:
        return gpu_monolith_hash_one(lptr, leaf_size, dptr);
    default:
        return;
    }
}

__device__ void gpu_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash, u64 hash_type)
{
    switch (hash_type)
    {
    case HashPoseidon:
        return gpu_poseidon_hash_two(hash1, hash2, hash);
    case HashKeccak:
        return gpu_keccak_hash_two(hash1, hash2, hash);
    case HashPoseidonBN128:
        return gpu_poseidon_bn128_hash_two(hash1, hash2, hash);
    case HashPoseidon2:
        return gpu_poseidon2_hash_two(hash1, hash2, hash);
    case HashMonolith:
        return gpu_monolith_hash_two(hash1, hash2, hash);
    default:
        return;
    }
}

/*
 * Compute only leaves hashes with direct mapping (digest i corresponds to leaf i).
 */
__global__ void compute_leaves_hashes_direct(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u64 hash_type)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + (tid * HASH_SIZE_U64);
    gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, hash_type);

    // if with stride
    // u64 *lptr = leaves + tid;
    // u64 *dptr = digests_buf + (tid * HASH_SIZE_U64);
    // gpu_poseidon_hash_one_stride((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, leaves_count);
}

/*
 * Compute only leaves hashes with direct mapping per subtree over the entire digest buffer.
 */
__global__ void compute_leaves_hashes_linear_all(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 subtree_leaves_len, u32 subtree_digests_len, u64 hash_type)
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

    gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, hash_type);

    // if with stride
    // u64 *lptr = leaves + tid;
    // u64 *dptr = digests_buf + didx * HASH_SIZE_U64;
    // gpu_poseidon_hash_one_stride((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, leaves_count);
}

/*
 * Compute only leaves hashes with direct mapping per subtree (one subtree per GPU).
 */
__global__ void compute_leaves_hashes_linear_per_gpu(u64 *leaves, u32 leaf_size, u64 *digests_buf, u32 subtree_leaves_len, u32 subtree_digests_len, u64 hash_type)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= subtree_leaves_len)
        return;

    int didx = (subtree_digests_len - subtree_leaves_len) + tid;
    // printf("%d\n", didx);
    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + (didx * HASH_SIZE_U64);
    gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, hash_type);
}

/*
 * Compute leaves hashes with indirect mapping (digest at digest_idx[i] corresponds to leaf i).
 */
__global__ void compute_leaves_hashes(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 *digest_idx, u64 hash_type)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + digest_idx[tid] * HASH_SIZE_U64;
    gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, hash_type);
}

/*
 * Compute leaves hashes with indirect mapping and offset (digest at digest_idx[i] corresponds to leaf i).
 */
__global__ void compute_leaves_hashes_offset(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 *digest_idx, u64 offset, u64 hash_type)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + (digest_idx[tid] - offset) * HASH_SIZE_U64;
    gpu_hash_one((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, hash_type);
}

/*
 * Compute internal Merkle tree hashes in linear structure. Only one round.
 */
__global__ void compute_internal_hashes_linear_all(u64 *digests_buf, u32 round_size, u32 last_idx, u32 subtree_count, u32 subtree_digests_len, u64 hash_type)
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
    gpu_hash_two((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr, hash_type);
}

__global__ void compute_internal_hashes_linear_per_gpu(u64 *digests_buf, u32 round_size, u32 last_idx, u64 hash_type)
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
    gpu_hash_two((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr, hash_type);
}

__global__ void compute_caps_hashes_linear(u64 *caps_buf, u64 *digests_buf, u64 cap_buf_size, u64 subtree_digests_len, u64 hash_type)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cap_buf_size)
        return;

    // compute hash
    u64 *sptr1 = digests_buf + tid * subtree_digests_len * HASH_SIZE_U64;
    u64 *dptr = caps_buf + tid * HASH_SIZE_U64;
    gpu_hash_two((gl64_t *)sptr1, (gl64_t *)(sptr1 + HASH_SIZE_U64), (gl64_t *)dptr, hash_type);
}

void fill_digests_buf_linear_gpu_with_gpu_ptr(
    void *digests_buf_gpu_ptr,
    void *cap_buf_gpu_ptr,
    void *leaves_buf_gpu_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type,
    u64 gpu_id)
{
    CHECKCUDAERR(cudaSetDevice(gpu_id));

    // (special case) compute leaf hashes on GPU
    if (cap_buf_size == leaves_buf_size)
    {
        compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)cap_buf_gpu_ptr, hash_type);
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
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)cap_buf_gpu_ptr, hash_type);
        }
        else
        {
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)digests_buf_gpu_ptr, hash_type);
            if (cap_buf_size <= TPB)
            {
                compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, hash_type);
            }
            else
            {
                compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, hash_type);
            }
        }
        return;
    }

    // (general case) compute leaf hashes on GPU
    compute_leaves_hashes_linear_all<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)digests_buf_gpu_ptr, subtree_leaves_len, subtree_digests_len, hash_type);

    // compute internal hashes on GPU
    u64 r = (u64)log2(subtree_leaves_len) - 1;
    u64 last_index = subtree_digests_len - subtree_leaves_len;

    for (; (1 << r) * cap_buf_size > TPB && r > 0; r--)
    {
        // printf("GPU Round (64) %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<((1 << r) * cap_buf_size) / TPB + 1, TPB>>>((u64 *)digests_buf_gpu_ptr, (1 << r), last_index, cap_buf_size, subtree_digests_len, hash_type);
    }

    for (; r > 0; r--)
    {
        // printf("GPU Round (1) %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<1, TPB>>>((u64 *)digests_buf_gpu_ptr, (1 << r), last_index, cap_buf_size, subtree_digests_len, hash_type);
    }

    // compute cap hashes on GPU
    if (cap_buf_size <= TPB)
    {
        compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, hash_type);
    }
    else
    {
        compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, hash_type);
    }
}

// The provided pointers need to be on GPU 0
void fill_digests_buf_linear_multigpu_with_gpu_ptr(
    void *digests_buf_gpu_ptr,
    void *cap_buf_gpu_ptr,
    void *leaves_buf_gpu_ptr,
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 hash_type)
{
    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));

    cudaStream_t gpu_stream[16];
    u64 *gpu_leaves_ptrs[16];
    u64 *gpu_digests_ptrs[16];
    u64 *gpu_caps_ptrs[16];
    gpu_leaves_ptrs[0] = (u64 *)leaves_buf_gpu_ptr;

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
            compute_leaves_hashes_direct<<<leaves_buf_size / (nDevices * TPB) + 1, TPB>>>(gpu_leaves_ptrs[i], leaves_buf_size / nDevices, leaf_size, gpu_caps_ptrs[i], hash_type);
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
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)cap_buf_gpu_ptr, hash_type);
        }
        else
        {
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)digests_buf_gpu_ptr, hash_type);
            if (cap_buf_size <= TPB)
            {
                compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, hash_type);
            }
            else
            {
                compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, hash_type);
            }
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
            compute_leaves_hashes_linear_per_gpu<<<blocks, threads, 0, gpu_stream[i]>>>(gpu_leaves_ptrs[i], leaf_size, gpu_digests_ptrs[i], subtree_leaves_len, subtree_digests_len, hash_type);

            u64 r = (u64)log2(subtree_leaves_len) - 1;
            u64 last_index = subtree_digests_len - subtree_leaves_len;

            for (; (1 << r) > TPB; r--)
            {
                last_index -= (1 << r);
                compute_internal_hashes_linear_per_gpu<<<(1 << r) / TPB + 1, TPB, 0, gpu_stream[i]>>>(gpu_digests_ptrs[i], (1 << r), last_index, hash_type);
            }
            for (; r > 0; r--)
            {
                last_index -= (1 << r);
                compute_internal_hashes_linear_per_gpu<<<1, (1 << r), 0, gpu_stream[i]>>>(gpu_digests_ptrs[i], (1 << r), last_index, hash_type);
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
        compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, hash_type);
    }
    else
    {
        compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len, hash_type);
    }

#pragma omp parallel for num_threads(nDevices)
    for (int i = 0; i < nDevices; i++)
    {
        CHECKCUDAERR(cudaSetDevice(i));
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[i]));
        CHECKCUDAERR(cudaFree(gpu_leaves_ptrs[i]));
        CHECKCUDAERR(cudaFree(gpu_digests_ptrs[i]));
    }
}
