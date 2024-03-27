#include "int_types.h"

#include "merkle.h"
#include "merkle_private.h"

#include "cuda_utils.cuh"
#include "poseidon.cuh"
#include "poseidon.h"
#include "poseidon2.h"
#include "poseidon_bn128.h"
#include "keccak.h"

// #include "goldilocks.hpp"

#define TPB 64

// pointers to the actual hash functions (could be Poseidon or Keccak)
__device__ void (*gpu_hash_one_ptr)(gl64_t *input, u32 size, gl64_t *hash);
__device__ void (*gpu_hash_two_ptr)(gl64_t *hash1, gl64_t *hash2, gl64_t *hash);

__global__ void init_gpu_functions_poseidon_kernel()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gpu_hash_one_ptr = &gpu_poseidon_hash_one;
    gpu_hash_two_ptr = &gpu_poseidon_hash_two;
}

__global__ void init_gpu_functions_poseidon2_kernel()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gpu_hash_one_ptr = &gpu_poseidon2_hash_one;
    gpu_hash_two_ptr = &gpu_poseidon2_hash_two;
}

__global__ void init_gpu_functions_poseidon_bn128_kernel()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gpu_hash_one_ptr = &gpu_poseidon_bn128_hash_one;
    gpu_hash_two_ptr = &gpu_poseidon_bn128_hash_two;
}

__global__ void init_gpu_functions_keccak_kernel()
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;

    gpu_hash_one_ptr = &gpu_keccak_hash_one;
    gpu_hash_two_ptr = &gpu_keccak_hash_two;
}

void init_gpu_functions(u64 hash_type)
{
    static int64_t initialize_hash_type = -1;

    if (initialize_hash_type == -1 || initialize_hash_type != hash_type)
    {
        initialize_hash_type = hash_type;

        int nDevices = 0;
        CHECKCUDAERR(cudaGetDeviceCount(&nDevices));

        for (int d = 0; d < nDevices; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            switch (hash_type)
            {
            case 0:
                init_gpu_functions_poseidon_kernel<<<1, 1>>>();
                cpu_hash_one_ptr = &cpu_poseidon_hash_one;
                cpu_hash_two_ptr = &cpu_poseidon_hash_two;
                break;
            case 1:
                init_gpu_functions_keccak_kernel<<<1, 1>>>();
                cpu_hash_one_ptr = &cpu_keccak_hash_one;
                cpu_hash_two_ptr = &cpu_keccak_hash_two;
                break;
            case 2:
                init_gpu_functions_poseidon_bn128_kernel<<<1, 1>>>();
                cpu_hash_one_ptr = &cpu_poseidon_bn128_hash_one;
                cpu_hash_two_ptr = &cpu_poseidon_bn128_hash_two;
                break;
            case 3:
                init_gpu_functions_poseidon2_kernel<<<1, 1>>>();
                cpu_hash_one_ptr = &cpu_poseidon2_hash_one;
                cpu_hash_two_ptr = &cpu_poseidon2_hash_two;
                break;
            default:
                init_gpu_functions_poseidon_kernel<<<1, 1>>>();
                cpu_hash_one_ptr = &cpu_poseidon_hash_one;
                cpu_hash_two_ptr = &cpu_poseidon_hash_two;
            }
        }
    }
}

void init_gpu_functions(u64 hash_type, int gpu_id)
{
    static int64_t initialize_hash_type_one = -1;

    if (initialize_hash_type_one == -1 || initialize_hash_type_one != hash_type)
    {
        initialize_hash_type_one = hash_type;

        CHECKCUDAERR(cudaSetDevice(gpu_id));
        switch (hash_type)
        {
        case 0:
            init_gpu_functions_poseidon_kernel<<<1, 1>>>();
            cpu_hash_one_ptr = &cpu_poseidon_hash_one;
            cpu_hash_two_ptr = &cpu_poseidon_hash_two;
            break;
        case 1:
            init_gpu_functions_keccak_kernel<<<1, 1>>>();
            cpu_hash_one_ptr = &cpu_keccak_hash_one;
            cpu_hash_two_ptr = &cpu_keccak_hash_two;
            break;
        case 2:
            init_gpu_functions_poseidon_bn128_kernel<<<1, 1>>>();
            cpu_hash_one_ptr = &cpu_poseidon_bn128_hash_one;
            cpu_hash_two_ptr = &cpu_poseidon_bn128_hash_two;
            break;
        case 3:
            init_gpu_functions_poseidon2_kernel<<<1, 1>>>();
            cpu_hash_one_ptr = &cpu_poseidon2_hash_one;
            cpu_hash_two_ptr = &cpu_poseidon2_hash_two;
            break;
        default:
            init_gpu_functions_poseidon_kernel<<<1, 1>>>();
            cpu_hash_one_ptr = &cpu_poseidon_hash_one;
            cpu_hash_two_ptr = &cpu_poseidon_hash_two;
        }
    }
}

// GPU kernels

/*
 * Compute only leaves hashes with direct mapping (digest i corresponds to leaf i).
 */
__global__ void compute_leaves_hashes_direct(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + (tid * HASH_SIZE_U64);
    gpu_hash_one_ptr((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);

    // if with stride
    // u64 *lptr = leaves + tid;
    // u64 *dptr = digests_buf + (tid * HASH_SIZE_U64);
    // gpu_poseidon_hash_one_stride((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, leaves_count);
}

/*
 * Compute only leaves hashes with direct mapping per subtree over the entire digest buffer.
 */
__global__ void compute_leaves_hashes_linear_all(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 subtree_leaves_len, u32 subtree_digests_len)
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
    gpu_hash_one_ptr((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);

    // if with stride
    // u64 *lptr = leaves + tid;
    // u64 *dptr = digests_buf + didx * HASH_SIZE_U64;
    // gpu_poseidon_hash_one_stride((gl64_t *)lptr, leaf_size, (gl64_t *)dptr, leaves_count);
}

/*
 * Compute only leaves hashes with direct mapping per subtree (one subtree per GPU).
 */
__global__ void compute_leaves_hashes_linear_per_gpu(u64 *leaves, u32 leaf_size, u64 *digests_buf, u32 subtree_leaves_len, u32 subtree_digests_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= subtree_leaves_len)
        return;

    int didx = (subtree_digests_len - subtree_leaves_len) + tid;
    // printf("%d\n", didx);
    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + (didx * HASH_SIZE_U64);
    gpu_hash_one_ptr((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);
}

/*
 * Compute leaves hashes with indirect mapping (digest at digest_idx[i] corresponds to leaf i).
 */
__global__ void compute_leaves_hashes(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 *digest_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + digest_idx[tid] * HASH_SIZE_U64;
    gpu_hash_one_ptr((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);
}

/*
 * Compute leaves hashes with indirect mapping and offset (digest at digest_idx[i] corresponds to leaf i).
 */
__global__ void compute_leaves_hashes_offset(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 *digest_idx, u64 offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    u64 *lptr = leaves + (tid * leaf_size);
    u64 *dptr = digests_buf + (digest_idx[tid] - offset) * HASH_SIZE_U64;
    gpu_hash_one_ptr((gl64_t *)lptr, leaf_size, (gl64_t *)dptr);
}

/*
 * Compute internal Merkle tree hashes based on precomputed indexes. Cover all rounds.
 */
__global__ void compute_internal_hashes(u64 *digests_buf, u32 max_rounds, u32 max_round_size, u32 *round_size, HashTask *internal_index, u32 round_threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int r = max_rounds; r > round_threshold; r--)
    {
        if (tid < round_size[r])
        {
            HashTask *ht = &internal_index[r * max_round_size + tid];
            u64 *dptr = digests_buf + (ht->target_index * HASH_SIZE_U64);
            u64 *sptr1 = digests_buf + (ht->left_index * HASH_SIZE_U64);
            u64 *sptr2 = digests_buf + (ht->right_index * HASH_SIZE_U64);
            gpu_hash_two_ptr((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr);
        }
        // __syncthreads();
    }
}

/*
 * Compute internal Merkle tree hashes based on precomputed indexes. Only one round.
 */
__global__ void compute_internal_hashes_per_round(u64 *digests_buf, u32 round_size, u32 round_idx, u32 max_round_size, HashTask *internal_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= round_size)
        return;

    HashTask *ht = &internal_index[round_idx * max_round_size + tid];
    u64 *dptr = digests_buf + (ht->target_index * HASH_SIZE_U64);
    u64 *sptr1 = digests_buf + (ht->left_index * HASH_SIZE_U64);
    u64 *sptr2 = digests_buf + (ht->right_index * HASH_SIZE_U64);
    gpu_hash_two_ptr((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr);
}

__global__ void compute_caps_hashes_per_round(u64 *caps_buf, u64 *digests_buf, u32 round_size, u32 round_idx, u32 max_round_size, HashTask *internal_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= round_size)
        return;

    HashTask *ht = &internal_index[round_idx * max_round_size + tid];
    u64 *dptr = caps_buf + (ht->target_index * HASH_SIZE_U64);
    u64 *sptr1 = digests_buf + (ht->left_index * HASH_SIZE_U64);
    u64 *sptr2 = digests_buf + (ht->right_index * HASH_SIZE_U64);
    gpu_hash_two_ptr((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr);
}

/*
 * Compute internal Merkle tree hashes in linear structure. Only one round.
 */
__global__ void compute_internal_hashes_linear_all(u64 *digests_buf, u32 round_size, u32 last_idx, u32 subtree_count, u32 subtree_digests_len)
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
    gpu_hash_two_ptr((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr);
}

__global__ void compute_internal_hashes_linear_per_gpu(u64 *digests_buf, u32 round_size, u32 last_idx)
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
    gpu_hash_two_ptr((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr);
}

__global__ void compute_caps_hashes_linear(u64 *caps_buf, u64 *digests_buf, u64 cap_buf_size, u64 subtree_digests_len)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cap_buf_size)
        return;

    // compute hash
    u64 *sptr1 = digests_buf + tid * subtree_digests_len * HASH_SIZE_U64;
    u64 *dptr = caps_buf + tid * HASH_SIZE_U64;
    gpu_hash_two_ptr((gl64_t *)sptr1, (gl64_t *)(sptr1 + HASH_SIZE_U64), (gl64_t *)dptr);
}

/*
 * Compute internal Merkle tree hashes in linear structure on the entire buffer.
 */
__global__ void compute_internal_hashes_linear(u64 *digests_buf, u32 round_size, u32 last_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= round_size)
        return;

    u64 *dptr = digests_buf + tid * HASH_SIZE_U64;
    u64 *sptr1 = digests_buf + (2 * (tid + 1) + last_idx) * HASH_SIZE_U64;
    u64 *sptr2 = digests_buf + (2 * (tid + 1) + 1 + last_idx) * HASH_SIZE_U64;
    gpu_hash_two_ptr((gl64_t *)sptr1, (gl64_t *)sptr2, (gl64_t *)dptr);
}

// CPU functions
void fill_digests_buf_in_rounds_in_c_on_gpu(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height)
{
    u64 *gpu_leaves;
    u64 *gpu_digests;
    u64 *gpu_caps;
    u32 *gpu_indexes;
    u32 *gpu_round_size;
    HashTask *gpu_internal_indexes;

    u32 leaves_size_bytes = leaves_buf_size * leaf_size * 8;
    CHECKCUDAERR(cudaMalloc(&gpu_leaves, leaves_size_bytes));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_leaves, global_leaves_buf, leaves_size_bytes, cudaMemcpyHostToDevice));

    if (cap_buf_size == leaves_buf_size)
    {
        u32 digests_size_bytes = cap_buf_size * HASH_SIZE_U64 * sizeof(u64);
        CHECKCUDAERR(cudaMalloc(&gpu_digests, digests_size_bytes));
        compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>(gpu_leaves, leaves_buf_size, leaf_size, gpu_digests);
        CHECKCUDAERR(cudaMemcpy(global_cap_buf, gpu_digests, digests_size_bytes, cudaMemcpyDeviceToHost));

        // free
        cudaFree(gpu_digests);
        cudaFree(gpu_leaves);

        return;
    }

    // 1.1 copy leaves from CPU to GPU
    // 1.2 (in parallel) run fill_tree_get_index on CPU
    // 2.1 compute leaf hashes on GPU
    // 2.2 (in parallel) copy task index data to GPU
    // 3. compute internal hashes on GPU
    // 4. copy data from GPU to CPU
    // 5. compute cap hashes on CPU

    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    // 1.2 (in parallel) run fill_tree_get_index on CPU
    for (u64 k = 0; k < cap_buf_size; k++)
    {
        fill_subtree_get_index(k, k * subtree_digests_len, subtree_digests_len, k * subtree_leaves_len, subtree_leaves_len, leaf_size, 0);
    }

    u32 digests_size_bytes = digests_buf_size * HASH_SIZE_U64 * sizeof(u64);
    u32 caps_size_bytes = cap_buf_size * HASH_SIZE_U64 * sizeof(u64);
    CHECKCUDAERR(cudaMalloc(&gpu_digests, digests_size_bytes));
    CHECKCUDAERR(cudaMalloc(&gpu_caps, caps_size_bytes));
    CHECKCUDAERR(cudaMalloc(&gpu_indexes, leaves_buf_size * sizeof(u32)));
    CHECKCUDAERR(cudaMalloc(&gpu_internal_indexes, (global_max_round + 1) * global_max_round_size * sizeof(HashTask)));
    CHECKCUDAERR(cudaMalloc(&gpu_round_size, (global_max_round + 1) * sizeof(u32)));

    // 2.1 compute leaf hashes on GPU
    CHECKCUDAERR(cudaMemcpyAsync(gpu_indexes, global_leaf_index, leaves_buf_size * sizeof(u32), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_round_size, global_round_size, (global_max_round + 1) * sizeof(u32), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpyAsync((void *)gpu_internal_indexes, (void *)global_internal_index, (global_max_round + 1) * global_max_round_size * sizeof(HashTask), cudaMemcpyHostToDevice));
    compute_leaves_hashes<<<leaves_buf_size / TPB + 1, TPB>>>(gpu_leaves, leaves_buf_size, leaf_size, gpu_digests, gpu_indexes);

    int r = global_max_round;
    for (; global_round_size[r] > TPB; r--)
    {
        compute_internal_hashes_per_round<<<global_round_size[r] / TPB, TPB>>>(gpu_digests, global_round_size[r], r, global_max_round_size, gpu_internal_indexes);
    }

    for (; r > 0; r--)
    {
        compute_internal_hashes_per_round<<<1, global_round_size[r]>>>(gpu_digests, global_round_size[r], r, global_max_round_size, gpu_internal_indexes);
    }

    CHECKCUDAERR(cudaMemcpyAsync(global_digests_buf, gpu_digests, digests_size_bytes, cudaMemcpyDeviceToHost));

    compute_caps_hashes_per_round<<<1, global_round_size[0]>>>(gpu_caps, gpu_digests, global_round_size[0], 0, global_max_round_size, gpu_internal_indexes);

    CHECKCUDAERR(cudaMemcpy(global_cap_buf, gpu_caps, caps_size_bytes, cudaMemcpyDeviceToHost));

    // free
    cudaFree(gpu_digests);
    cudaFree(gpu_caps);
    cudaFree(gpu_indexes);
    cudaFree(gpu_leaves);
    cudaFree(gpu_internal_indexes);
    cudaFree(gpu_round_size);
}

void fill_digests_buf_in_rounds_in_c_on_gpu_v1(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height)
{
    u64 *gpu_leaves;
    u64 *gpu_digests;
    u32 *gpu_indexes;
    u32 *gpu_round_size;
    HashTask *gpu_internal_indexes;

    u32 leaves_size_bytes = leaves_buf_size * leaf_size * 8;
    CHECKCUDAERR(cudaMalloc(&gpu_leaves, leaves_size_bytes));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_leaves, global_leaves_buf, leaves_size_bytes, cudaMemcpyHostToDevice));

    if (cap_buf_size == leaves_buf_size)
    {
        u32 digests_size_bytes = cap_buf_size * HASH_SIZE_U64 * sizeof(u64);
        CHECKCUDAERR(cudaMalloc(&gpu_digests, digests_size_bytes));
        compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>(gpu_leaves, leaves_buf_size, leaf_size, gpu_digests);
        CHECKCUDAERR(cudaMemcpy(global_cap_buf, gpu_digests, digests_size_bytes, cudaMemcpyDeviceToHost));

        // free
        cudaFree(gpu_digests);
        cudaFree(gpu_leaves);

        return;
    }

    // 1.1 copy leaves from CPU to GPU
    // 1.2 (in parallel) run fill_tree_get_index on CPU
    // 2.1 compute leaf hashes on GPU
    // 2.2 (in parallel) copy task index data to GPU
    // 3. compute internal hashes on GPU
    // 4. copy data from GPU to CPU
    // 5. compute cap hashes on CPU

    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    // 1.2 (in parallel) run fill_tree_get_index on CPU
    for (u64 k = 0; k < cap_buf_size; k++)
    {
        fill_subtree_get_index(k, k * subtree_digests_len, subtree_digests_len, k * subtree_leaves_len, subtree_leaves_len, leaf_size, 0);
    }

    u32 digests_size_bytes = digests_buf_size * HASH_SIZE_U64 * sizeof(u64);
    CHECKCUDAERR(cudaMalloc(&gpu_digests, digests_size_bytes));
    CHECKCUDAERR(cudaMalloc(&gpu_indexes, leaves_buf_size * sizeof(u32)));
    CHECKCUDAERR(cudaMalloc(&gpu_internal_indexes, (global_max_round + 1) * global_max_round_size * sizeof(HashTask)));
    CHECKCUDAERR(cudaMalloc(&gpu_round_size, (global_max_round + 1) * sizeof(u32)));

    // 2.1 compute leaf hashes on GPU
    CHECKCUDAERR(cudaMemcpyAsync(gpu_indexes, global_leaf_index, leaves_buf_size * sizeof(u32), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_round_size, global_round_size, (global_max_round + 1) * sizeof(u32), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpyAsync((void *)gpu_internal_indexes, (void *)global_internal_index, (global_max_round + 1) * global_max_round_size * sizeof(HashTask), cudaMemcpyHostToDevice));
    compute_leaves_hashes<<<leaves_buf_size / TPB + 1, TPB>>>(gpu_leaves, leaves_buf_size, leaf_size, gpu_digests, gpu_indexes);

    // compute_internal_hashes<<<max_round_size / TPB, TPB>>>(gpu_digests, max_round, max_round_size, gpu_round_size, gpu_internal_indexes, TPB);

    int r = global_max_round;
    for (; global_round_size[r] > TPB; r--)
    {
        compute_internal_hashes_per_round<<<global_round_size[r] / TPB, TPB>>>(gpu_digests, global_round_size[r], r, global_max_round_size, gpu_internal_indexes);
    }

    CHECKCUDAERR(cudaMemcpy(global_digests_buf, gpu_digests, digests_size_bytes, cudaMemcpyDeviceToHost));

    // internal rounds on digest buffer on CPU -- for testing only!
    // for (int r = max_round-1; r > 0; r--) {

    for (; r > 0; r--)
    {
        for (int i = 0; i < global_round_size[r]; i++)
        {
            HashTask *ht = &global_internal_index[r * global_max_round_size + i];
            cpu_hash_two_ptr(global_digests_buf + (ht->left_index * HASH_SIZE_U64), global_digests_buf + (ht->right_index * HASH_SIZE_U64), global_digests_buf + (ht->target_index * HASH_SIZE_U64));
        }
    }

    // cap buffer (on CPU)
    for (int i = 0; i < global_round_size[0]; i++)
    {
        HashTask *ht = &global_internal_index[i];
        cpu_hash_two_ptr(global_digests_buf + (ht->left_index * HASH_SIZE_U64), global_digests_buf + (ht->right_index * HASH_SIZE_U64), global_cap_buf + (ht->target_index * HASH_SIZE_U64));
    }

    // free
    cudaFree(gpu_digests);
    cudaFree(gpu_indexes);
    cudaFree(gpu_leaves);
    cudaFree(gpu_internal_indexes);
    cudaFree(gpu_round_size);
}

void compute_size_per_gpu(u64 total, u64 ndev, u64 *per_dev, u64 *last_dev)
{
    u64 x_per_dev = total / ndev + 1;
    if (x_per_dev * ndev > total)
    {
        x_per_dev = total / ndev;
    }
    u64 x_last_dev = total - (ndev - 1) * x_per_dev;
    assert(x_last_dev <= total);
    *per_dev = x_per_dev;
    *last_dev = x_last_dev;
}

/*
void fill_digests_buf_in_rounds_in_c_on_multigpu(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height,
    u64 ngpus)
{
    u64 *gpu_leaves[16];
    u64 *gpu_digests[16];
    u64 *gpu_caps[16];
    u32 *gpu_indexes[16];
    u32 *gpu_round_size[16];
    HashTask *gpu_internal_indexes[16];
    cudaStream_t gpu_stream[16];

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    assert(ngpus <= nDevices);

    u64 leaves_per_gpu;
    u64 leaves_last_gpu;
    compute_size_per_gpu(leaves_buf_size, ngpus, &leaves_per_gpu, &leaves_last_gpu);
    u64 leaves_size_bytes_per_gpu = leaves_per_gpu * leaf_size * sizeof(u64);
    u64 leaves_size_bytes_last_gpu = leaves_last_gpu * leaf_size * sizeof(u64);

    for(int d = 0; d < ngpus; d++)
    {
        u64 cnt = (d == ngpus - 1) ? leaves_last_gpu : leaves_per_gpu;
        u64 sz = cnt * sizeof(u64);
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaMalloc(&gpu_leaves[d], sz));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_leaves[d], global_leaves_buf + d * cnt, sz, cudaMemcpyHostToDevice, gpu_stream[d]));
    }

    if (cap_buf_size == leaves_buf_size)
    {
        u64 digests_size_bytes = leaves_buf_size * HASH_SIZE_U64 * sizeof(u64);

        u64 digests_per_gpu;
        u64 digests_last_gpu;
        compute_size_per_gpu(leaves_buf_size, ngpus, &digests_per_gpu, &digests_last_gpu);
        for(int d = 0; d < ngpus; d++)
        {
            u64 digests_cnt = (d == ngpus - 1) ? digests_last_gpu : digests_per_gpu;
            u64 digests_size_bytes = digests_cnt * HASH_SIZE_U64 * sizeof(u64);
            u64 leaves_cnt = (d == ngpus - 1) ? leaves_last_gpu : leaves_per_gpu;
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaMalloc(&gpu_digests[d], digests_size_bytes));
            compute_leaves_hashes_direct<<<leaves_cnt / TPB + 1, TPB, 0, gpu_stream[d]>>>(gpu_leaves[d], leaves_cnt, leaf_size, gpu_digests[d]);
            CHECKCUDAERR(cudaMemcpyAsync(global_cap_buf + d * digests_cnt * HASH_SIZE, gpu_digests[d], digests_size_bytes, cudaMemcpyDeviceToHost, gpu_stream[d]));
        }
        for(int d = 0; d < ngpus; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
            cudaFree(gpu_digests[d]);
            cudaFree(gpu_leaves[d]);
            CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
        }
        return;
    }

    // 1.1 copy leaves from CPU to GPU
    // 1.2 (in parallel) run fill_tree_get_index on CPU
    // 2.1 compute leaf hashes on GPU
    // 2.2 (in parallel) copy task index data to GPU
    // 3. compute internal hashes on GPU
    // 4. copy data from GPU to CPU
    // 5. compute cap hashes on CPU

    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    // 1.2 (in parallel) run fill_tree_get_index on CPU
    for (u64 k = 0; k < cap_buf_size; k++)
    {
        fill_subtree_get_index(k, k * subtree_digests_len, subtree_digests_len, k * subtree_leaves_len, subtree_leaves_len, leaf_size, 0);
    }

    u64 digests_per_gpu;
    u64 digests_last_gpu;
    compute_size_per_gpu(digests_buf_size, ngpus, &digests_per_gpu, &digests_last_gpu);
    u64 caps_per_gpu;
    u64 caps_last_gpu;
    compute_size_per_gpu(cap_buf_size, ngpus, &caps_per_gpu, &caps_last_gpu);

    for(int d = 0; d < ngpus; d++)
    {
        u64 digests_size = ((d == ngpus - 1) ? digests_last_gpu : digests_per_gpu) * HASH_SIZE_U64 * sizeof(u64);
        u64 leaves_cnt = (d == ngpus - 1) ? leaves_last_gpu : leaves_per_gpu;

        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaMalloc(&gpu_digests[d], digests_size));
        CHECKCUDAERR(cudaMalloc(&gpu_indexes[d], leaves_cnt * sizeof(u32)));
        CHECKCUDAERR(cudaMalloc(&gpu_internal_indexes[d], (global_max_round + 1) * global_max_round_size * sizeof(HashTask)));
        CHECKCUDAERR(cudaMalloc(&gpu_round_size[d], (global_max_round + 1) * sizeof(u32)));
    }
    if (cap_buf_size > (1 << 20))
    {
        for(int d = 0; d < ngpus; d++)
        {
            u64 caps_size = ((d == ngpus - 1) ? caps_last_gpu : caps_per_gpu) * HASH_SIZE_U64 * sizeof(u64);
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaMalloc(&gpu_caps[d], caps_size));
        }
    }
    else
    {
        u32 caps_size_bytes = cap_buf_size * HASH_SIZE_U64 * sizeof(u64);
        CHECKCUDAERR(cudaSetDevice(0));
        CHECKCUDAERR(cudaMalloc(&gpu_caps[0], caps_size_bytes));
    }

    // 2.1 compute leaf hashes on GPU
    for(int d = 0; d < ngpus; d++)
    {
        u64 leaves_cnt = (d == ngpus - 1) ? leaves_last_gpu : leaves_per_gpu;
        u64 digests_size_bytes = leaves_cnt * HASH_SIZE_U64 * sizeof(u64);

        compute_leaves_hashes_direct<<<leaves_cnt / TPB + 1, TPB, 0, gpu_stream[d]>>>(gpu_leaves[d], leaves_cnt, leaf_size, gpu_digests[d]);
        CHECKCUDAERR(cudaMemcpyAsync(global_cap_buf + d * leaves_cnt * HASH_SIZE_U64, gpu_digests[d], digests_size_bytes, cudaMemcpyDeviceToHost, gpu_stream[d]));

        CHECKCUDAERR(cudaMemcpyAsync(gpu_indexes[d], global_leaf_index + d * leaves_cnt, leaves_cnt * sizeof(u32), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync(gpu_round_size[d], global_round_size + d, (global_max_round + 1) * sizeof(u32), cudaMemcpyHostToDevice, gpu_stream[d]));
        CHECKCUDAERR(cudaMemcpyAsync((void *)gpu_internal_indexes, (void *)global_internal_index, (global_max_round + 1) * global_max_round_size * sizeof(HashTask), cudaMemcpyHostToDevice, gpu_stream[d]));
        compute_leaves_hashes<<<leaves_cnt / TPB + 1, TPB, 0, gpu_stream[d]>>>(gpu_leaves[d], leaves_cnt, leaf_size, gpu_digests[d], gpu_indexes[d], d * lea);
    }

    int r = global_max_round;
    for (; global_round_size[r] > TPB; r--)
    {
        compute_internal_hashes_per_round<<<global_round_size[r] / TPB, TPB>>>(gpu_digests, global_round_size[r], r, global_max_round_size, gpu_internal_indexes);
    }

    for (; r > 0; r--)
    {
        compute_internal_hashes_per_round<<<1, global_round_size[r]>>>(gpu_digests, global_round_size[r], r, global_max_round_size, gpu_internal_indexes);
    }

    CHECKCUDAERR(cudaMemcpyAsync(global_digests_buf, gpu_digests, digests_size_bytes, cudaMemcpyDeviceToHost));

    compute_caps_hashes_per_round<<<1, global_round_size[0]>>>(gpu_caps, gpu_digests, global_round_size[0], 0, global_max_round_size, gpu_internal_indexes);

    CHECKCUDAERR(cudaMemcpy(global_cap_buf, gpu_caps, caps_size_bytes, cudaMemcpyDeviceToHost));

    // free
    cudaFree(gpu_digests);
    cudaFree(gpu_caps);
    cudaFree(gpu_indexes);
    cudaFree(gpu_leaves);
    cudaFree(gpu_internal_indexes);
    cudaFree(gpu_round_size);
}
*/

void fill_digests_buf_linear_gpu(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height)
{
    // 1. copy leaves from CPU to GPU
    // 2. compute leaf hashes on GPU
    // 3. compute internal hashes on GPU
    // 4. copy data from GPU to CPU
    // 5. internal hashes on CPU
    // 6. compute cap hashes on CPU

    u64 *gpu_leaves;
    u64 *gpu_digests;

    // 1. copy leaves from CPU to GPU
    u64 leaves_size_bytes = leaves_buf_size * leaf_size * 8;
    CHECKCUDAERR(cudaMalloc(&gpu_leaves, leaves_size_bytes));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_leaves, global_leaves_buf, leaves_size_bytes, cudaMemcpyHostToDevice));

    // 2.1. (special case) compute leaf hashes on GPU
    if (cap_buf_size == leaves_buf_size)
    {
        u64 digests_size_bytes = cap_buf_size * HASH_SIZE_U64 * sizeof(u64);
        CHECKCUDAERR(cudaMalloc(&gpu_digests, digests_size_bytes));
        compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>(gpu_leaves, leaves_buf_size, leaf_size, gpu_digests);
        CHECKCUDAERR(cudaMemcpy(global_cap_buf, gpu_digests, digests_size_bytes, cudaMemcpyDeviceToHost));

        // free
        cudaFree(gpu_digests);
        cudaFree(gpu_leaves);

        return;
    }

    // 2. compute leaf hashes on GPU
    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    u64 digests_size_bytes = digests_buf_size * HASH_SIZE_U64 * sizeof(u64);
    CHECKCUDAERR(cudaMalloc(&gpu_digests, digests_size_bytes));

    // 2.2. (special cases) compute leaf hashes on CPU
    if (subtree_leaves_len <= 2)
    {
        // for all the subtrees
#pragma omp parallel for
        for (u64 k = 0; k < cap_buf_size; k++)
        {
            // printf("Subtree %d\n", k);
            u64 *leaves_buf_ptr = global_leaves_buf + k * subtree_leaves_len * leaf_size;
            u64 *digests_buf_ptr = global_digests_buf + k * subtree_digests_len * HASH_SIZE_U64;
            u64 *cap_buf_ptr = global_cap_buf + k * HASH_SIZE_U64;

            // if one leaf => return its hash
            if (subtree_leaves_len == 1)
            {
                cpu_hash_one_ptr(leaves_buf_ptr, leaf_size, digests_buf_ptr);
                memcpy(cap_buf_ptr, digests_buf_ptr, HASH_SIZE);
            }
            else
            {
                // if two leaves => return their concat hash
                if (subtree_leaves_len == 2)
                {
                    cpu_hash_one_ptr(leaves_buf_ptr, leaf_size, digests_buf_ptr);
                    cpu_hash_one_ptr(leaves_buf_ptr + leaf_size, leaf_size, digests_buf_ptr + HASH_SIZE_U64);
                    cpu_hash_two_ptr(digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64, cap_buf_ptr);
                }
            }
        }
        // free
        cudaFree(gpu_digests);
        cudaFree(gpu_leaves);
        return;
    }

    // 2.3. (general case) compute leaf hashes on GPU
    compute_leaves_hashes_linear_all<<<leaves_buf_size / TPB + 1, TPB>>>(gpu_leaves, leaves_buf_size, leaf_size, gpu_digests, subtree_leaves_len, subtree_digests_len);

    // 3. compute internal hashes on GPU
    u64 r = (u64)log2(subtree_leaves_len) - 1;
    u64 last_index = subtree_digests_len - subtree_leaves_len;

    for (; (1 << r) * cap_buf_size > TPB; r--)
    {
        // printf("GPU Round %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<((1 << r) * cap_buf_size) / TPB + 1, TPB>>>(gpu_digests, (1 << r), last_index, cap_buf_size, subtree_digests_len);
    }

    for (; r > 0; r--)
    {
        // printf("GPU Round %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<1, (1 << r) * cap_buf_size>>>(gpu_digests, (1 << r), last_index, cap_buf_size, subtree_digests_len);
    }

    // 4. copy data from GPU to CPU
    CHECKCUDAERR(cudaMemcpyAsync(global_digests_buf, gpu_digests, digests_buf_size * HASH_SIZE, cudaMemcpyDeviceToHost));

    // 5. compute cap hashes on GPU (we reuse gpu_leaves buffer)
    if (cap_buf_size <= TPB)
    {
        compute_caps_hashes_linear<<<1, cap_buf_size>>>(gpu_leaves, gpu_digests, cap_buf_size, subtree_digests_len);
    }
    else
    {
        compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>(gpu_leaves, gpu_digests, cap_buf_size, subtree_digests_len);
    }

    // 6. copy data from GPU to CPU (we reuse gpu_leaves buffer)
    CHECKCUDAERR(cudaMemcpy(global_cap_buf, gpu_leaves, cap_buf_size * HASH_SIZE, cudaMemcpyDeviceToHost));

    // free
    cudaFree(gpu_digests);
    cudaFree(gpu_leaves);
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
    uint64_t hash_type)
{
    // TODO: take gpu_id as param
    CHECKCUDAERR(cudaSetDevice(0));

    init_gpu_functions(hash_type, 0);

    // (special case) compute leaf hashes on GPU
    if (cap_buf_size == leaves_buf_size)
    {
        compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)cap_buf_gpu_ptr);
        return;
    }

    // 2. compute leaf hashes on GPU
    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    // (special cases) compute leaf hashes on CPU
    if (subtree_leaves_len <= 2)
    {
        if (subtree_leaves_len == 1)
        {
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)cap_buf_gpu_ptr);
        }
        else
        {
            compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)digests_buf_gpu_ptr);
            if (cap_buf_size <= TPB)
            {
                compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len);
            }
            else
            {
                compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len);
            }
        }
        return;
    }

    // (general case) compute leaf hashes on GPU
    compute_leaves_hashes_linear_all<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)digests_buf_gpu_ptr, subtree_leaves_len, subtree_digests_len);

    // compute internal hashes on GPU
    u64 r = (u64)log2(subtree_leaves_len) - 1;
    u64 last_index = subtree_digests_len - subtree_leaves_len;

    for (; (1 << r) * cap_buf_size > TPB; r--)
    {
        // printf("GPU Round %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<((1 << r) * cap_buf_size) / TPB + 1, TPB>>>((u64 *)digests_buf_gpu_ptr, (1 << r), last_index, cap_buf_size, subtree_digests_len);
    }

    for (; r > 0; r--)
    {
        // printf("GPU Round %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<1, (1 << r) * cap_buf_size>>>((u64 *)digests_buf_gpu_ptr, (1 << r), last_index, cap_buf_size, subtree_digests_len);
    }

    // compute cap hashes on GPU
    if (cap_buf_size <= TPB)
    {
        compute_caps_hashes_linear<<<1, cap_buf_size>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len);
    }
    else
    {
        compute_caps_hashes_linear<<<cap_buf_size / TPB + 1, TPB>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, cap_buf_size, subtree_digests_len);
    }
}

void fill_digests_buf_linear_gpu_v1(
    u64 digests_buf_size,
    u64 cap_buf_size,
    u64 leaves_buf_size,
    u64 leaf_size,
    u64 cap_height)
{
    u64 *gpu_leaves;
    u64 *gpu_digests;

    u64 leaves_size_bytes = leaves_buf_size * leaf_size * 8;
    CHECKCUDAERR(cudaMalloc(&gpu_leaves, leaves_size_bytes));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_leaves, global_leaves_buf, leaves_size_bytes, cudaMemcpyHostToDevice));

    if (cap_buf_size == leaves_buf_size)
    {
        u64 digests_size_bytes = cap_buf_size * HASH_SIZE_U64 * sizeof(u64);
        CHECKCUDAERR(cudaMalloc(&gpu_digests, digests_size_bytes));
        compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>(gpu_leaves, leaves_buf_size, leaf_size, gpu_digests);
        CHECKCUDAERR(cudaMemcpy(global_cap_buf, gpu_digests, digests_size_bytes, cudaMemcpyDeviceToHost));

        // free
        cudaFree(gpu_digests);
        cudaFree(gpu_leaves);

        return;
    }

    // 1. copy leaves from CPU to GPU
    // 2. compute leaf hashes on GPU
    // 3. compute internal hashes on GPU
    // 4. copy data from GPU to CPU
    // 5. internal hashes on CPU
    // 6. compute cap hashes on CPU

    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    u64 digests_size_bytes = digests_buf_size * HASH_SIZE_U64 * sizeof(u64);
    CHECKCUDAERR(cudaMalloc(&gpu_digests, digests_size_bytes));

    // for all the subtrees
    for (u64 k = 0; k < cap_buf_size; k++)
    {
        // printf("Subtree %d, Leaves %lu, Digests %lu\n", k, subtree_leaves_len, subtree_digests_len);
        u64 *leaves_buf_ptr = global_leaves_buf + k * subtree_leaves_len * leaf_size;
        u64 *digests_buf_ptr = global_digests_buf + k * subtree_digests_len * HASH_SIZE_U64;
        u64 *cap_buf_ptr = global_cap_buf + k * HASH_SIZE_U64;

        // if one leaf => return it hash
        if (subtree_leaves_len == 1)
        {
            cpu_hash_one_ptr(leaves_buf_ptr, leaf_size, digests_buf_ptr);
            memcpy(cap_buf_ptr, digests_buf_ptr, HASH_SIZE);
            continue;
        }
        // if two leaves => return their concat hash
        if (subtree_leaves_len == 2)
        {
            cpu_hash_one_ptr(leaves_buf_ptr, leaf_size, digests_buf_ptr);
            cpu_hash_one_ptr(leaves_buf_ptr + leaf_size, leaf_size, digests_buf_ptr + HASH_SIZE_U64);
            cpu_hash_two_ptr(digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64, cap_buf_ptr);
            continue;
        }

        // 2. compute leaf hashes on GPU
        u64 *gpu_digests_chunk_ptr = gpu_digests + k * subtree_digests_len * HASH_SIZE_U64;
        u64 *gpu_digests_curr_ptr = gpu_digests_chunk_ptr + (subtree_digests_len - subtree_leaves_len) * HASH_SIZE_U64;
        u64 *gpu_leaves_chunk_ptr = gpu_leaves + k * subtree_leaves_len * leaf_size;
        compute_leaves_hashes_direct<<<subtree_leaves_len / TPB + 1, TPB>>>(gpu_leaves_chunk_ptr, subtree_leaves_len, leaf_size, gpu_digests_curr_ptr);

        // 3. compute internal hashes on GPU
        u64 r = (u64)log2(subtree_leaves_len) - 1;
        u64 last_index = subtree_digests_len - subtree_leaves_len;

        for (; (1 << r) > TPB; r--)
        {
            last_index -= (1 << r);
            // printf("GPU round %d\n", r);
            gpu_digests_curr_ptr = gpu_digests_chunk_ptr + last_index * HASH_SIZE_U64;
            compute_internal_hashes_linear<<<(1 << r) / TPB + 1, TPB>>>(gpu_digests_curr_ptr, (1 << r), last_index);
        }

        // 4. copy data from GPU to CPU
        CHECKCUDAERR(cudaMemcpy(digests_buf_ptr, gpu_digests_chunk_ptr, subtree_digests_len * HASH_SIZE, cudaMemcpyDeviceToHost));

        // 5. internal hashes on CPU
        for (; r > 0; r--)
        {
            last_index -= (1 << r);
            // printf("CPU round %d Last idx %d\n", r, last_index);
            u64 *digests_buf_ptr2 = digests_buf_ptr + last_index * HASH_SIZE_U64;
            for (int idx = 0; idx < (1 << r); idx++)
            {
                u64 left_idx = 2 * (idx + 1) + last_index;
                u64 right_idx = left_idx + 1;
                u64 *left_ptr = digests_buf_ptr2 + (left_idx * HASH_SIZE_U64);
                u64 *right_ptr = digests_buf_ptr2 + (right_idx * HASH_SIZE_U64);
                // printf("%lu %lu\n", *left_ptr, *right_ptr);
                cpu_hash_two_ptr(left_ptr, right_ptr, digests_buf_ptr2 + (idx * HASH_SIZE_U64));
            }
        }

        // 6. compute cap hashes on CPU
        cpu_hash_two_ptr(digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64, cap_buf_ptr);

    } // end for k

    // free
    cudaFree(gpu_digests);
    cudaFree(gpu_leaves);
}

void fill_digests_buf_linear_multigpu(
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height,
    uint64_t ngpus)
{
    u64 *gpu_leaves[16];
    u64 *gpu_digests[16];
    cudaStream_t gpu_stream[16];

    int nDevices = 0;
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    assert(ngpus <= nDevices);
    assert(ngpus <= 16);

    // (special case) compute leaf hashes on GPU
    if (cap_buf_size == leaves_buf_size)
    {
        u64 leaves_per_gpu;
        u64 leaves_last_gpu;
        compute_size_per_gpu(leaves_buf_size, ngpus, &leaves_per_gpu, &leaves_last_gpu);

#pragma omp parallel for num_threads(ngpus)
        for (int d = 0; d < ngpus; d++)
        {
            u64 leaves_cnt = (d == ngpus - 1) ? leaves_last_gpu : leaves_per_gpu;
            u64 sz = leaves_cnt * sizeof(u64);
            u64 digests_size_bytes = leaves_cnt * HASH_SIZE_U64 * sizeof(u64);
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
            CHECKCUDAERR(cudaMalloc(&gpu_leaves[d], sz));
            CHECKCUDAERR(cudaMalloc(&gpu_digests[d], digests_size_bytes));
            CHECKCUDAERR(cudaMemcpyAsync(gpu_leaves[d], global_leaves_buf + d * leaves_cnt, sz, cudaMemcpyHostToDevice, gpu_stream[d]));
            compute_leaves_hashes_direct<<<leaves_cnt / TPB + 1, TPB, 0, gpu_stream[d]>>>(gpu_leaves[d], leaves_cnt, leaf_size, gpu_digests[d]);
            CHECKCUDAERR(cudaMemcpyAsync(global_cap_buf + d * leaves_cnt * HASH_SIZE, gpu_digests[d], digests_size_bytes, cudaMemcpyDeviceToHost, gpu_stream[d]));
        }
        for (int d = 0; d < ngpus; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
            cudaFree(gpu_digests[d]);
            cudaFree(gpu_leaves[d]);
            CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
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

    // (special cases) compute leaf hashes on CPU
    if (subtree_leaves_len <= 2)
    {
        // for all the subtrees
#pragma omp parallel for
        for (u64 k = 0; k < cap_buf_size; k++)
        {
            // printf("Subtree %d\n", k);
            u64 *leaves_buf_ptr = global_leaves_buf + k * subtree_leaves_len * leaf_size;
            u64 *digests_buf_ptr = global_digests_buf + k * subtree_digests_len * HASH_SIZE_U64;
            u64 *cap_buf_ptr = global_cap_buf + k * HASH_SIZE_U64;

            // if one leaf => return it hash
            if (subtree_leaves_len == 1)
            {
                cpu_hash_one_ptr(leaves_buf_ptr, leaf_size, digests_buf_ptr);
                memcpy(cap_buf_ptr, digests_buf_ptr, HASH_SIZE);
            }
            else
            {
                // if two leaves => return their concat hash
                if (subtree_leaves_len == 2)
                {
                    cpu_hash_one_ptr(leaves_buf_ptr, leaf_size, digests_buf_ptr);
                    cpu_hash_one_ptr(leaves_buf_ptr + leaf_size, leaf_size, digests_buf_ptr + HASH_SIZE_U64);
                    cpu_hash_two_ptr(digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64, cap_buf_ptr);
                }
            }
        }
        return;
    }

    // we compute one sub-tree on each GPU
    for (int d = 0; d < ngpus; d++)
    {
        u64 leaves_size_bytes = subtree_leaves_len * leaf_size * sizeof(u64);
        u64 digests_size_bytes = subtree_digests_len * HASH_SIZE_U64 * sizeof(u64);
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaStreamCreate(gpu_stream + d));
        CHECKCUDAERR(cudaMalloc(&(gpu_leaves[d]), leaves_size_bytes));
        CHECKCUDAERR(cudaMalloc(&(gpu_digests[d]), digests_size_bytes));
    }

    for (int k = 0; k < cap_buf_size; k += ngpus)
    {
#pragma omp parallel for num_threads(ngpus)
        for (int d = 0; d < ngpus; d++)
        {
            if (k + d >= cap_buf_size)
            {
                continue;
            }

            u64 leaves_size_bytes = subtree_leaves_len * leaf_size * sizeof(u64);
            u64 digests_size_bytes = subtree_digests_len * HASH_SIZE_U64 * sizeof(u64);

            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaMemcpyAsync(gpu_leaves[d], global_leaves_buf + (k + d) * subtree_leaves_len * leaf_size, leaves_size_bytes, cudaMemcpyHostToDevice, gpu_stream[d]));

            int blocks = (subtree_leaves_len % TPB == 0) ? subtree_leaves_len / TPB : subtree_leaves_len / TPB + 1;
            int threads = (subtree_leaves_len < TPB) ? subtree_leaves_len : TPB;
            compute_leaves_hashes_linear_per_gpu<<<blocks, threads, 0, gpu_stream[d]>>>(gpu_leaves[d], leaf_size, gpu_digests[d], subtree_leaves_len, subtree_digests_len);

            u64 r = (u64)log2(subtree_leaves_len) - 1;
            u64 last_index = subtree_digests_len - subtree_leaves_len;

            for (; (1 << r) > TPB; r--)
            {
                last_index -= (1 << r);
                compute_internal_hashes_linear_per_gpu<<<(1 << r) / TPB + 1, TPB, 0, gpu_stream[d]>>>(gpu_digests[d], (1 << r), last_index);
            }
            for (; r > 0; r--)
            {
                last_index -= (1 << r);
                compute_internal_hashes_linear_per_gpu<<<1, (1 << r), 0, gpu_stream[d]>>>(gpu_digests[d], (1 << r), last_index);
            }
            CHECKCUDAERR(cudaMemcpyAsync(global_digests_buf + (k + d) * subtree_digests_len * HASH_SIZE_U64, gpu_digests[d], digests_size_bytes, cudaMemcpyDeviceToHost, gpu_stream[d]));
        }
        // sync
        for (int d = 0; d < ngpus; d++)
        {
            CHECKCUDAERR(cudaSetDevice(d));
            CHECKCUDAERR(cudaStreamSynchronize(gpu_stream[d]));
        }
    }

    // compute cap hashes on CPU
#pragma omp parallel for
    for (int k = 0; k < cap_buf_size; k++)
    {
        u64 *digests_buf_ptr = global_digests_buf + k * subtree_digests_len * HASH_SIZE_U64;
        u64 *cap_buf_ptr = global_cap_buf + k * HASH_SIZE_U64;
        cpu_hash_two_ptr(digests_buf_ptr, digests_buf_ptr + HASH_SIZE_U64, cap_buf_ptr);
    }

    // free
    for (int d = 0; d < ngpus; d++)
    {
        cudaFree(gpu_digests[d]);
        cudaFree(gpu_leaves[d]);
        CHECKCUDAERR(cudaStreamDestroy(gpu_stream[d]));
    }
}

void fill_digests_buf_in_rounds_in_c_on_gpu_with_gpu_ptr(
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
    init_gpu_functions(hash_type);
    fill_init_rounds(leaves_buf_size, log2(leaves_buf_size) + 1);

    if (cap_buf_size == leaves_buf_size)
    {
        compute_leaves_hashes_direct<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)cap_buf_gpu_ptr);
        return;
    }

    // 1. run fill_tree_get_index on CPU
    // 2.1 compute leaf hashes on GPU
    // 2.2 (in parallel) copy task index data to GPU
    // 3. compute internal hashes on GPU
    // 4. compute cap hashes on GPU

    u64 subtree_digests_len = digests_buf_size >> cap_height;
    u64 subtree_leaves_len = leaves_buf_size >> cap_height;
    u64 digests_chunks = digests_buf_size / subtree_digests_len;
    u64 leaves_chunks = leaves_buf_size / subtree_leaves_len;
    assert(digests_chunks == cap_buf_size);
    assert(digests_chunks == leaves_chunks);

    // 1. run fill_tree_get_index on CPU
    for (u64 k = 0; k < cap_buf_size; k++)
    {
        fill_subtree_get_index(k, k * subtree_digests_len, subtree_digests_len, k * subtree_leaves_len, subtree_leaves_len, leaf_size, 0);
    }

    u32 *gpu_indexes;
    u32 *gpu_round_size;
    HashTask *gpu_internal_indexes;

    CHECKCUDAERR(cudaMalloc(&gpu_indexes, leaves_buf_size * sizeof(u32)));
    CHECKCUDAERR(cudaMalloc(&gpu_internal_indexes, (global_max_round + 1) * global_max_round_size * sizeof(HashTask)));
    CHECKCUDAERR(cudaMalloc(&gpu_round_size, (global_max_round + 1) * sizeof(u32)));

    // 2.1 and 2.2
    CHECKCUDAERR(cudaMemcpyAsync(gpu_indexes, global_leaf_index, leaves_buf_size * sizeof(u32), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpyAsync(gpu_round_size, global_round_size, (global_max_round + 1) * sizeof(u32), cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpyAsync((void *)gpu_internal_indexes, (void *)global_internal_index, (global_max_round + 1) * global_max_round_size * sizeof(HashTask), cudaMemcpyHostToDevice));
    compute_leaves_hashes<<<leaves_buf_size / TPB + 1, TPB>>>((u64 *)leaves_buf_gpu_ptr, leaves_buf_size, leaf_size, (u64 *)digests_buf_gpu_ptr, gpu_indexes);

    // 3.
    int r = global_max_round;
    for (; global_round_size[r] > TPB; r--)
    {
        compute_internal_hashes_per_round<<<global_round_size[r] / TPB, TPB>>>((u64 *)digests_buf_gpu_ptr, global_round_size[r], r, global_max_round_size, gpu_internal_indexes);
    }
    for (; r > 0; r--)
    {
        compute_internal_hashes_per_round<<<1, global_round_size[r]>>>((u64 *)digests_buf_gpu_ptr, global_round_size[r], r, global_max_round_size, gpu_internal_indexes);
    }

    // 4.
    compute_caps_hashes_per_round<<<1, global_round_size[0]>>>((u64 *)cap_buf_gpu_ptr, (u64 *)digests_buf_gpu_ptr, global_round_size[0], 0, global_max_round_size, gpu_internal_indexes);

    // free
    cudaFree(gpu_indexes);
    cudaFree(gpu_internal_indexes);
    cudaFree(gpu_round_size);

    fill_delete_rounds();
}

// #define TESTING
#ifdef TESTING

#define LEAF_SIZE_U64 135

#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include "test_leaves.h"

void read_leaves(u64 *leaves, u32 n_leaves, u32 leaf_size)
{
    FILE *f = fopen("/home/ubuntu/git/zkdex-plonky2-circom-poc/leaves16.bin", "rb");
    assert(f != NULL);
    for (u32 i = 0; i < n_leaves * leaf_size; i++)
    {
        u32 n = fread(leaves + i, 8, 1, f);
        if (n != 1)
        {
            printf("Error reading binary file!");
        }
    }

    fclose(f);
}

int main0()
{
    struct timeval t0, t1;

    u32 cap_h = 0;
    u64 n_caps = (1 << cap_h);
    u64 n_leaves = 1048576;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 leaf_size = 16;
    u64 rounds = log2(n_digests) + 1;

    printf("Digests %lu, Leaves %lu, Caps %lu, Leaf size %lu\n", n_digests, n_leaves, n_caps, leaf_size);

    fill_init(n_digests, n_leaves, n_caps, leaf_size, HASH_SIZE_U64, 2);
    init_gpu_functions(2);
    fill_init_rounds(n_leaves, rounds);
    read_leaves(global_leaves_buf, n_leaves, leaf_size);

    gettimeofday(&t0, 0);
    fill_digests_buf_in_rounds_in_c_on_gpu(n_digests, n_caps, n_leaves, leaf_size, cap_h);
    gettimeofday(&t1, 0);
    long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on GPU: %ld us\n", elapsed);

    fill_delete_rounds();
    fill_delete();

    return 0;
}

/*
__global__ void compute_sbox_gpu(u64 *input, u64 *output, u32 count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count)
        return;
    gl64_t tmp = input[tid];
    output[tid] = PoseidonPermutationGPU::sbox_monomial(tmp);
}

void compute_sbox_driver(u64 *input, u32 size)
{
    u64 *cpu_ret = (u64 *)malloc(size * sizeof(u64));
    assert(cpu_ret != NULL);
    u64 *gpu_ret = (u64 *)malloc(size * sizeof(u64));
    assert(gpu_ret != NULL);
    u64 *gpu_in, *gpu_out;
    CHECKCUDAERR(cudaMalloc(&gpu_in, size * sizeof(u64)));
    CHECKCUDAERR(cudaMalloc(&gpu_out, size * sizeof(u64)));
    CHECKCUDAERR(cudaMemcpy(gpu_in, input, size * sizeof(u64), cudaMemcpyHostToDevice));
    compute_sbox_gpu<<<size / 32 + 1, 32>>>(gpu_in, gpu_out, size);
    CHECKCUDAERR(cudaMemcpy(gpu_ret, gpu_out, size * sizeof(u64), cudaMemcpyDeviceToHost));

    for (u32 i = 0; i < size; i++)
    {
        GoldilocksField tmp = GoldilocksField(input[i]);
        tmp = PoseidonPermutation::sbox_monomial(tmp);
        cpu_ret[i] = tmp.to_noncanonical_u64();
        if (cpu_ret[i] != gpu_ret[i])
        {
            printf("Diff at %d: %lx %lx\n", i, cpu_ret[i], gpu_ret[i]);
        }
    }

    cudaFree(gpu_in);
    cudaFree(gpu_out);
    free(cpu_ret);
    free(gpu_ret);
}

__global__ void compute_mds(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 *digest_idx)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= leaves_count)
        return;

    gl64_t leaf_elements[12];
    gl64_t hash[4];
    u64 *ptr = leaves + (tid * leaf_size);
    for (u32 i = 0; i < 12; i++)
    {
        leaf_elements[i] = ptr[i % 7];
    }
    PoseidonPermutationGPU perm = PoseidonPermutationGPU();
    perm.set_from_slice(leaf_elements, 12, 0);
    perm.mds_layer(perm.state, leaf_elements);
    ptr = digests_buf + digest_idx[tid] * 4;
    for (u32 i = 0; i < 4; i++)
    {
        ptr[i] = leaf_elements[i];
    }
}
*/

/*
 * Compute the hashes of the leaves only. Compare with CPU results.
 */
void run_gpu_leaves_hashes()
{
    u32 n_leaves = 8;
    u32 leaf_size = 7;

    global_leaves_buf = (u64 *)test_leaves_8;

    // Compute on CPU
    global_digests_buf = (u64 *)malloc(n_leaves * HASH_SIZE_U64 * sizeof(u64));
    assert(global_digests_buf != NULL);

    for (int i = 0; i < n_leaves; i++)
    {
        u64 *dptr = global_digests_buf + i * HASH_SIZE_U64;
        u64 *lptr = global_leaves_buf + i * leaf_size;
        cpu_hash_one_ptr(lptr, leaf_size, dptr);
    }

    // Compute on GPU
    u64 *gpu_leaves;
    u64 *gpu_digests;
    u32 *gpu_indexes;
    u32 *cpu_indexes;

    u32 leaves_size_bytes = n_leaves * leaf_size * 8;
    u32 digests_size_bytes = n_leaves * HASH_SIZE_U64 * sizeof(u64);
    CHECKCUDAERR(cudaMalloc(&gpu_leaves, leaves_size_bytes));
    CHECKCUDAERR(cudaMalloc(&gpu_indexes, n_leaves * sizeof(u32)));
    CHECKCUDAERR(cudaMalloc(&gpu_digests, digests_size_bytes));

    CHECKCUDAERR(cudaMemcpy(gpu_leaves, global_leaves_buf, leaves_size_bytes, cudaMemcpyHostToDevice));

    cpu_indexes = (u32 *)malloc(n_leaves * sizeof(u32));
    assert(cpu_indexes != NULL);
    for (int i = 0; i < n_leaves; i++)
    {
        cpu_indexes[i] = i;
    }
    CHECKCUDAERR(cudaMemcpy(gpu_indexes, cpu_indexes, n_leaves * sizeof(u32), cudaMemcpyHostToDevice));

    compute_leaves_hashes<<<n_leaves / TPB + 1, TPB>>>(gpu_leaves, n_leaves, leaf_size, gpu_digests, gpu_indexes);

    u64 *digests_buf2 = (u64 *)malloc(n_leaves * HASH_SIZE_U64 * sizeof(u64));
    CHECKCUDAERR(cudaMemcpy(digests_buf2, gpu_digests, digests_size_bytes, cudaMemcpyDeviceToHost));

    bool is_diff = false;
    for (int i = 0; i < n_leaves; i++)
    {
        for (int j = 0; j < HASH_SIZE_U64; j++)
        {
            if (global_digests_buf[i * HASH_SIZE_U64 + j] != digests_buf2[i * HASH_SIZE_U64 + j])
            {
                printf("Diff at %d %d %lx %lx\n", i, j, global_digests_buf[i * HASH_SIZE_U64 + j], digests_buf2[i * HASH_SIZE_U64 + j]);
                is_diff = true;
            }
        }
    }
    if (!is_diff)
    {
        printf("Poseidon hash test done. No diff!\n");
    }

    /*
        for (int i = 0; i < n_leaves; i++) {
            for (int j = 0; j < HASH_SIZE_U64; j++) {
                if (global_leaves_buf[i * leaf_size + j] != digests_buf2[i * HASH_SIZE_U64 + j]) {
                    printf("Diff at %d %d %lx %lx\n", i, j, global_leaves_buf[i * leaf_size + j], digests_buf2[i * HASH_SIZE_U64 + j]);
                }
            }
        }
    */

    // Free and cleanup
    cudaFree(gpu_leaves);
    cudaFree(gpu_indexes);
    cudaFree(gpu_digests);

    free(digests_buf2);
    free(cpu_indexes);
    free(global_digests_buf);
}

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
void run_gpu_cpu_verify()
{
    struct timeval t0, t1;

    u64 n_caps = 2;
    u64 n_leaves = (1 << 14);
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = log2(n_digests) + 1;
    u64 cap_h = log2(n_caps);

    global_digests_buf = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    global_cap_buf = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    // global_leaves_buf = (u64 *)test_leaves_1024;
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
    printf("Number of leaves: %ld\n", n_leaves);

    init_gpu_functions(0);

    gettimeofday(&t0, 0);
    fill_digests_buf_linear_gpu_v1(n_digests, n_caps, n_leaves, 7, cap_h);
    gettimeofday(&t1, 0);
    long elapsed = (t1.tv_sec - t0.tv_sec) * 1000000 + t1.tv_usec - t0.tv_usec;
    printf("Time on GPU: %ld us\n", elapsed);

    // fill_init_rounds(n_leaves, rounds);
    // fill_digests_buf_in_rounds_in_c_on_gpu(n_digests, n_caps, n_leaves, 7, cap_h);
    // fill_digests_buf_in_rounds_in_c(n_digests, n_caps, n_leaves, 7, cap_h);
    /*
        printf("After fill...\n");
        for (int i = 0; i < n_digests; i++)
            print_hash(global_digests_buf + i * 4);

        printf("Cap...\n");
        for (int i = 0; i < n_caps; i++)
            print_hash(global_cap_buf + i * 4);
    */

    u64 *digests_buf2 = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    memcpy(digests_buf2, global_digests_buf, n_digests * HASH_SIZE_U64 * sizeof(u64));
    u64 *cap_buf2 = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    memcpy(cap_buf2, global_cap_buf, n_caps * HASH_SIZE_U64 * sizeof(u64));

    // fill_digests_buf_in_c(n_digests, n_caps, n_leaves, 7, cap_h);
    fill_digests_buf_linear_cpu(n_digests, n_caps, n_leaves, 7, cap_h);

    compare_results(global_digests_buf, digests_buf2, n_digests, global_cap_buf, cap_buf2, n_caps);
    /*
        printf("After fill...\n");
        for (int i = 0; i < n_digests; i++)
            print_hash(global_digests_buf + i * 4);

        printf("Cap...\n");
        for (int i = 0; i < n_caps; i++)
            print_hash(global_cap_buf + i * 4);
    */

    // fill_delete_rounds();

    free(global_digests_buf);
    free(digests_buf2);
    free(global_cap_buf);
    free(cap_buf2);
}

void run_gpu_test()
{
    u64 n_caps = 1024;
    u64 n_leaves = 1024;
    u64 n_digests = 2 * (n_leaves - n_caps);
    u64 rounds = 10;

    global_digests_buf = (u64 *)malloc(n_digests * HASH_SIZE_U64 * sizeof(u64));
    global_cap_buf = (u64 *)malloc(n_caps * HASH_SIZE_U64 * sizeof(u64));
    assert(global_digests_buf != NULL && global_cap_buf != NULL);
    global_leaves_buf = (u64 *)test_leaves_8;
    global_cap_buf_end = global_cap_buf + n_caps * HASH_SIZE_U64;
    global_digests_buf_end = global_digests_buf + n_digests * HASH_SIZE_U64;
    global_leaves_buf_end = global_leaves_buf + n_leaves * 7;

    fill_init_rounds(n_leaves, rounds);

    fill_digests_buf_in_rounds_in_c_on_gpu(n_digests, n_caps, n_leaves, 7, 1);

    fill_delete_rounds();

    free(global_digests_buf);
    free(global_cap_buf);
}

/*
 * Run on GPU, CPU (1 core), CPU (multi-core) and get the execution times.
 */
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

int main(int argc, char **argv)
{
    int size = 10;
    if (argc > 1)
    {
        size = atoi(argv[1]);
    }
    assert(3 <= size);

    // run_gpu_cpu_verify();

    run_gpu_cpu_comparison(size);

    return 0;
}

#endif // TESTING
