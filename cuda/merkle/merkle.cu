#include "int_types.h"

#include "merkle.h"
#include "merkle_private.h"

#include "cuda_utils.cuh"
#include "poseidon.cuh"
#include "poseidon.h"
#include "poseidon2.h"
#include "poseidon_bn128.h"
#include "keccak.h"

// TODO - benchmark and select best TPB
#define TPB 64

/*
 * Selectors of GPU hash functions.
 */
__device__ void gpu_hash_one(gl64_t *lptr, uint32_t leaf_size, gl64_t *dptr, uint64_t hash_type)
{
    switch (hash_type)
    {
    case 0:
        return gpu_poseidon_hash_one(lptr, leaf_size, dptr);
    case 1:
        return gpu_keccak_hash_one(lptr, leaf_size, dptr);
    case 2:
        return gpu_poseidon_bn128_hash_one(lptr, leaf_size, dptr);
    case 3:
        return gpu_poseidon2_hash_one(lptr, leaf_size, dptr);
    default:
        return;
    }
}

__device__ void gpu_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash, uint64_t hash_type)
{
    switch (hash_type)
    {
    case 0:
        return gpu_poseidon_hash_two(hash1, hash2, hash);
    case 1:
        return gpu_keccak_hash_two(hash1, hash2, hash);
    case 2:
        return gpu_poseidon_bn128_hash_two(hash1, hash2, hash);
    case 3:
        return gpu_poseidon2_hash_two(hash1, hash2, hash);
    default:
        return;
    }
}

/*
 * Compute only leaves hashes with direct mapping (digest i corresponds to leaf i).
 */
__global__ void compute_leaves_hashes_direct(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, uint64_t hash_type)
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
__global__ void compute_leaves_hashes_linear_all(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 subtree_leaves_len, u32 subtree_digests_len, uint64_t hash_type)
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
__global__ void compute_leaves_hashes_linear_per_gpu(u64 *leaves, u32 leaf_size, u64 *digests_buf, u32 subtree_leaves_len, u32 subtree_digests_len, uint64_t hash_type)
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
__global__ void compute_leaves_hashes(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 *digest_idx, uint64_t hash_type)
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
__global__ void compute_leaves_hashes_offset(u64 *leaves, u32 leaves_count, u32 leaf_size, u64 *digests_buf, u32 *digest_idx, u64 offset, uint64_t hash_type)
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
__global__ void compute_internal_hashes_linear_all(u64 *digests_buf, u32 round_size, u32 last_idx, u32 subtree_count, u32 subtree_digests_len, uint64_t hash_type)
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

__global__ void compute_internal_hashes_linear_per_gpu(u64 *digests_buf, u32 round_size, u32 last_idx, uint64_t hash_type)
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

__global__ void compute_caps_hashes_linear(u64 *caps_buf, u64 *digests_buf, u64 cap_buf_size, u64 subtree_digests_len, uint64_t hash_type)
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
    uint64_t digests_buf_size,
    uint64_t cap_buf_size,
    uint64_t leaves_buf_size,
    uint64_t leaf_size,
    uint64_t cap_height,
    uint64_t hash_type,
    uint64_t gpu_id)
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

    for (; (1 << r) * cap_buf_size > TPB; r--)
    {
        // printf("GPU Round %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<((1 << r) * cap_buf_size) / TPB + 1, TPB>>>((u64 *)digests_buf_gpu_ptr, (1 << r), last_index, cap_buf_size, subtree_digests_len, hash_type);
    }

    for (; r > 0; r--)
    {
        // printf("GPU Round %u\n", r);
        last_index -= (1 << r);
        compute_internal_hashes_linear_all<<<1, (1 << r) * cap_buf_size>>>((u64 *)digests_buf_gpu_ptr, (1 << r), last_index, cap_buf_size, subtree_digests_len, hash_type);
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
