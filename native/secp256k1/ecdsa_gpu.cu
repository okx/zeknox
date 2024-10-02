/*************************************************************************
 * Written in 2020-2022 by Elichai Turkel                                *
 * To the extent possible under law, the author(s) have dedicated all    *
 * copyright and related and neighboring rights to the software in this  *
 * file to the public domain worldwide. This software is distributed     *
 * without any warranty. For the CC0 Public Domain Dedication, see       *
 * EXAMPLES_COPYING or https://creativecommons.org/publicdomain/zero/1.0 *
 *************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <sys/time.h>
#include <sys/random.h>

#include "ecmult_gen.h"
#include "precomputed_ecmult.h"
#include "secp256k1_local.h"

#include "../utils/cuda_utils.cuh"

// #include "examples_util.h"

static void secure_erase(void *ptr, size_t len)
{
#if defined(_MSC_VER)
    /* SecureZeroMemory is guaranteed not to be optimized out by MSVC. */
    SecureZeroMemory(ptr, len);
#elif defined(__GNUC__)
    /* We use a memory barrier that scares the compiler away from optimizing out the memset.
     *
     * Quoting Adam Langley <agl@google.com> in commit ad1907fe73334d6c696c8539646c21b11178f20f
     * in BoringSSL (ISC License):
     *    As best as we can tell, this is sufficient to break any optimisations that
     *    might try to eliminate "superfluous" memsets.
     * This method used in memzero_explicit() the Linux kernel, too. Its advantage is that it is
     * pretty efficient, because the compiler can still implement the memset() efficiently,
     * just not remove it entirely. See "Dead Store Elimination (Still) Considered Harmful" by
     * Yang et al. (USENIX Security 2017) for more background.
     */
    memset(ptr, 0, len);
    __asm__ __volatile__("" : : "r"(ptr) : "memory");
#else
    void *(*volatile const volatile_memset)(void *, int, size_t) = memset;
    volatile_memset(ptr, 0, len);
#endif
}

/* Returns 1 on success, and 0 on failure. */
static int fill_random(unsigned char *data, size_t size)
{
#if defined(_WIN32)
    NTSTATUS res = BCryptGenRandom(NULL, data, size, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (res != STATUS_SUCCESS || size > ULONG_MAX)
    {
        return 0;
    }
    else
    {
        return 1;
    }
#elif defined(__linux__) || defined(__FreeBSD__)
    /* If `getrandom(2)` is not available you should fallback to /dev/urandom */
    ssize_t res = getrandom(data, size, 0);
    if (res < 0 || (size_t)res != size)
    {
        return 0;
    }
    else
    {
        return 1;
    }
#elif defined(__APPLE__) || defined(__OpenBSD__)
    /* If `getentropy(2)` is not available you should fallback to either
     * `SecRandomCopyBytes` or /dev/urandom */
    int res = getentropy(data, size);
    if (res == 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
#endif
    return 0;
}

static void print_hex(unsigned char *data, size_t size)
{
    size_t i;
    printf("0x");
    for (i = 0; i < size; i++)
    {
        printf("%02x", data[i]);
    }
    printf("\n");
}

__global__ void secp256k1_ecdsa_sign_batch(int batch_size, const secp256k1_context *ctxs, secp256k1_ecdsa_signature *signatures, const unsigned char *msghash32s, const unsigned char *seckeys, int *rets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    rets[tid] = secp256k1_ecdsa_sign(ctxs, signatures + tid, msghash32s + tid * 32, seckeys + tid * 32, NULL, NULL);
}

int main(void)
{
    constexpr uint32_t batch_size = (1 << 14);
    constexpr uint32_t hash_size = 32;
    constexpr uint32_t seckey_size = 32;
    constexpr uint32_t compressed_pubkey_size = 33;
    constexpr uint32_t serialized_signature_size = 64;

    // allocate CPU data
    unsigned char *cpu_msg_hash = (unsigned char *)malloc(hash_size * batch_size);
    assert(cpu_msg_hash != NULL);

    unsigned char *cpu_seckey = (unsigned char *)malloc(seckey_size * batch_size);
    assert(cpu_msg_hash != NULL);

    unsigned char randomize[32];

    unsigned char *cpu_compressed_pubkey = (unsigned char *)malloc(compressed_pubkey_size * batch_size);
    assert(cpu_compressed_pubkey != NULL);

    unsigned char *cpu_serialized_signature = (unsigned char *)malloc(serialized_signature_size * batch_size);
    assert(cpu_serialized_signature != NULL);

    unsigned char *gpu_serialized_signature = (unsigned char *)malloc(serialized_signature_size * batch_size);
    assert(gpu_serialized_signature != NULL);

    secp256k1_pubkey *pubkey = (secp256k1_pubkey *)malloc(sizeof(secp256k1_pubkey) * batch_size);
    assert(pubkey != NULL);

    secp256k1_ecdsa_signature *cpu_sigs = (secp256k1_ecdsa_signature *)malloc(batch_size * sizeof(secp256k1_ecdsa_signature));
    assert(cpu_sigs != NULL);

    secp256k1_ecdsa_signature *gpu_sigs = (secp256k1_ecdsa_signature *)malloc(batch_size * sizeof(secp256k1_ecdsa_signature));
    assert(gpu_sigs != NULL);

    size_t len;
    int is_cpu_signature_valid, is_gpu_signature_valid, is_signature_valid2;
    int return_val;

    /* Before we can call actual API functions, we need to create a "context". */
    secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!fill_random(randomize, sizeof(randomize)))
    {
        printf("Failed to generate randomness\n");
        return 1;
    }
    /* Randomizing the context is recommended to protect against side-channel
     * leakage See `secp256k1_context_randomize` in secp256k1.h for more
     * information about it. This should never fail. */
    return_val = secp256k1_context_randomize(ctx, randomize);
    assert(return_val);

    /*** Key Generation ***/

    /* If the secret key is zero or out of range (bigger than secp256k1's
     * order), we try to sample a new key. Note that the probability of this
     * happening is negligible. */
    for (uint32_t i = 0; i < batch_size; i++)
    {
        while (1)
        {
            if (!fill_random(cpu_seckey + i * seckey_size, seckey_size))
            {
                printf("Failed to generate randomness!\n");
                return 1;
            }
            if (secp256k1_ec_seckey_verify(ctx, cpu_seckey + i * seckey_size))
            {
                break;
            }
        }

        /* Public key creation using a valid context with a verified secret key should never fail */
        return_val = secp256k1_ec_pubkey_create(ctx, pubkey + i, cpu_seckey + i * seckey_size);
        assert(return_val);

        /* Serialize the pubkey in a compressed form(33 bytes). Should always return 1. */
        len = compressed_pubkey_size;
        return_val = secp256k1_ec_pubkey_serialize(ctx, cpu_compressed_pubkey + i * compressed_pubkey_size, &len, pubkey + i, SECP256K1_EC_COMPRESSED);
        assert(return_val);
        /* Should be the same size as the size of the output, because we passed a 33 byte array. */
        assert(len == compressed_pubkey_size);
    }

    /*** Signing ***/

    /* Generate an ECDSA signature `noncefp` and `ndata` allows you to pass a
     * custom nonce function, passing `NULL` will use the RFC-6979 safe default.
     * Signing with a valid context, verified secret key
     * and the default nonce function should never fail. */

    // GPU
    const int tpb = 256;
    unsigned char *gpu_msg;
    unsigned char *gpu_seckey;
    secp256k1_ecdsa_signature *gpu_sig;
    secp256k1_context *gpu_ctx;
    int *gpu_rets;
    int *cpu_rets = (int *)malloc(batch_size * sizeof(int));
    assert(cpu_rets != NULL);
    struct timeval start, end;

    CHECKCUDAERR(cudaMalloc(&gpu_ctx, sizeof(secp256k1_context)));
    CHECKCUDAERR(cudaMalloc(&gpu_msg, batch_size * hash_size));
    CHECKCUDAERR(cudaMalloc(&gpu_seckey, batch_size * seckey_size));
    CHECKCUDAERR(cudaMalloc(&gpu_sig, batch_size * sizeof(secp256k1_ecdsa_signature)));
    CHECKCUDAERR(cudaMalloc(&gpu_rets, batch_size * sizeof(int)));

    // start timing (include data transfers)
    printf("Starting GPU signing...\n");
    gettimeofday(&start, NULL);
    CHECKCUDAERR(cudaMemcpy(gpu_msg, cpu_msg_hash, batch_size * hash_size, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(gpu_seckey, cpu_seckey, batch_size * seckey_size, cudaMemcpyHostToDevice));
    CHECKCUDAERR(cudaMemcpy(gpu_ctx, ctx, sizeof(secp256k1_context), cudaMemcpyHostToDevice));

    secp256k1_ecdsa_sign_batch<<<batch_size / tpb, tpb>>>(batch_size, gpu_ctx, gpu_sig, gpu_msg, gpu_seckey, gpu_rets);
    CHECKCUDAERR(cudaMemcpy(cpu_rets, gpu_rets, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(gpu_sigs, gpu_sig, batch_size * sizeof(secp256k1_ecdsa_signature), cudaMemcpyDeviceToHost));
    gettimeofday(&end, NULL);
    printf("GPU time: %ld us\n", (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec);

    CHECKCUDAERR(cudaFree(gpu_msg));
    CHECKCUDAERR(cudaFree(gpu_seckey));
    CHECKCUDAERR(cudaFree(gpu_sig));
    CHECKCUDAERR(cudaFree(gpu_ctx));
    CHECKCUDAERR(cudaFree(gpu_rets));

    // CPU
    printf("Starting CPU signing...\n");
    gettimeofday(&start, NULL);

    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++)
    {
        return_val = secp256k1_ecdsa_sign(ctx, &cpu_sigs[i], cpu_msg_hash + i * hash_size, cpu_seckey + i * seckey_size, NULL, NULL);
        assert(return_val);
    }
    gettimeofday(&end, NULL);
    printf("CPU time: %ld us\n", (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec);

    /*** Verification ***/
    for (int i = 0; i < batch_size; i++)
    {
        /* Serialize the signature in a compact form. Should always return 1
         * according to the documentation in secp256k1.h. */
        return_val = secp256k1_ecdsa_signature_serialize_compact(ctx, cpu_serialized_signature + i * compressed_pubkey_size, cpu_sigs + i);
        assert(return_val);
        return_val = secp256k1_ecdsa_signature_serialize_compact(ctx, gpu_serialized_signature + i * compressed_pubkey_size, gpu_sigs + i);
        assert(return_val);

        /* Deserialize the signature. This will return 0 if the signature can't be parsed correctly. */
        if (!secp256k1_ecdsa_signature_parse_compact(ctx, cpu_sigs + i, cpu_serialized_signature + i * compressed_pubkey_size))
        {
            printf("Failed parsing the CPU signature\n");
            return 1;
        }
        if (!secp256k1_ecdsa_signature_parse_compact(ctx, gpu_sigs + i, gpu_serialized_signature + i * compressed_pubkey_size))
        {
            printf("Failed parsing the GPU signature\n");
            return 1;
        }

        /* Deserialize the public key. This will return 0 if the public key can't be parsed correctly. */
        if (!secp256k1_ec_pubkey_parse(ctx, pubkey + i, cpu_compressed_pubkey + i * compressed_pubkey_size, compressed_pubkey_size))
        {
            printf("Failed parsing the public key\n");
            return 1;
        }

        is_cpu_signature_valid = secp256k1_ecdsa_verify(ctx, cpu_sigs + i, cpu_msg_hash + i * hash_size, pubkey + i);
        assert(is_cpu_signature_valid);
        is_gpu_signature_valid = secp256k1_ecdsa_verify(ctx, gpu_sigs + i, cpu_msg_hash, pubkey + i);
        assert(is_gpu_signature_valid);
    }

    /* Verify a signature. This will return 1 if it's valid and 0 if it's not. */
    is_cpu_signature_valid = secp256k1_ecdsa_verify(ctx, cpu_sigs, cpu_msg_hash, pubkey);
    is_gpu_signature_valid = secp256k1_ecdsa_verify(ctx, gpu_sigs, cpu_msg_hash, pubkey);

    printf("Is the CPU signature valid? %s\n", is_cpu_signature_valid ? "true" : "false");
    printf("Is the GPU signature valid? %s\n", is_gpu_signature_valid ? "true" : "false");
    printf("Secret Key: ");
    print_hex(cpu_seckey, seckey_size);
    printf("Public Key: ");
    print_hex(cpu_compressed_pubkey, compressed_pubkey_size);
    printf("Signature CPU: ");
    print_hex(cpu_serialized_signature, serialized_signature_size);
    printf("Signature GPU: ");
    print_hex(gpu_serialized_signature, serialized_signature_size);

    /* This will clear everything from the context and free the memory */
    secp256k1_context_destroy(ctx);

    /* Bonus example: if all we need is signature verification (and no key
       generation or signing), we don't need to use a context created via
       secp256k1_context_create(). We can simply use the static (i.e., global)
       context secp256k1_context_static. See its description in
       include/secp256k1.h for details. */
    is_signature_valid2 = secp256k1_ecdsa_verify(secp256k1_context_static,
                                                 cpu_sigs, cpu_msg_hash, pubkey);
    assert(is_signature_valid2 == is_cpu_signature_valid);

    /* It's best practice to try to clear secrets from memory after using them.
     * This is done because some bugs can allow an attacker to leak memory, for
     * example through "out of bounds" array access (see Heartbleed), or the OS
     * swapping them to disk. Hence, we overwrite the secret key buffer with zeros.
     *
     * Here we are preventing these writes from being optimized out, as any good compiler
     * will remove any writes that aren't used. */
    secure_erase(cpu_seckey, seckey_size);

    // Free memory
    free(cpu_msg_hash);
    free(cpu_seckey);
    free(cpu_compressed_pubkey);
    free(cpu_serialized_signature);
    free(gpu_serialized_signature);
    free(pubkey);
    free(cpu_sigs);
    free(gpu_sigs);
    free(cpu_rets);

    printf("Done.\n");
    return 0;
}
