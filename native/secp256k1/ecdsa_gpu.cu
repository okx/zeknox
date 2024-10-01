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

static void secure_erase(void *ptr, size_t len) {
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
static int fill_random(unsigned char* data, size_t size) {
#if defined(_WIN32)
    NTSTATUS res = BCryptGenRandom(NULL, data, size, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    if (res != STATUS_SUCCESS || size > ULONG_MAX) {
        return 0;
    } else {
        return 1;
    }
#elif defined(__linux__) || defined(__FreeBSD__)
    /* If `getrandom(2)` is not available you should fallback to /dev/urandom */
    ssize_t res = getrandom(data, size, 0);
    if (res < 0 || (size_t)res != size ) {
        return 0;
    } else {
        return 1;
    }
#elif defined(__APPLE__) || defined(__OpenBSD__)
    /* If `getentropy(2)` is not available you should fallback to either
     * `SecRandomCopyBytes` or /dev/urandom */
    int res = getentropy(data, size);
    if (res == 0) {
        return 1;
    } else {
        return 0;
    }
#endif
    return 0;
}

static void print_hex(unsigned char* data, size_t size) {
    size_t i;
    printf("0x");
    for (i = 0; i < size; i++) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

__global__ void secp256k1_ecdsa_sign_batch(int batch_size, const secp256k1_context* ctxs, secp256k1_ecdsa_signature *signatures, const unsigned char *msghash32s, const unsigned char *seckeys, int *rets) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    rets[tid] = secp256k1_ecdsa_sign(ctxs, signatures + tid, msghash32s + tid * 32, seckeys + tid * 32, NULL, NULL);
}

int main(void) {
    /* Instead of signing the message directly, we must sign a 32-byte hash.
     * Here the message is "Hello, world!" and the hash function was SHA-256.
     * An actual implementation should just call SHA-256, but this example
     * hardcodes the output to avoid depending on an additional library.
     * See https://bitcoin.stackexchange.com/questions/81115/if-someone-wanted-to-pretend-to-be-satoshi-by-posting-a-fake-signature-to-defrau/81116#81116 */
    unsigned char msg_hash[32] = {
        0x31, 0x5F, 0x5B, 0xDB, 0x76, 0xD0, 0x78, 0xC4,
        0x3B, 0x8A, 0xC0, 0x06, 0x4E, 0x4A, 0x01, 0x64,
        0x61, 0x2B, 0x1F, 0xCE, 0x77, 0xC8, 0x69, 0x34,
        0x5B, 0xFC, 0x94, 0xC7, 0x58, 0x94, 0xED, 0xD3,
    };
    unsigned char seckey[32];
    unsigned char randomize[32];
    unsigned char compressed_pubkey[33];
    unsigned char serialized_signature_cpu[64], serialized_signature_gpu[64];
    size_t len;
    int is_cpu_signature_valid, is_gpu_signature_valid, is_signature_valid2;
    int return_val;
    secp256k1_pubkey pubkey;
    secp256k1_ecdsa_signature sig_cpu, sig_gpu;
    /* Before we can call actual API functions, we need to create a "context". */
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!fill_random(randomize, sizeof(randomize))) {
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
    while (1) {
        if (!fill_random(seckey, sizeof(seckey))) {
            printf("Failed to generate randomness\n");
            return 1;
        }
        if (secp256k1_ec_seckey_verify(ctx, seckey)) {
            break;
        }
    }

    /* Public key creation using a valid context with a verified secret key should never fail */
    return_val = secp256k1_ec_pubkey_create(ctx, &pubkey, seckey);
    assert(return_val);

    /* Serialize the pubkey in a compressed form(33 bytes). Should always return 1. */
    len = sizeof(compressed_pubkey);
    return_val = secp256k1_ec_pubkey_serialize(ctx, compressed_pubkey, &len, &pubkey, SECP256K1_EC_COMPRESSED);
    assert(return_val);
    /* Should be the same size as the size of the output, because we passed a 33 byte array. */
    assert(len == sizeof(compressed_pubkey));

    /*** Signing ***/

    /* Generate an ECDSA signature `noncefp` and `ndata` allows you to pass a
     * custom nonce function, passing `NULL` will use the RFC-6979 safe default.
     * Signing with a valid context, verified secret key
     * and the default nonce function should never fail. */

    // GPU
    const int nth = (1 << 13);
    const int tpb = 256;
    unsigned char *gpu_msg;
    unsigned char *gpu_seckey;
    secp256k1_ecdsa_signature *gpu_sig;
    secp256k1_context *gpu_ctx;
    int *gpu_ret;
    int rets[nth];
    struct timeval start, end;

    gettimeofday(&start, NULL);
    CHECKCUDAERR(cudaMalloc(&gpu_msg, nth * 32));
    CHECKCUDAERR(cudaMalloc(&gpu_seckey, nth * 32));
    CHECKCUDAERR(cudaMalloc(&gpu_sig, nth * sizeof(secp256k1_ecdsa_signature)));
    CHECKCUDAERR(cudaMalloc(&gpu_ctx, sizeof(secp256k1_context)));
    CHECKCUDAERR(cudaMalloc(&gpu_ret, nth * sizeof(int)));

    for (int i = 0; i < nth; i++) {
        CHECKCUDAERR(cudaMemcpy(gpu_msg + i * 32, msg_hash, 32, cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpu_seckey + i * 32, seckey, 32, cudaMemcpyHostToDevice));
    }
    CHECKCUDAERR(cudaMemcpy(gpu_ctx, ctx, sizeof(secp256k1_context), cudaMemcpyHostToDevice));

    secp256k1_ecdsa_sign_batch<<<nth/tpb, tpb>>>(nth, gpu_ctx, gpu_sig, gpu_msg, gpu_seckey, gpu_ret);
    CHECKCUDAERR(cudaMemcpy(rets, gpu_ret, nth * sizeof(int), cudaMemcpyDeviceToHost));
    CHECKCUDAERR(cudaMemcpy(&sig_gpu, gpu_sig, sizeof(secp256k1_ecdsa_signature), cudaMemcpyDeviceToHost));
    gettimeofday(&end, NULL);
    printf("GPU time: %ld us\n", (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec);

    CHECKCUDAERR(cudaFree(gpu_msg));
    CHECKCUDAERR(cudaFree(gpu_seckey));
    CHECKCUDAERR(cudaFree(gpu_sig));
    CHECKCUDAERR(cudaFree(gpu_ctx));
    CHECKCUDAERR(cudaFree(gpu_ret));

    // CPU
    gettimeofday(&start, NULL);

    // #pragma omp parallel
    for (int i = 0; i < nth; i++) {
        return_val = secp256k1_ecdsa_sign(ctx, &sig_cpu, msg_hash, seckey, NULL, NULL);
        assert(return_val);
    }
    gettimeofday(&end, NULL);
    printf("CPU time: %ld us\n", (end.tv_sec - start.tv_sec) * 1000000 + end.tv_usec - start.tv_usec);


    /* Serialize the signature in a compact form. Should always return 1
     * according to the documentation in secp256k1.h. */
    return_val = secp256k1_ecdsa_signature_serialize_compact(ctx, serialized_signature_cpu, &sig_cpu);
    assert(return_val);
    return_val = secp256k1_ecdsa_signature_serialize_compact(ctx, serialized_signature_gpu, &sig_gpu);
    assert(return_val);


    /*** Verification ***/

    /* Deserialize the signature. This will return 0 if the signature can't be parsed correctly. */
    if (!secp256k1_ecdsa_signature_parse_compact(ctx, &sig_cpu, serialized_signature_cpu)) {
        printf("Failed parsing the CPU signature\n");
        return 1;
    }
    if (!secp256k1_ecdsa_signature_parse_compact(ctx, &sig_gpu, serialized_signature_gpu)) {
        printf("Failed parsing the GPU signature\n");
        return 1;
    }

    /* Deserialize the public key. This will return 0 if the public key can't be parsed correctly. */
    if (!secp256k1_ec_pubkey_parse(ctx, &pubkey, compressed_pubkey, sizeof(compressed_pubkey))) {
        printf("Failed parsing the public key\n");
        return 1;
    }

    /* Verify a signature. This will return 1 if it's valid and 0 if it's not. */
    is_cpu_signature_valid = secp256k1_ecdsa_verify(ctx, &sig_cpu, msg_hash, &pubkey);
    is_gpu_signature_valid = secp256k1_ecdsa_verify(ctx, &sig_gpu, msg_hash, &pubkey);

    printf("Is the CPU signature valid? %s\n", is_cpu_signature_valid ? "true" : "false");
    printf("Is the GPU signature valid? %s\n", is_gpu_signature_valid ? "true" : "false");
    printf("Secret Key: ");
    print_hex(seckey, sizeof(seckey));
    printf("Public Key: ");
    print_hex(compressed_pubkey, sizeof(compressed_pubkey));
    printf("Signature CPU: ");
    print_hex(serialized_signature_cpu, sizeof(serialized_signature_cpu));
    printf("Signature GPU: ");
    print_hex(serialized_signature_gpu, sizeof(serialized_signature_gpu));

    /* This will clear everything from the context and free the memory */
    secp256k1_context_destroy(ctx);

    /* Bonus example: if all we need is signature verification (and no key
       generation or signing), we don't need to use a context created via
       secp256k1_context_create(). We can simply use the static (i.e., global)
       context secp256k1_context_static. See its description in
       include/secp256k1.h for details. */
    is_signature_valid2 = secp256k1_ecdsa_verify(secp256k1_context_static,
                                                 &sig_cpu, msg_hash, &pubkey);
    assert(is_signature_valid2 == is_cpu_signature_valid);

    /* It's best practice to try to clear secrets from memory after using them.
     * This is done because some bugs can allow an attacker to leak memory, for
     * example through "out of bounds" array access (see Heartbleed), or the OS
     * swapping them to disk. Hence, we overwrite the secret key buffer with zeros.
     *
     * Here we are preventing these writes from being optimized out, as any good compiler
     * will remove any writes that aren't used. */
    secure_erase(seckey, sizeof(seckey));

    return 0;
}
