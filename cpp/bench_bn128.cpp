#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <stdexcept>
#include <blst_t.hpp>
#include <vect.h>
#include <ntt/ntt.h>
#include "../src/lib.h"
#include <gmp.h>
#include "alt_bn128.hpp"
#include <time.h>
#include "fft.hpp"
#include <random>

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace AltBn128;

__uint128_t g_lehmer64_state = 0xAAAAAAAAAAAAAAAALL;

// Fast random generator
// https://lemire.me/blog/2019/03/19/the-fastest-conventional-random-number-generator-that-can-pass-big-crush/
uint8_t lehmer64()
{
    g_lehmer64_state *= 0xda942042e4dd58b5LL;
    return g_lehmer64_state >> 120;
}

void print_char_array(uint8_t *p, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {

        if (i % 32 == 0)
        {
            printf("\n");
        }
        printf("%02x", p[i]);
    }
}



int main(int argc, char **argv)
{
    int lg_n_size = atoi(argv[1]);
    printf("log n size: %d \n", lg_n_size);
    int N = 1 << lg_n_size;

    uint8_t *raw_data = new uint8_t[N * 32];

    for (int i = 0; i < N * 32; i++)
    {
        uint8_t random_int = lehmer64();
        // printf("random byte: %d \n", random_int);
        *(raw_data + i) = i % 32 == 31 ? 0 : random_int; // TODO: this is to make the input less than MOD; otherwise, the test will fail
    }
    // print_char_array(raw_data, N*32);

    AltBn128::FrElement *cpu_data_in = new AltBn128::FrElement[N];
    for (int i = 0; i < N; i++)
    {
        int result = Fr.fromRprLE(cpu_data_in[i], (const uint8_t *)(raw_data + i * 32), 32);
    }

    FFT<typename Engine::Fr> fft(N);
    printf("start cpu fft\n");
    double start, end;
    double cpu_time_used;
    start = omp_get_wtime();
    fft.fft(cpu_data_in, N);
    end = omp_get_wtime();
    cpu_time_used = ((double)(end - start));
    printf("Time used cpu fft (ms): %.3lf\n", cpu_time_used * 1000); // lf stands for long float
    delete[] cpu_data_in;

    fr_t *gpu_data_in = (fr_t *)malloc(N * sizeof(fr_t));
    for (int i = 0; i < N; i++)
    {
        gpu_data_in[i] = *(fr_t *)(raw_data + 32 * i);
    }
    delete[] raw_data;

    size_t device_id = 0;
    double start_gpu, end_gpu;
    double gpu_time_used;
    start_gpu = omp_get_wtime();
    compute_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);
    end_gpu = omp_get_wtime();
    gpu_time_used = ((double)(end_gpu - start_gpu));
    printf("Time used  gpu fft (ms): %.3lf\n", gpu_time_used * 1000); // lf stands for long float


    delete[] gpu_data_in;
}
