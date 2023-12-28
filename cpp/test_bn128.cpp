#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <gtest/gtest.h>
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
uint64_t lehmer64()
{
    g_lehmer64_state *= 0xda942042e4dd58b5LL;
    return g_lehmer64_state >> 64;
}

uint8_t random_byte()
{
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define the distribution for uint8_t values
    std::uniform_int_distribution<uint8_t> distribution(0, std::numeric_limits<uint8_t>::max());

    // Generate a random uint8_t value
    uint8_t randomValue = distribution(gen);

    // Print the generated value
    // std::cout << "Random uint8_t value: " << static_cast<unsigned>(randomValue) << std::endl;
    return randomValue;
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

TEST(altBn128, fft_cpu_self_consistency)
{
    int lg_n_size = 10;
    int N = 1 << lg_n_size;

    uint8_t *raw_data = new uint8_t[N * 32];

    for (int i = 0; i < N * 32; i++)
    {
        uint8_t random_int = random_byte();
        *(raw_data + i) = i % 32 == 31 ? 0 : random_int; // TODO: this is to make the input less than MOD; otherwise, the test will fail
    }
    // print_char_array(raw_data, N*32);

    AltBn128::FrElement *cpu_data_in = new AltBn128::FrElement[N];
    for (int i = 0; i < N; i++)
    {
        int result = Fr.fromRprLE(cpu_data_in[i], (const uint8_t *)(raw_data + i * 32), 32);
    }

    FFT<typename Engine::Fr> fft(N);
    double start, end;
    double cpu_time_used;
    start = omp_get_wtime();
    fft.fft(cpu_data_in, N);
    end = omp_get_wtime();
    cpu_time_used = ((double)(end - start));
    // printf("\n Time used fft (us): %.3lf\n", cpu_time_used * 1000000); // lf stands for long float
    fft.ifft(cpu_data_in, N);

    AltBn128::FrElement aux;
    for (int i = 0; i < N; i++)
    {
        Fr.fromRprLE(aux, (const uint8_t *)(raw_data + i * 32), 32);
        ASSERT_TRUE(Fr.eq(cpu_data_in[i], aux));
    }
    delete[] raw_data;
    delete[] cpu_data_in;
}

TEST(altBn128, fft_gpu_self_consistency)
{
    int lg_n_size = 10;
    int N = 1 << lg_n_size;

    uint8_t *raw_data = new uint8_t[N * 32];

    for (int i = 0; i < N * 32; i++)
    {
        uint8_t random_int = random_byte();
        *(raw_data + i) = i % 32 == 31 ? 0 : random_int; // TODO: this is to make the input less than MOD; otherwise, the test will fail
    }
    // printf("raw data \n");
    // print_char_array(raw_data, N*32);

    fr_t *gpu_data_in = (fr_t *)malloc(N * sizeof(fr_t));
    for (int i = 0; i < N; i++)
    {
        gpu_data_in[i] = *(fr_t *)(raw_data + 32 * i);
    }
    // printf("gpu_data_in \n");
    // print_char_array((uint8_t*)gpu_data_in, N*32);

    size_t device_id = 0;
    compute_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);
    compute_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::inverse, Ntt_Types::Type::standard);
    // printf("gpu_data_in \n");
    // print_char_array((uint8_t*)gpu_data_in, N*32);
    // printf("gpu result \n");
    for (int i = 0; i < N; i++)
    {
        uint8_t raw_result[32], gpu_result[32];
        // Fr.toRprLE(cpu_data_in[i], cpu_result, 32);
        memcpy(gpu_result, (uint8_t *)(gpu_data_in + i), 32);
        memcpy(raw_result, (uint8_t *)(raw_data + i*32), 32);
        // print_char_array(gpu_result, 32);
        // print_char_array(raw_result, 32);
        for (int i = 0; i < 32; i++)
        {
            ASSERT_EQ(raw_result[i], (gpu_result[i]));
        }
    }
    delete[] raw_data;
    delete[] gpu_data_in;
}

TEST(altBn128, fft_cpu_consistent_with_gpu)
{
    int lg_n_size = 10;
    int N = 1 << lg_n_size;

    uint8_t *raw_data = new uint8_t[N * 32];

    for (int i = 0; i < N * 32; i++)
    {
        uint8_t random_int = random_byte();
        *(raw_data + i) = i % 32 == 31 ? 0 : random_int; // TODO: this is to make the input less than MOD; otherwise, the test will fail
    }
    // print_char_array(raw_data, N*32);

    AltBn128::FrElement *cpu_data_in = new AltBn128::FrElement[N];
    for (int i = 0; i < N; i++)
    {
        int result = Fr.fromRprLE(cpu_data_in[i], (const uint8_t *)(raw_data + i * 32), 32);
    }

    FFT<typename Engine::Fr> fft(N);
    double start, end;
    double cpu_time_used;
    start = omp_get_wtime();
    fft.fft(cpu_data_in, N);
    end = omp_get_wtime();
    cpu_time_used = ((double)(end - start));
    // printf("\n Time used fft (us): %.3lf\n", cpu_time_used * 1000000); // lf stands for long float

    fr_t *gpu_data_in = (fr_t *)malloc(N * sizeof(fr_t));
    for (int i = 0; i < N; i++)
    {
        gpu_data_in[i] = *(fr_t *)(raw_data + 32 * i);
    }
    delete[] raw_data;

    size_t device_id = 0;
    compute_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);

    for (int i = 0; i < N; i++)
    {
        uint8_t cpu_result[32], gpu_result[32];
        Fr.toRprLE(cpu_data_in[i], cpu_result, 32);
        memcpy(gpu_result, (uint8_t *)(gpu_data_in + i), 32);

        for (int i = 0; i < 32; i++)
        {

            ASSERT_EQ(cpu_result[i], (gpu_result[i]));
        }
    }
    delete[] cpu_data_in;
    delete[] gpu_data_in;
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}