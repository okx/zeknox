#include <gtest/gtest.h>
#if defined(FEATURE_GOLDILOCKS)
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <blst_t.hpp>
#include <vect.h>
#include <ntt/ntt.h>
#include <lib.h>
#include <time.h>
// #include "fft.hpp"
#include <random>
#include <cmath>
#include <chrono>
#include <thread>

#define assertm(exp, msg) assert(((void)msg, exp))

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
    printf("\n");
}

void print_u64_array(uint64_t *p, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {

        if (i % 4 == 0)
        {
            printf("\n");
        }
        printf("%lx,", p[i]);
    }
    printf("\n");
}

TEST(gl64, fft_gpu_self_consistency)
{
    int lg_n_size = 19;
    int N = 1 << lg_n_size;

    uint64_t *raw_data = new uint64_t[N];

    for (int i = 0; i < N; i++)
    {
        *(raw_data + i) = lehmer64();
    }

    fr_t *gpu_data_in = (fr_t *)malloc(N * sizeof(fr_t));
    for (int i = 0; i < N; i++)
    {
        gpu_data_in[i] = *(fr_t *)(raw_data + i);
    }
    // printf("gpu_data_in \n");
    // print_char_array((uint8_t*)gpu_data_in, N*32);

    size_t device_id = 0;
    printf("data before \n");
    print_char_array((uint8_t *)gpu_data_in, 16);
    // compute_batched_ntt(device_id, gpu_data_in, lg_n_size, 1, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);

    init_twiddle_factors(0, lg_n_size);

    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    auto start_time = std::chrono::high_resolution_clock::now();

    init_twiddle_factors(0, lg_n_size);

    Ntt_Types::NTTConfig cfg{
        batches: 1,
        order: Ntt_Types::InputOutputOrder::NN,
        ntt_type: Ntt_Types::Type::standard,
    };
    compute_batched_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::Direction::forward, cfg);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    long long microseconds = duration.count();

    std::cout << "Time elapsed in batch ntt: " << microseconds << " microseconds" << std::endl;
    printf("data end \n");
    print_char_array((uint8_t *)gpu_data_in, 16);
    // compute_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::inverse, Ntt_Types::Type::standard);
    // // printf("gpu_data_in \n");

    // // printf("gpu result \n");
    // for (int i = 0; i < N; i++)
    // {
    //     uint8_t raw_result[32], gpu_result[32];
    //     // Fr.toRprLE(cpu_data_in[i], cpu_result, 32);
    //     memcpy(gpu_result, (uint8_t *)(gpu_data_in + i), 32);
    //     memcpy(raw_result, (uint8_t *)(raw_data + i * 32), 32);
    //     // print_char_array(gpu_result, 32);
    //     // print_char_array(raw_result, 32);
    //     for (int i = 0; i < 32; i++)
    //     {
    //         ASSERT_EQ(raw_result[i], (gpu_result[i]));
    //     }
    // }
    // delete[] raw_data;
    // delete[] gpu_data_in;
}

#endif
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
