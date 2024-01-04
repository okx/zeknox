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
        memcpy(raw_result, (uint8_t *)(raw_data + i * 32), 32);
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

TEST(altBn128, msm)
{

    int N = 1;

    G1PointAffine point_one = G1PointAffine{};
    G1.copy(point_one, G1.one());

    G1PointAffine point_two = G1PointAffine{};
    G1.add(point_two, point_one, G1.one());
    // uint64_t[4] x = point_one.x.v;
    printf("point one in raw bytes \n");
    print_u64_array(point_one.x.v, 4);
    print_u64_array(point_one.y.v, 4);
    std::string x_str = F1.toString(point_one.x, 16);
    std::string y_str = F1.toString(point_one.y, 16);
    printf("point one x in hex: %s \n", x_str.c_str());
    printf("point one y in hex: %s \n", y_str.c_str());

    // G1PointAffine cpu_point;
    // F1.fromString(cpu_point.x, "1", 16);
    // F1.fromString(cpu_point.y, "2", 16);
    // print_u64_array(cpu_point.x.v, 4);
    // std::string x1_str = F1.toString(cpu_point.x, 16);
    // printf("point1 x in hex: %s \n", x1_str.c_str());
    // print_u64_array(cpu_point.y.v, 4);

    // G1PointAffine cpu_point2;
    // int result = F1.fromRprLE(cpu_point2.x, (const uint8_t *)point_one.x.v, 32);
    // printf("result: %d \n", result);
    // F1.fromString(cpu_point2.y, "2", 32);
    // print_u64_array(cpu_point2.x.v, 4);
    // print_u64_array(cpu_point2.y.v, 4);

    // std::string x2_str = F1.toString(cpu_point2.x, 16);
    // printf("point2 x in hex: %s \n", x2_str.c_str());
    // {
    //     x : Fq.fromString()
    //     y : point_one.x.y,
    // };

    
    // typedef uint8_t Scalar[32];

    // Scalar *scalars = new Scalar[N];
  

    // uint8_t *raw_scalars = new uint8_t[N * 32];
    // uint8_t *raw_points_x = new uint8_t[N * 32];
    // uint8_t *raw_points_y = new uint8_t[N * 32];

    // for (int i = 0; i < N * 32; i++)
    // {
    //     uint8_t random_scalar = random_byte();
    //     *(raw_scalars + i) = i % 32 == 31 ? 0 : random_scalar; // TODO: this is to make the input less than MOD; otherwise, the test will fail

    //     uint8_t random_x = random_byte();
    //     *(raw_points_x + i) = i % 32 == 31 ? 0 : random_x; // TODO: this is to make the input less than MOD; otherwise, the test will fail

    //     uint8_t random_y = random_byte();
    //     *(raw_points_y + i) = i % 32 == 31 ? 0 : random_y; // TODO: this is to make the input less than MOD; otherwise, the test will fail
    // }

    // for (int i = 0; i < N; i++)
    // {
    //     fp_t x = *(fp_t *)cpu_points[i].x.v;
    //     fp_t y = *(fp_t *)cpu_points[i].y.v;
    //     // affine_t p_affine = affine_t(x, y);
    //     // G1.
    //     uint64_t a[4]{1, 2, 3, 4};
    //     G1PointAffine cpu_point = G1PointAffine{
    //         x : *a,
    //         y : *a,
    //     };
    //     // points[i] = p_affine;
    // }
    uint8_t scalars[32]{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    G1PointAffine *cpu_points = new G1PointAffine[N];
    cpu_points[0] = point_one;
    uint32_t point_sz = sizeof(cpu_points[0]);
    // printf("point_sz: %d \n", point_sz);
    // printf("point_0_x: %x \n", cpu_points[0].x);
    G1Point p1;
    G1.multiMulByScalar(p1, cpu_points, (uint8_t *)scalars, 32, N);
    std::string p1_str = G1.toString(p1, 16);
    printf("cpu_p1_str: %s \n", p1_str.c_str());

    
    // // RustError::by_value mult_pippenger_inf(point_t* out, const affine_t points[],
    // //                                    size_t npoints, const scalar_t scalars[],
    // //                                    size_t ffi_affine_sz);

    // affine_t points[NMExp];

    // // printf("raw data \n");
    // // print_char_array(raw_data, N*32);

    // fr_t *gpu_acalar = (fr_t *)malloc(NMExp * sizeof(fr_t));
    // for (int i = 0; i < NMExp; i++)
    // {
    //     gpu_acalar[i] = *(fr_t *)(gpu_raw_data + 32 * i);
    // }
    // printf("gpu point_0 x: %x \n", points[0].X);
    // // print_char_array((uint8_t*)points[0].X, 32);
    fp_t x = *(fp_t *)point_one.x.v;
    fp_t y = *(fp_t *)point_one.y.v;
    affine_t p_affine = affine_t(x, y);
    point_t *out = new point_t{};
    affine_t point_inputs[1]{p_affine};
    mult_pippenger_inf(out, point_inputs, 1, (fr_t *)scalars, 72);
    // G1Point result_point;
    // G1PointAffine result_affine;
    // G1.copy(result_affine, result_point);
    // std::string out_str = G1.toString(out, 16);
    // printf("out_str: %s \n", out_str.c_str());
    printf("gpu result X\n");
    print_u64_array((uint64_t *)(&out->X), 4);
    print_char_array((uint8_t *)(uint64_t *)(&out->X), 32);

    F1Element gpu_x;
    F1.fromRprLE(gpu_x, (uint8_t *)(uint64_t *)(&out->X), 32);
    std::string gpu_x_str = F1.toString(gpu_x, 10);
    printf("\n gpu_x_str: %s\n", gpu_x_str.c_str());

     F1Element gpu_y;
    F1.fromRprLE(gpu_y, (uint8_t *)(uint64_t *)(&out->Y), 32);
    std::string gpu_y_str = F1.toString(gpu_y, 10);
    printf("\n gpu_y_str: %s\n", gpu_y_str.c_str());

     F1Element gpu_z;
    F1.fromRprLE(gpu_z, (uint8_t *)(uint64_t *)(&out->Z), 32);
    std::string gpu_z_str = F1.toString(gpu_z, 10);
    printf("\n gpu_z_str: %s\n", gpu_z_str.c_str());


    // printf("gpu result Y\n");
    // print_u64_array((uint64_t *)(&out->Y), 4);
    // printf("gpu result Z\n");
    // print_u64_array((uint64_t *)(&out->Z), 4);
    //     printf("gpu result z %x \n", out->Z);

    // mpz_t e;
    // mpz_init_set_ui(e, acc);

    // Scalar sAcc;

    // for (int i = 0; i < 32; i++)
    //     sAcc[i] = 0;
    // mpz_export((void *)sAcc, NULL, -1, 8, -1, 0, e);
    // mpz_clear(e);

    // G1Point p2;
    // G1.mulByScalar(p2, G1.one(), sAcc, 32);

    // ASSERT_TRUE(G1.eq(p1, p2));

    // delete[] cpu_points;
    // delete[] scalars;
}
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}