#include <gtest/gtest.h>
#if defined(FEATURE_BN254)
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <blst_t.hpp>
#include <vect.h>
#include <ntt/ntt.h>
#include <msm/msm.h>
#include <gmp.h>
#include <lib.h>
#include "alt_bn128.hpp"
#include <time.h>
#include "fft.hpp"
#include <random>
#include <cmath>
#include <cuda_runtime.h>

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

void print_f1_element(F1Element &e)
{
    std::string element = F1.toString(e, 10);
    printf("element in decimal: %s \n", element.c_str());
}

void print_g1_point(G1Point &p)
{
    std::string x = F1.toString(p.x, 10);
    std::string y = F1.toString(p.y, 10);
    std::string zz = F1.toString(p.zz, 10);
    std::string zzz = F1.toString(p.zzz, 10);
    printf("x: %s, y:%s, zz:%s, zzz:%s \n", x.c_str(), y.c_str(), zz.c_str(), zzz.c_str());

    std::string point_str = G1.toString(p, 10);
    printf("point_str: %s \n", point_str.c_str());
}

void print_g1_point_affine(G1PointAffine &p)
{
    std::string x = F1.toString(p.x, 16);
    std::string y = F1.toString(p.y, 16);
    printf("x: %s, y:%s, \n", x.c_str(), y.c_str());
}

#if defined(FEATURE_BN254)
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

// TODO: use ntt_batch
// TEST(altBn128, fft_gpu_self_consistency)
// {
//     int lg_n_size = 10;
//     int N = 1 << lg_n_size;

//     uint8_t *raw_data = new uint8_t[N * 32];

//     for (int i = 0; i < N * 32; i++)
//     {
//         uint8_t random_int = random_byte();
//         *(raw_data + i) = i % 32 == 31 ? 0 : random_int; // TODO: this is to make the input less than MOD; otherwise, the test will fail
//     }
//     // printf("raw data \n");
//     // print_char_array(raw_data, N*32);

//     fr_t *gpu_data_in = (fr_t *)malloc(N * sizeof(fr_t));
//     for (int i = 0; i < N; i++)
//     {
//         gpu_data_in[i] = *(fr_t *)(raw_data + 32 * i);
//     }
//     // printf("gpu_data_in \n");
//     // print_char_array((uint8_t*)gpu_data_in, N*32);

//     size_t device_id = 0;
//     auto e = compute_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);
//     auto err = compute_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::inverse, Ntt_Types::Type::standard);
//     // printf("gpu_data_in \n");
//     // print_char_array((uint8_t*)gpu_data_in, N*32);
//     // printf("gpu result \n");
//     for (int i = 0; i < N; i++)
//     {
//         uint8_t raw_result[32], gpu_result[32];
//         // Fr.toRprLE(cpu_data_in[i], cpu_result, 32);
//         memcpy(gpu_result, (uint8_t *)(gpu_data_in + i), 32);
//         memcpy(raw_result, (uint8_t *)(raw_data + i * 32), 32);
//         // print_char_array(gpu_result, 32);
//         // print_char_array(raw_result, 32);
//         for (int i = 0; i < 32; i++)
//         {
//             ASSERT_EQ(raw_result[i], (gpu_result[i]));
//         }
//     }
//     delete[] raw_data;
//     delete[] gpu_data_in;
// }

// TODO: use ntt_batch
// TEST(altBn128, fft_cpu_consistent_with_gpu)
// {
//     int lg_n_size = 10;
//     int N = 1 << lg_n_size;

//     uint8_t *raw_data = new uint8_t[N * 32];

//     for (int i = 0; i < N * 32; i++)
//     {
//         uint8_t random_int = random_byte();
//         *(raw_data + i) = i % 32 == 31 ? 0 : random_int; // TODO: this is to make the input less than MOD; otherwise, the test will fail
//     }
//     // print_char_array(raw_data, N*32);

//     AltBn128::FrElement *cpu_data_in = new AltBn128::FrElement[N];
//     for (int i = 0; i < N; i++)
//     {
//         int result = Fr.fromRprLE(cpu_data_in[i], (const uint8_t *)(raw_data + i * 32), 32);
//     }

//     FFT<typename Engine::Fr> fft(N);
//     double start, end;
//     double cpu_time_used;
//     start = omp_get_wtime();
//     fft.fft(cpu_data_in, N);
//     end = omp_get_wtime();
//     cpu_time_used = ((double)(end - start));
//     // printf("\n Time used fft (us): %.3lf\n", cpu_time_used * 1000000); // lf stands for long float

//     fr_t *gpu_data_in = (fr_t *)malloc(N * sizeof(fr_t));
//     for (int i = 0; i < N; i++)
//     {
//         gpu_data_in[i] = *(fr_t *)(raw_data + 32 * i);
//     }
//     delete[] raw_data;

//     size_t device_id = 0;
//     compute_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);

//     for (int i = 0; i < N; i++)
//     {
//         uint8_t cpu_result[32], gpu_result[32];
//         Fr.toRprLE(cpu_data_in[i], cpu_result, 32);
//         memcpy(gpu_result, (uint8_t *)(gpu_data_in + i), 32);

//         for (int i = 0; i < 32; i++)
//         {

//             ASSERT_EQ(cpu_result[i], (gpu_result[i]));
//         }
//     }
//     delete[] cpu_data_in;
//     delete[] gpu_data_in;
// }
#ifdef BUILD_MSM
TEST(altBn128, msm_bn254_g1_cpu_self_consistency)
{

    int NMExp = 40000;

    typedef uint8_t Scalar[32];

    Scalar *scalars = new Scalar[NMExp];
    G1PointAffine *bases = new G1PointAffine[NMExp];

    uint64_t acc = 0;
    for (int i = 0; i < NMExp; i++)
    {
        if (i == 0)
        {
            G1.copy(bases[0], G1.one());
        }
        else
        {
            G1.add(bases[i], bases[i - 1], G1.one());
        }
        for (int j = 0; j < 32; j++)
            scalars[i][j] = 0;
        *(int *)&scalars[i][0] = i + 1;
        acc += (i + 1) * (i + 1);
    }

    G1Point p1;
    G1.multiMulByScalar(p1, bases, (uint8_t *)scalars, 32, NMExp);

    mpz_t e;
    mpz_init_set_ui(e, acc);

    Scalar sAcc;

    for (int i = 0; i < 32; i++)
        sAcc[i] = 0;
    mpz_export((void *)sAcc, NULL, -1, 8, -1, 0, e);
    mpz_clear(e);

    G1Point p2;
    G1.mulByScalar(p2, G1.one(), sAcc, 32);

    ASSERT_TRUE(G1.eq(p1, p2));

    delete[] bases;
    delete[] scalars;
}

TEST(altBn128, msm_inputs_not_on_device_bn254_g1_curve_gpu_consistency_with_cpu)
{

    int NMExp = 1 << 12;

    typedef uint8_t Scalar[32];

    Scalar *scalars = new Scalar[NMExp];
    G1PointAffine *bases = new G1PointAffine[NMExp];

    uint64_t acc = 0;
    for (int i = 0; i < NMExp; i++)
    {
        if (i == 0)
        {
            G1.copy(bases[0], G1.one());
        }
        else
        {
            G1.add(bases[i], bases[i - 1], G1.one());
        }
        for (int j = 0; j < 32; j++)
            scalars[i][j] = 0;
        *(int *)&scalars[i][0] = i + 1;
        acc += (i + 1) * (i + 1);
    }

    G1Point p1;
    G1.multiMulByScalar(p1, bases, (uint8_t *)scalars, 32, NMExp);

    mpz_t e;
    mpz_init_set_ui(e, acc); // init to an unsigned integer

    Scalar sAcc;
    std::fill(std::begin(sAcc), std::end(sAcc), 0);

    // export a multi-precision integer `e` into a more traditional byte array format `sAcc`;
    // each block is 8 bytes
    // blocks are little endian order; and witin each block, it is also little endian order;
    mpz_export((void *)sAcc, NULL, -1, 8, -1, 0, e);
    mpz_clear(e);

    G1Point p2;
    G1.mulByScalar(p2, G1.one(), sAcc, 32);

    ASSERT_TRUE(G1.eq(p1, p2));

    g1_affine_t *gpu_result = new g1_affine_t{};
    size_t sz = sizeof(g1_affine_t);

    std::cout << "size of g1_affine_t: " << sizeof(g1_affine_t) << " size of scalar_field_t: " << sizeof(scalar_field_t) << std::endl;
    size_t device_id = 0;
    MSM_Config cfg{
        ffi_affine_sz : static_cast<uint32_t>(sz),
        npoints : static_cast<uint32_t>(NMExp),
        are_input_point_in_mont : true,
        are_input_scalar_in_mont : false,
        are_output_point_in_mont : false,
        are_inputs_on_device : false,
        are_outputs_on_device : false,
    };
    mult_pippenger_g1(device_id, gpu_result, (g1_affine_t *)bases, (scalar_field_t *)scalars, cfg);

    // remain in Montgmery Space
    F1Element *gpu_x = (F1Element *)(&gpu_result->y);
    F1Element *gpu_y = (F1Element *)(&gpu_result->y);

    G1PointAffine gpu_point_result{
        x : F1.zero(),
        y : F1.zero(),
    };
    uint32_t *x = gpu_result->x.export_limbs();
    uint32_t *y = gpu_result->y.export_limbs();
    F1.fromRprLE(gpu_point_result.x, (uint8_t *)(x), 32);
    F1.fromRprLE(gpu_point_result.y, (uint8_t *)(y), 32);
    ASSERT_TRUE(G1.eq(p1, gpu_point_result));
    delete[] bases;
    delete[] scalars;
}

TEST(altBn128, msm_inputs_on_device_bn254_g1_curve_gpu_consistency_with_cpu)
{
    int NMExp = 1 << 12;

    typedef uint8_t Scalar[32];

    Scalar *scalars = new Scalar[NMExp];
    G1PointAffine *bases = new G1PointAffine[NMExp];

    uint64_t acc = 0;
    for (int i = 0; i < NMExp; i++)
    {
        if (i == 0)
        {
            G1.copy(bases[0], G1.one());
        }
        else
        {
            G1.add(bases[i], bases[i - 1], G1.one());
        }
        for (int j = 0; j < 32; j++)
            scalars[i][j] = 0;
        *(int *)&scalars[i][0] = i + 1;
        acc += (i + 1) * (i + 1);
    }

    G1Point p1;
    G1.multiMulByScalar(p1, bases, (uint8_t *)scalars, 32, NMExp);

    mpz_t e;
    mpz_init_set_ui(e, acc); // init to an unsigned integer

    Scalar sAcc;
    std::fill(std::begin(sAcc), std::end(sAcc), 0);

    // export a multi-precision integer `e` into a more traditional byte array format `sAcc`;
    // each block is 8 bytes
    // blocks are little endian order; and witin each block, it is also little endian order;
    mpz_export((void *)sAcc, NULL, -1, 8, -1, 0, e);
    mpz_clear(e);

    G1Point p2;
    G1.mulByScalar(p2, G1.one(), sAcc, 32);

    ASSERT_TRUE(G1.eq(p1, p2));

    g1_affine_t *gpu_result = new g1_affine_t{};
    size_t sz = sizeof(g1_affine_t);

    std::cout << "size of g1_affine_t: " << sizeof(g1_affine_t) << " size of scalar_field_t: " << sizeof(scalar_field_t) << std::endl;
    size_t device_id = 0;
    MSM_Config cfg{
        ffi_affine_sz : static_cast<uint32_t>(sz),
        npoints : static_cast<uint32_t>(NMExp),
        are_input_point_in_mont : true,
        are_input_scalar_in_mont : false,
        are_output_point_in_mont : false,
        are_inputs_on_device : true,
        are_outputs_on_device : false,
    };

    g1_affine_t *d_points;
    cudaMalloc((void **)&d_points, NMExp * sizeof(g1_affine_t));
    cudaMemcpy(d_points, bases, NMExp * sizeof(g1_affine_t), cudaMemcpyHostToDevice);

    scalar_field_t *d_scalars;
    cudaMalloc((void **)&d_scalars, NMExp * sizeof(scalar_field_t));
    cudaMemcpy(d_scalars, scalars, NMExp * sizeof(scalar_field_t), cudaMemcpyHostToDevice);

    mult_pippenger_g1(device_id, gpu_result, (g1_affine_t *)d_points, (scalar_field_t *)d_scalars, cfg);

    // remain in Montgmery Space
    F1Element *gpu_x = (F1Element *)(&gpu_result->x);
    F1Element *gpu_y = (F1Element *)(&gpu_result->y);

    G1PointAffine gpu_point_result{
        x : F1.zero(),
        y : F1.zero(),
    };
    uint32_t *x = gpu_result->x.export_limbs();
    uint32_t *y = gpu_result->y.export_limbs();
    F1.fromRprLE(gpu_point_result.x, (uint8_t *)(x), 32);
    F1.fromRprLE(gpu_point_result.y, (uint8_t *)(y), 32);
    ASSERT_TRUE(G1.eq(p1, gpu_point_result));

    delete[] bases;
    delete[] scalars;
}
#endif

#if defined(G2_ENABLED)

TEST(altBn128, msm_bn254_g2_curve_gpu_consistency_with_cpu)
{
    unsigned batch_size = 1;
    int lg_n_size = 12;

    unsigned msm_size = 1 << lg_n_size;
    unsigned N = batch_size * msm_size;

    scalar_field_t *scalars = new scalar_field_t[N];
    g2_affine_t *points = new g2_affine_t[N];

    for (unsigned i = 0; i < N; i++)
    {
        points[i] = (i % msm_size < 10) ? g2_projective_t::to_affine(g2_projective_t::rand_host()) : points[i - 10];
        scalars[i] = scalar_field_t::rand_host();
    }
    uint32_t large_bucket_factor = 2;
    g2_affine_t *gpu_result_affine = new g2_affine_t();
    // std::cout << gpu_result_projective[0] << std::endl;
    // std::cout << "size of affine_t: " << sizeof(affine_t) << " size of fr_t: " << sizeof(fr_t) << std::endl;
    size_t device_id = 0;
    MSM_Config cfg{
        npoints : static_cast<uint32_t>(N),
        are_inputs_on_device : false,
        large_bucket_factor : large_bucket_factor,
    };
    mult_pippenger_g2(device_id, gpu_result_affine, points, scalars, cfg);

    G2Point cpu_result_projective;
    G2PointAffine cpu_result_affine;
    G2PointAffine cpu_base_points_affine[N] = {};

    for (int i = 0; i < N; i++)
    {
        g2_affine_t g2_affine = points[i];

        uint32_t *x_real = g2_affine.x.real.export_limbs();
        uint32_t *x_imag = g2_affine.x.imaginary.export_limbs();
        uint32_t *y_real = g2_affine.y.real.export_limbs();
        uint32_t *y_imag = g2_affine.y.imaginary.export_limbs();
        F1.fromRprLE(cpu_base_points_affine[i].x.a, (uint8_t *)(x_real), 32);
        F1.fromRprLE(cpu_base_points_affine[i].x.b, (uint8_t *)(x_imag), 32);
        F1.fromRprLE(cpu_base_points_affine[i].y.a, (uint8_t *)(y_real), 32);
        F1.fromRprLE(cpu_base_points_affine[i].y.b, (uint8_t *)(y_imag), 32);
    }

    G2.multiMulByScalar(cpu_result_projective, cpu_base_points_affine, (uint8_t *)scalars, 32, N);
    G2.copy(cpu_result_affine, cpu_result_projective);

    G2PointAffine gpu_result_affine_in_host_format;
    uint32_t *x_real = gpu_result_affine->x.real.export_limbs();
    uint32_t *x_imag = gpu_result_affine->x.imaginary.export_limbs();
    uint32_t *y_real = gpu_result_affine->y.real.export_limbs();
    uint32_t *y_imag = gpu_result_affine->y.imaginary.export_limbs();
    F1.fromRprLE(gpu_result_affine_in_host_format.x.a, (uint8_t *)(x_real), 32);
    F1.fromRprLE(gpu_result_affine_in_host_format.x.b, (uint8_t *)(x_imag), 32);
    F1.fromRprLE(gpu_result_affine_in_host_format.y.a, (uint8_t *)(y_real), 32);
    F1.fromRprLE(gpu_result_affine_in_host_format.y.b, (uint8_t *)(y_imag), 32);

    ASSERT_TRUE(G2.eq(cpu_result_affine, gpu_result_affine_in_host_format));
}

// TEST(altBn128, msm_bn254_g2_with_zero_points)
// {

//     size_t N = 32;
//     typedef uint8_t Scalar[32];
//     Scalar *scalars = new Scalar[N];

//     G2PointAffine *points_cpu = new G2PointAffine[N];

//     uint64_t acc = 0;
//     for (unsigned i = 0; i < N; i++)
//     {
//         if (i == 0)
//         {
//             G2.copy(points_cpu[0], G2.zeroAffine());
//         }
//         else
//         {
//             G2.add(points_cpu[i], points_cpu[i - 1], G2.oneAffine());
//         }
//         *(int *)&scalars[i][0] = i + 1;
//         acc += (i * (i + 1));
//     }
//     // printf("generation done\n");
//     G2Point cpu_result_projective;
//     G2.multiMulByScalar(cpu_result_projective, points_cpu, (uint8_t *)scalars, 32, N);

//     G2Point cpu_result_expected = G2.zero();
//     for (int i = 0; i < acc; i++)
//     {
//         G2.add(cpu_result_expected, cpu_result_expected, G2.one());
//     }

//     ASSERT_TRUE(G2.eq(cpu_result_projective, cpu_result_expected));

//     g2_affine_t *points_gpu = (g2_affine_t *)malloc(sizeof(g2_affine_t) * N);
//     for (int i = 0; i < N; i++)
//     {
//         F1.toRprLE((points_cpu + i)->x.a, (uint8_t *)((points_gpu + i)->x.real.export_limbs()), 32);
//         F1.toRprLE((points_cpu + i)->x.b, (uint8_t *)((points_gpu + i)->x.imaginary.export_limbs()), 32);
//         F1.toRprLE((points_cpu + i)->y.a, (uint8_t *)((points_gpu + i)->y.real.export_limbs()), 32);
//         F1.toRprLE((points_cpu + i)->y.b, (uint8_t *)((points_gpu + i)->y.imaginary.export_limbs()), 32);
//     }

//     g2_projective_t *gpu_result_projective = (g2_projective_t *)malloc(sizeof(g2_projective_t));

//     // *(int *)&scalars[0][0] = 0;
//     memset(scalars, 0, 32); // should set to zero as the corresponding point is zero. otherwise, kernel will return zero
//     mult_pippenger_g2(gpu_result_projective, points_gpu, N, (scalar_field_t *)scalars, 10, false, false);
//     g2_affine_t gpu_result_affine = g2_projective_t::to_affine(*gpu_result_projective);

//     G2PointAffine gpu_result_affine_in_host_format;
//     F1.fromRprLE(gpu_result_affine_in_host_format.x.a, (uint8_t *)(gpu_result_affine.x.real.export_limbs()), 32);
//     F1.fromRprLE(gpu_result_affine_in_host_format.x.b, (uint8_t *)(gpu_result_affine.x.imaginary.export_limbs()), 32);
//     F1.fromRprLE(gpu_result_affine_in_host_format.y.a, (uint8_t *)(gpu_result_affine.y.real.export_limbs()), 32);
//     F1.fromRprLE(gpu_result_affine_in_host_format.y.b, (uint8_t *)(gpu_result_affine.y.imaginary.export_limbs()), 32);

//     G2Point result_gpu;
//     G2.copy(result_gpu, gpu_result_affine_in_host_format);

//     ASSERT_TRUE(G2.eq(result_gpu, cpu_result_expected));
// }
#endif // G2_ENABLED
#endif // BUILD_MSM
#endif // FEATURE_BN254
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
