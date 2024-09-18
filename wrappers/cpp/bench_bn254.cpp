#if defined(FEATURE_BN254)
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

void bench_fft(int lg_n_size)
{

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

void bench_msm_bn254_g1(int lg_n_size)
{

    unsigned batch_size = 1;
    unsigned msm_size = 1 << lg_n_size;
    unsigned N = batch_size * msm_size;

    uint8_t *raw_scalars = (uint8_t *)malloc(N * 32);
    uint8_t *raw_point_x = (uint8_t *)malloc(N * 32);
    uint8_t *raw_point_y = (uint8_t *)malloc(N * 32);

    for (int i = 0; i < N * 32; i++)
    {
        *(raw_scalars + i) = i % 32 == 31 ? 0 : lehmer64();
        *(raw_point_x + i) = i % 32 == 31 ? 0 : lehmer64();
        *(raw_point_y + i) = i % 32 == 31 ? 0 : lehmer64();
    }

    G1PointAffine *cpu_points = (G1PointAffine *)malloc(N * sizeof(G1PointAffine));
    for (int i = 0; i < N; i++)
    {
        F1.fromRprLE((cpu_points + i)->x, (uint8_t *)(raw_point_x + i), 32);
        F1.fromRprLE((cpu_points + i)->y, (uint8_t *)(raw_point_y + i), 32);
    }

    G1Point p1;
    double start_cpu, end_cpu;
    double cpu_time_used;
    start_cpu = omp_get_wtime();
    G1.multiMulByScalar(p1, cpu_points, raw_scalars, 32, N);
    end_cpu = omp_get_wtime();
    cpu_time_used = ((double)(end_cpu - start_cpu));
    printf("Time used  cpu msm g1 (ms): %.3lf\n", cpu_time_used * 1000); // lf stands for long float

    point_t *gpu_result = new point_t{};
    size_t sz = sizeof(affine_t);
    double start_gpu, end_gpu;
    double gpu_time_used;
    start_gpu = omp_get_wtime();

    mult_pippenger(gpu_result, (affine_t *)cpu_points, N, (fr_t *)raw_scalars, sz);
    end_gpu = omp_get_wtime();
    gpu_time_used = ((double)(end_gpu - start_gpu));
    printf("Time used  gpu msm g1 (ms): %.3lf\n", gpu_time_used * 1000); // lf stands for long float

    delete[] cpu_points;
}

#if defined(G2_ENABLED)
void bench_msm_bn254_g2(int lg_n_size)
{
    unsigned msm_size = 1 << lg_n_size;

    scalar_field_t *scalars = new scalar_field_t[msm_size];
    g2_affine_t *points = new g2_affine_t[msm_size];

    for (unsigned i = 0; i < msm_size; i++)
    {
        points[i] = (i % msm_size < 100) ? g2_projective_t::to_affine(g2_projective_t::rand_host()) : points[i - 100];
        scalars[i] = scalar_field_t::rand_host();
    }
    size_t large_bucket_factor = 10;
    g2_projective_t *gpu_result_projective = new g2_projective_t();

    double start_gpu, end_gpu;
    double gpu_time_used;
    start_gpu = omp_get_wtime();
    mult_pippenger_g2(gpu_result_projective, points, msm_size, scalars, large_bucket_factor, false, false);
    end_gpu = omp_get_wtime();
    gpu_time_used = ((double)(end_gpu - start_gpu));
    printf("Time used  gpu msm bn254 g2 (ms): %.3lf\n", gpu_time_used * 1000); // lf stands for long float

    G2Point cpu_result_projective;
    G2PointAffine *cpu_base_points_affine = (G2PointAffine *)points;

    for (int i = 0; i < msm_size; i++)
    {
        F1.fromRprLE(cpu_base_points_affine[i].x.a, (uint8_t *)((points + i)->x.real.export_limbs()), 32);
        F1.fromRprLE(cpu_base_points_affine[i].x.b, (uint8_t *)((points + i)->x.imaginary.export_limbs()), 32);
        F1.fromRprLE(cpu_base_points_affine[i].y.a, (uint8_t *)((points + i)->y.real.export_limbs()), 32);
        F1.fromRprLE(cpu_base_points_affine[i].y.b, (uint8_t *)((points + i)->y.imaginary.export_limbs()), 32);
    }
    double start_cpu, end_cpu;
    double cpu_time_used;
    start_cpu = omp_get_wtime();
    G2.multiMulByScalar(cpu_result_projective, cpu_base_points_affine, (uint8_t *)scalars, 32, msm_size);
    end_cpu = omp_get_wtime();
    cpu_time_used = ((double)(end_cpu - start_cpu));
    printf("Time used  cpu msm bn254 g2 (ms): %.3lf\n", cpu_time_used * 1000); // lf stands for long float
    delete[] cpu_base_points_affine;
    delete[] scalars;
}
#endif

#endif
int main(int argc, char **argv)
{
#if defined(FEATURE_BN254)
    int lg_n_size = atoi(argv[1]);
// bench_fft(lg_n_size);
// bench_msm_bn254_g1(lg_n_size);
#if defined(G2_ENABLED)
    bench_msm_bn254_g2(lg_n_size);
#endif
#endif
}
