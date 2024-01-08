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

TEST(altBn128, msm_self_consistency) {

    int NMExp = 40000;

    typedef uint8_t Scalar[32];

    Scalar *scalars = new Scalar[NMExp];
    G1PointAffine *bases = new G1PointAffine[NMExp];

    uint64_t acc=0;
    for (int i=0; i<NMExp; i++) {
        if (i==0) {
            G1.copy(bases[0], G1.one());
        } else {
            G1.add(bases[i], bases[i-1], G1.one());
        }
        for (int j=0; j<32; j++) scalars[i][j] = 0;
        *(int *)&scalars[i][0] = i+1;
        acc += (i+1)*(i+1);
    }

    G1Point p1;
    G1.multiMulByScalar(p1, bases, (uint8_t *)scalars, 32, NMExp);

    mpz_t e;
    mpz_init_set_ui(e, acc);

    Scalar sAcc;

    for (int i=0;i<32;i++) sAcc[i] = 0;
    mpz_export((void *)sAcc, NULL, -1, 8, -1, 0, e);
    mpz_clear(e);

    G1Point p2;
    G1.mulByScalar(p2, G1.one(), sAcc, 32);

    ASSERT_TRUE(G1.eq(p1, p2));

    delete[] bases;
    delete[] scalars;
}

TEST(altBn128, msm_cpu_gpu_consistency) {

    int NMExp = 1<<14;

    typedef uint8_t Scalar[32];

    Scalar *scalars = new Scalar[NMExp];
    G1PointAffine *bases = new G1PointAffine[NMExp];

    uint64_t acc=0;
    for (int i=0; i<NMExp; i++) {
        if (i==0) {
            G1.copy(bases[0], G1.one());
        } else {
            G1.add(bases[i], bases[i-1], G1.one());
        }
        for (int j=0; j<32; j++) scalars[i][j] = 0;
        *(int *)&scalars[i][0] = i+1;
        acc += (i+1)*(i+1);
    }

    G1Point p1;
    G1.multiMulByScalar(p1, bases, (uint8_t *)scalars, 32, NMExp);

    mpz_t e;
    mpz_init_set_ui(e, acc);

    Scalar sAcc;

    for (int i=0;i<32;i++) sAcc[i] = 0;
    mpz_export((void *)sAcc, NULL, -1, 8, -1, 0, e);
    mpz_clear(e);

    G1Point p2;
    G1.mulByScalar(p2, G1.one(), sAcc, 32);

    ASSERT_TRUE(G1.eq(p1, p2));


    point_t *gpu_result = new point_t{};
    size_t sz = sizeof(affine_t);

    mult_pippenger(gpu_result, (affine_t*)bases, NMExp, (fr_t *)scalars, sz);



    // remain in Montgmery Space
    F1Element* gpu_x = (F1Element*)(&gpu_result->X);
    F1Element* gpu_y= (F1Element*)(&gpu_result->Y);
    F1Element* gpu_z= (F1Element*)(&gpu_result->Z);

    G1Point gpu_point_result{
        x : F1.zero(),
        y : F1.zero(),
        zz : F1.zero(),
        zzz : F1.zero(),
    };
    F1.copy(gpu_point_result.x, *gpu_x);
    F1.copy(gpu_point_result.y, *gpu_y);
    F1.square(gpu_point_result.zz, *gpu_z);
    F1.mul(gpu_point_result.zzz, gpu_point_result.zz, *gpu_z);
    print_g1_point(gpu_point_result);
    ASSERT_TRUE(G1.eq(p1, gpu_point_result));

    delete[] bases;
    delete[] scalars;
}

TEST(altBn128, msm2)
{

    int N = 1<<1;

    // uint8_t scalars[N*32] = {};
    typedef uint8_t Scalar[32];

    Scalar *scalars = new Scalar[N];
    uint8_t input_points_x[N*32] = {};
    uint8_t input_points_y[N*32] = {};

    for (int i = 0; i < N * 32; i++)
    {
        uint8_t random_scalar = random_byte();
        // TODO: why scalar matters, range of fr_t. get real fr_t from rapidsnark
        *(int *)&scalars[i][0] = i+1;
  
        uint8_t random_x = random_byte();
        input_points_x[i] = i % 32 == 31 ? 0 : random_x; // TODO: this is to make the input less than MOD; otherwise, the test will fail

        uint8_t random_y = random_byte();
        input_points_y[i] = i % 32 == 31 ? 0 : random_y; // TODO: this is to make the input less than MOD; otherwise, the test will fail
    }     
    G1PointAffine cpu_base_points_affine[N] = {};
    affine_t gpu_base_points_affine[N] ={}; 

    for(int i=0; i < N; i++) {
        G1PointAffine cpu_point_affine = G1.zeroAffine();
        F1.fromRprLE(cpu_point_affine.x, ((const uint8_t *)input_points_x+i*32), 32);
        F1.fromRprLE(cpu_point_affine.y, ((const uint8_t *)input_points_y+i*32), 32);
        cpu_base_points_affine[i]=cpu_point_affine;
    
        fp_t x = *(fp_t *)cpu_point_affine.x.v;
        fp_t y = *(fp_t *)cpu_point_affine.y.v;
        affine_t p_affine = affine_t(x, y);
        gpu_base_points_affine[i]=p_affine;
    }

    for(int i=0; i < N; i++){
        printf("cpu input base point: %d \n", i);
        print_char_array((uint8_t*)(uint64_t*)(&cpu_base_points_affine[i].x),32);
        printf("cpu input scalar %d \n", i);
        print_char_array((uint8_t*)scalars+i*32,32);
    }
    G1Point cpu_result;
    G1.multiMulByScalar(cpu_result, cpu_base_points_affine, (uint8_t*)scalars, 32, N);

    // print_g1_point(cpu_result);

    fr_t gpu_scalars[N] ={};
    for(int i=0; i < N; i++) {
      gpu_scalars[i]= *((fr_t *)scalars+i);
    }

    point_t *gpu_result = new point_t{};
    size_t sz = sizeof(affine_t);
    for(int i=0; i < N; i++){
        printf("gpu input base point: %d \n", i);
        print_char_array((uint8_t*)(uint64_t*)(&gpu_base_points_affine[i].X),32);

         printf("gpu input scalar %d \n", i);
        print_char_array((uint8_t*)(uint64_t*)(&gpu_scalars[i]),32);
    }
    mult_pippenger(gpu_result, gpu_base_points_affine, N, gpu_scalars, sz);



    // remain in Montgmery Space
    F1Element* gpu_x = (F1Element*)(&gpu_result->X);
    F1Element* gpu_y= (F1Element*)(&gpu_result->Y);
    F1Element* gpu_z= (F1Element*)(&gpu_result->Z);

    G1Point gpu_point_result{
        x : F1.zero(),
        y : F1.zero(),
        zz : F1.zero(),
        zzz : F1.zero(),
    };
    F1.copy(gpu_point_result.x, *gpu_x);
    F1.copy(gpu_point_result.y, *gpu_y);
    F1.square(gpu_point_result.zz, *gpu_z);
    F1.mul(gpu_point_result.zzz, gpu_point_result.zz, *gpu_z);
    print_g1_point(gpu_point_result);
    ASSERT_TRUE(G1.eq(cpu_result, gpu_point_result));


}

TEST(altBn128, msm_n)
{

    int N = 1<<1;

    uint8_t scalars[N*32] = {};

    for (int i = 0; i < N * 32; i++)
    {
        scalars[i] = i % 32 == 0 ? 1 : 0; // TODO: this is to make the input less than MOD; otherwise, the test will fail       
    }     
    G1PointAffine cpu_base_points_affine[N] = {};
    affine_t gpu_base_points_affine[N] ={}; 

    for(int i=0; i < N; i++) {
        G1PointAffine cpu_point_affine = G1.oneAffine();
        cpu_base_points_affine[i]=cpu_point_affine;

        G1Point p;
        G1.copy(p, cpu_point_affine);
    
        fp_t x = *(fp_t *)cpu_point_affine.x.v;
        fp_t y = *(fp_t *)cpu_point_affine.y.v;
        affine_t p_affine = affine_t(x, y);
        gpu_base_points_affine[i]=p_affine;
    }

     for(int i=0; i < N; i++) {
        affine_t p = gpu_base_points_affine[i];
     }

    G1Point cpu_result;
    G1.multiMulByScalar(cpu_result, cpu_base_points_affine, scalars, 32, N);

    G1PointAffine cpu_expected_point_affine;
    G1.dbl(cpu_expected_point_affine,  G1.oneAffine());
    G1Point cpu_expected_point;
    G1.copy(cpu_expected_point, cpu_expected_point_affine);



    fr_t gpu_scalars[N] ={};
    for(int i=0; i < N; i++) {
       gpu_scalars[i]= *((fr_t *)scalars+i);
    }

    size_t fr_size = sizeof(fr_t);

    point_t* gpu_result = new point_t{};
    size_t sz = sizeof(affine_t);
    mult_pippenger(gpu_result, gpu_base_points_affine, N, gpu_scalars, sz);

    
    // remain in Montgmery Space
    F1Element* gpu_x = (F1Element*)(&gpu_result->X);
    F1Element* gpu_y= (F1Element*)(&gpu_result->Y);
    F1Element* gpu_z= (F1Element*)(&gpu_result->Z);

    G1Point gpu_point_result{
        x : F1.zero(),
        y : F1.zero(),
        zz : F1.zero(),
        zzz : F1.zero(),
    };
    F1.copy(gpu_point_result.x, *gpu_x);
    F1.copy(gpu_point_result.y, *gpu_y);
    F1.square(gpu_point_result.zz, *gpu_z);
    F1.mul(gpu_point_result.zzz, gpu_point_result.zz, *gpu_z);
    print_g1_point(gpu_point_result);
    ASSERT_TRUE(G1.eq(cpu_result, gpu_point_result));


}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}