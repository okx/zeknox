#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <blst_t.hpp>
#include <vect.h>
#include <ntt/ntt.h>
#include "../src/lib.h"
#include <gmp.h>
#include "alt_bn128.hpp"
#include <time.h>
#include <cassert>
#include "fft.hpp"


using namespace AltBn128;

__uint128_t g_lehmer64_state = 0xAAAAAAAAAAAAAAAALL;

// Fast random generator
// https://lemire.me/blog/2019/03/19/the-fastest-conventional-random-number-generator-that-can-pass-big-crush/

uint64_t lehmer64()
{
    g_lehmer64_state *= 0xda942042e4dd58b5LL;
    return g_lehmer64_state >> 64;
}

void print_vector(void *p, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        uint64_t *uint64Pointer = reinterpret_cast<uint64_t *>(p);
        std::cout << uint64Pointer[i] << std::endl;
    }
}

int main(int argc, char **argv)
{

    int lg_n_size = atoi(argv[1]);
    int N = 1 << lg_n_size;
    printf("start bench bn128 ntt with domain size: %d\n", N);

    uint8_t *data_cpu = new uint8_t[N * 32]{
        1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 
        2,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    uint8_t *data_gpu = new uint8_t[N * 32]{
        1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 
        2,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    // for (int i = 0; i < N * 4; i++)
    // {
    //     uint64_t random_int = lehmer64();
    //     *((uint64_t *)(data_cpu + i * 8)) = i%4 ==0?random_int:0;
    //     *((uint64_t *)(data_gpu + i * 8)) = i%4 ==0?random_int:0;
    //     printf("generated random u64 is: %lu \n", random_int);
    // }

    AltBn128::FrElement *field_elements_cpu = new AltBn128::FrElement[N];
    for (int i = 0; i < N; i++)
    {
       int result= Fr.fromRprLE(field_elements_cpu[i], (const uint8_t *)(data_cpu + i * 32), 32);
       std::cout << "resutl: " << result << std::endl;
    }
    for (int i = 0; i < N; i++)
    {
       std::string cpu_input = Fr.toString(field_elements_cpu[i], 10);
       std::cout << "cpu_input: " << i << cpu_input << std::endl;
    }



    FFT<typename Engine::Fr> fft(N);
    double start, end;
    double cpu_time_used;
    start = omp_get_wtime();
    fft.fft(field_elements_cpu, N);
    end = omp_get_wtime();
    cpu_time_used = ((double)(end - start));
    printf("Time used fft (us): %.3lf\n", cpu_time_used * 1000000); // lf stands for long float

    for (int i = 0; i < N; i++)
    {
       std::string cpu_output_str = Fr.toString(field_elements_cpu[i], 16);
       std::cout << "cpu_output_str: " << i << cpu_output_str << std::endl;
    }


    // fft.ifft(field_elements_cpu, N);

    AltBn128::FrElement aux;
    for (int i = 0; i < N; i++)
    {
        Fr.fromRprLE(aux, (const uint8_t *)(data_cpu + i * 32), 32);
        assert(Fr.eq(a[i] == aux));
    }

    delete[] field_elements_cpu;

    fr_t *data_in = (fr_t *)malloc(N * sizeof(fr_t));
     for (int i = 0; i < N; i++)
    {
        // cpu one is big endian while gpu is little endian
        uint64_t data[4] = { *((uint64_t *)(data_gpu + i * 32)), *((uint64_t *)(data_gpu + i * 32+8)), *((uint64_t *)(data_gpu + i * 32+16)), *((uint64_t *)(data_gpu + i * 32+24))}; 
        
        printf("use: %lu, %lu, %lu, %lu \n", data[0], data[1], data[2], data[3]);
        uint64_t *p = data;
        data_in[i] = *(fr_t *)p;
    }

    size_t device_id = 0;
    compute_ntt(device_id, data_in, 1, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);
    print_vector(data_in, 8);

}