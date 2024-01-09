#include <cstdint>
#include <gmp.h>
#include <vector>
#include "../src/lib.h"
#include "iostream"
#include "alt_bn128.hpp"

using namespace AltBn128;

void print_vector(void *p, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        uint64_t *uint64Pointer = reinterpret_cast<uint64_t *>(p);
        std::cout << uint64Pointer[i] << std::endl;
    }
}

void test_fft()
{
    fr_t *data_in = (fr_t *)malloc(2 * sizeof(fr_t));

    uint64_t data0[4] = {1, 0, 0, 0};
    uint64_t *p0 = data0;
    data_in[0] = *(fr_t *)p0;

    uint64_t data1[4] = {2, 0, 0, 0};
    uint64_t *p1 = data1;
    data_in[1] = *(fr_t *)p1;

    size_t device_id = 0;
    compute_ntt(device_id, data_in, 1, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);
    print_vector(data_in, 8);
    // point_t* out = new point_t{};
    // mult_pippenger_inf(out, {}, 0, {}, 0);
    return;
}

int main()
{
    int NMExp = 4;
    typedef uint8_t Scalar[32];

    Scalar *scalars = new Scalar[NMExp];
    G1PointAffine *bases = new G1PointAffine[NMExp];
    uint64_t acc = 0;
    // for (int i = 0; i < NMExp; i++)
    // {
    //     if (i == 0)
    //     {
    //         G1.copy(bases[0], G1.one());
    //     }
    //     else
    //     {
    //         G1.add(bases[i], bases[i - 1], G1.one());
    //     }
    //     for (int j = 0; j < 32; j++)
    //         scalars[i][j] = 0;
    //     *(int *)&scalars[i][0] = i + 1;
    //     acc += (i + 1) * (i + 1);
    // }
    // G1Point p1;
    // G1.multiMulByScalar(p1, bases, (uint8_t *)scalars, 32, NMExp);
    return 0;
}