#include <cstdint>
#include <blst_t.hpp>
#include <vect.h>
#include <ntt/ntt.h>
#include "iostream"
#include "src/lib.h"
#include <vector>

void print_vector(void *p, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {
        uint64_t *uint64Pointer = reinterpret_cast<uint64_t *>(p);
        std::cout << uint64Pointer[i] << std::endl;
    }
}

int main()
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
    return 0;
}