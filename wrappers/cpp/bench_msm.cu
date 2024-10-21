#include <iostream>
#include <cuda_runtime.h>
#include <msm/msm.h>
#include "alt_bn128.hpp"
#include <fstream>

using namespace AltBn128;

const int maxPoints = 5000000;
int nPoints;

typedef uint64_t Scalar[4];
Scalar scalars[maxPoints];
G1PointAffine bases[maxPoints];

void readScalar(std::string filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open())
    {
        std::cerr << "Unable to open file\n";
        return;
    }

    // Calculate the file size and seek beginning
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    nPoints = size / sizeof(Scalar);
    if (file.read(buffer.data(), size))
    {
        std::memcpy(scalars, buffer.data(), size);
    }
    else
    {
        std::cerr << "Read error\n";
        return;
    }
}

void readPoint(std::string filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open())
    {
        std::cerr << "Unable to open file\n";
        return;
    }

    // Calculate the file size and seek beginning
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    assert(size / sizeof(G1PointAffine) == nPoints);
    if (file.read(buffer.data(), size))
    {
        std::memcpy(bases, buffer.data(), size);
    }
    else
    {
        std::cerr << "Read error\n";
        return;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <scalar_file> <point_file>" << std::endl;
        return 1;
    }

    std::string scalar_file = argv[1];
    std::string point_file = argv[2];

    readScalar(scalar_file);
    readPoint(point_file);
    size_t device_id = 0;
    point_t *gpu_result = new point_t{};
    size_t sz = sizeof(affine_t);
    MSM_Config cfg{
        ffi_affine_sz : sz,
        npoints : nPoints,
        are_points_in_mont : false,
        are_inputs_on_device : false,
        are_outputs_on_device : false,
    };
    std::cout << "MSM: " << nPoints << " points" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    mult_pippenger(device_id, gpu_result, (affine_t *)bases, (fr_t *)scalars, cfg);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "MSM: done in " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}