#include <utils/all_gpus.hpp>
#include <iostream>

gpus_t::gpus_t()
{
    int n;
    if (cudaGetDeviceCount(&n) != cudaSuccess)
        return;
    for (int id = 0; id < n; id++)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, id) == cudaSuccess &&
            prop.major >= 7)
        {
            cudaSetDevice(id);
            gpus.push_back(new gpu_t(gpus.size(), id, prop));
        }
    }
    cudaSetDevice(0);
}
gpus_t::~gpus_t()
{
    for (auto *ptr : gpus)
        delete ptr;
}

const gpu_t &select_gpu(int id)
{
    auto &gpus = gpus_t::all();
    if (id == -1)
    {
        int cuda_id;
        CUDA_OK(cudaGetDevice(&cuda_id));
        for (auto *gpu : gpus)
            if (gpu->cid() == cuda_id)
                return *gpu;
        id = 0;
    }
    auto *gpu = gpus[id];
    gpu->select();
    return *gpu;
}

const cudaDeviceProp &gpu_props(int id)
{
    return gpus_t::all()[id]->props();
}

size_t ngpus()
{
    return gpus_t::all().size();
}

const std::vector<const gpu_t *> &all_gpus()
{
    return gpus_t::all();
}

extern "C" bool cuda_available()
{
    return gpus_t::all().size() != 0;
}

int _ConvertSMVer2Cores(int major, int minor)
{
    // Based on CUDA Compute Capability
    int cores;
    switch ((major << 4) + minor)
    {
    case 0x10: // 1.0
        cores = 8;
        break;
    case 0x11: // 1.1
        cores = 8;
        break;
    case 0x12: // 1.2
        cores = 8;
        break;
    case 0x13: // 1.3
        cores = 8;
        break;
    case 0x20: // 2.0
        cores = 32;
        break;
    case 0x21: // 2.1
        cores = 48;
        break;
    case 0x30: // 3.0
        cores = 192;
        break;
    case 0x32: // 3.2
        cores = 192;
        break;
    case 0x35: // 3.5
        cores = 192;
        break;
    case 0x37: // 3.7
        cores = 192;
        break;
    case 0x50: // 5.0
        cores = 128;
        break;
    case 0x52: // 5.2
        cores = 128;
        break;
    case 0x53: // 5.3
        cores = 128;
        break;
    case 0x60: // 6.0
        cores = 64;
        break;
    case 0x61: // 6.1
        cores = 128;
        break;
    case 0x62: // 6.2
        cores = 128;
        break;
    case 0x70: // 7.0
        cores = 64;
        break;
    case 0x72: // 7.2
        cores = 64;
        break;
    case 0x75: // 7.5
        cores = 64;
        break;
    case 0x80: // 8.0
        cores = 64;
        break;
    case 0x86: // 8.6
        cores = 64;
        break;
    default:
        cores = -1; // Unknown compute capability
        break;
    }

    return cores;
}

void list_all_gpus_prop()
{
    std::vector<const gpu_t *> gpus = all_gpus();
    for (const gpu_t *gpu : gpus)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gpu->id());
        std::cout << "Device " << gpu->id() << " - " << prop.name << std::endl;
        std::cout << "CUDA multi processor count: " << prop.multiProcessorCount << "   CUDA Cores: " << prop.multiProcessorCount * _ConvertSMVer2Cores(prop.major, prop.minor) << std::endl;
        std::cout << "Major: " << prop.major << ", Minor: " << prop.minor << std::endl;
        std::cout << "Integrated: " << prop.integrated << ", canMapHostMemory: " << prop.canMapHostMemory << std::endl;
        std::cout << "unifiedAddressing: " << prop.unifiedAddressing << std::endl;
        std::cout << "maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor << std::endl;
        std::cout << "managedMemory: " << prop.managedMemory << std::endl;
        std::cout << "isMultiGpuBoard: " << prop.isMultiGpuBoard << std::endl;
        std::cout << "pageableMemoryAccess: " << prop.pageableMemoryAccess << std::endl;
        std::cout << "canUseHostPointerForRegisteredMem: " << prop.canUseHostPointerForRegisteredMem << std::endl;
        std::cout << "directManagedMemAccessFromHost: " << prop.directManagedMemAccessFromHost << std::endl;
        std::cout << "memoryPoolsSupported: " << prop.memoryPoolsSupported << std::endl;
        std::cout << "gpuDirectRDMASupported: " << prop.gpuDirectRDMASupported << std::endl;
        std::cout << "clusterLaunch: " << prop.clusterLaunch << std::endl;

        int *canAccessPeer = new int;
        int device = gpu->id();
        int peerDevice = (device + 1) % gpus.size();
        cudaSetDevice(device);
        cudaDeviceEnablePeerAccess(peerDevice, 0);
        cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
        std::cout << "device: " << device << ", peer: " << peerDevice << ", can access: " << *canAccessPeer << std::endl;

        std::cout << std::endl;
    }
}
