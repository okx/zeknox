# gdr
https://developer.nvidia.com/gdrcopy

# pinned pageable memory
https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/

# nvlink command
```
nvidia-smi nvlink -h
```
## Showing NVLINK Status For Different GPUs
To show active NVLINK Connections, you must specify GPU index via -i
```
nvidia-smi nvlink --status -i 0
```
example output
```
GPU 0: NVIDIA H100 PCIe (UUID: GPU-a1bf2ff6-a98d-edac-a422-0bd80fbd9724)
         Link 0: 26.562 GB/s
         Link 1: 26.562 GB/s
         Link 2: 26.562 GB/s
         Link 3: 26.562 GB/s
         Link 4: <inactive>
         Link 5: 26.562 GB/s
         Link 6: 26.562 GB/s
         Link 7: 26.562 GB/s
         Link 8: 26.562 GB/s
         Link 9: <inactive>
         Link 10: <inactive>
         Link 11: 26.562 GB/s
         Link 12: 26.562 GB/s
         Link 13: 26.562 GB/s
         Link 14: 26.562 GB/s
         Link 15: <inactive>
         Link 16: <inactive>
```
## Display & Explore NVLINK Capabilities Per Link
Allows you to query to ensure each link associated with the GPU Index (specified by -i #) has specific capabilities related to P2P, System Memory, P2P Atomics, SLI.
```
nvidia-smi nvlink --capabilities -i 1
```
example output
```
Link 0, P2P is supported: true
Link 0, Access to system memory supported: true
Link 0, P2P atomics supported: true
Link 0, System memory atomics supported: false
Link 0, SLI is supported: false
Link 0, Link is supported: false
Link 1, P2P is supported: true
Link 1, Access to system memory supported: true
Link 1, P2P atomics supported: true
Link 1, System memory atomics supported: false
Link 1, SLI is supported: false
Link 1, Link is supported: false
Link 2, P2P is supported: true
Link 2, Access to system memory supported: true
Link 2, P2P atomics supported: true
Link 2, System memory atomics supported: false
Link 2, SLI is supported: false
Link 2, Link is supported: false
Link 3, P2P is supported: true
Link 3, Access to system memory supported: true
Link 3, P2P atomics supported: true
Link 3, System memory atomics supported: false
Link 3, SLI is supported: false
Link 3, Link is supported: false
```

## NVLink Usage Counters
nvidia-smi nvlink -g N -i N allows you to view the data being traversed on the different NVLink Link.
```
nvidia-smi nvlink -g 0 -i 0
```


# benchmarking p2p mem_copy
code for testing p2p memCopy
```
// P2P Test by Greg Gutmann
 
#include "stdio.h"
#include "stdint.h"
 
int main()
{
    // GPUs
    int gpuid_0 = 0;
    int gpuid_1 = 1;
 
    // Memory Copy Size
    uint32_t size = pow(2, 26); // 2^26 = 67MB
 
    // Allocate Memory
    uint32_t* dev_0;
    cudaSetDevice(gpuid_0);
    cudaMalloc((void**)&dev_0, size);
 
    uint32_t* dev_1;
    cudaSetDevice(gpuid_1);
    cudaMalloc((void**)&dev_1, size);
 
    //Check for peer access between participating GPUs: 
    int can_access_peer_0_1;
    int can_access_peer_1_0;
    cudaDeviceCanAccessPeer(&can_access_peer_0_1, gpuid_0, gpuid_1);
    cudaDeviceCanAccessPeer(&can_access_peer_1_0, gpuid_1, gpuid_0);
    printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_0, gpuid_1, can_access_peer_0_1);
    printf("cudaDeviceCanAccessPeer(%d->%d): %d\n", gpuid_1, gpuid_0, can_access_peer_1_0);
 
    if (can_access_peer_0_1 && can_access_peer_1_0) {
        // Enable P2P Access
        cudaSetDevice(gpuid_0);
        cudaDeviceEnablePeerAccess(gpuid_1, 0);
        cudaSetDevice(gpuid_1);
        cudaDeviceEnablePeerAccess(gpuid_0, 0);
    }
 
    // Init Timing Data
    uint32_t repeat = 10;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    // Init Stream
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
 
    // ~~ Start Test ~~
    cudaEventRecord(start, stream);
 
    //Do a P2P memcpy
    for (int i = 0; i < repeat; ++i) {
        cudaMemcpyAsync(dev_0, dev_1, size, cudaMemcpyDeviceToDevice, stream);
    }
 
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);
    // ~~ End of Test ~~
 
    // Check Timing & Performance
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);
    double time_s = time_ms / 1e3;
 
    double gb = size * repeat / (double)1e9;
    double bandwidth = gb / time_s;
 
    printf("Seconds: %f\n", time_s);
    printf("Unidirectional Bandwidth: %f (GB/s)\n", bandwidth);
 
    if (can_access_peer_0_1 && can_access_peer_1_0) {
        // Shutdown P2P Settings
        cudaSetDevice(gpuid_0);
        cudaDeviceDisablePeerAccess(gpuid_1);
        cudaSetDevice(gpuid_1);
        cudaDeviceDisablePeerAccess(gpuid_0);
    }
 
    // Clean Up
    cudaFree(dev_0);
    cudaFree(dev_1);
 
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
}
```

## some results
- on H100
Unidirectional Bandwidth: 253.631489 (GB/s)
- on 4090
Unidirectional Bandwidth: 25 (GB/s)