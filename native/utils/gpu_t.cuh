// Copyright Supranational LLC
// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// the code is based on Supranational sppark lib. many cusotmizations have been made.

#ifndef __ZEKNOX_UTIL_GPU_T_CUH__
#define __ZEKNOX_UTIL_GPU_T_CUH__

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#if __cplusplus < 201103L && !(defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
# error C++11 or later is required.
#endif

#include <atomic>
#include <vector>
#include "exception.cuh"
#include "assert.h"

#ifndef WARP_SZ
#define WARP_SZ 32
#endif

#define ALLOC_MEM true
#define DO_NOT_ALLOC_MEM false
#define LIFE_TIME_MANAGED_MANUALLY true

class gpu_t;
size_t ngpus();
const gpu_t &select_gpu(int id = 0);
const cudaDeviceProp &gpu_props(int id = 0);
const std::vector<const gpu_t *> &all_gpus();
extern "C" bool cuda_available();

class event_t
{
    cudaEvent_t event;

public:
    event_t() : event(nullptr)
    {
        CUDA_OK(cudaEventCreate(&event, cudaEventDisableTiming));
    }
    event_t(cudaStream_t stream) : event(nullptr)
    {
        CUDA_OK(cudaEventCreate(&event, cudaEventDisableTiming));
        CUDA_OK(cudaEventRecord(event, stream));
    }
    ~event_t()
    {
        if (event)
            cudaEventDestroy(event);
    }
    inline operator decltype(event)() const
    {
        return event;
    }

    inline void record(cudaStream_t stream)
    {
        CUDA_OK(cudaEventRecord(event, stream));
    }
    inline void wait(cudaStream_t stream)
    {
        CUDA_OK(cudaStreamWaitEvent(stream, event));
    }
};

struct launch_params_t
{
    dim3 gridDim, blockDim;
    size_t shared;

    launch_params_t(dim3 g, dim3 b, size_t sz = 0) : gridDim(g), blockDim(b), shared(sz) {}
    launch_params_t(int g, int b, size_t sz = 0) : gridDim(g), blockDim(b), shared(sz) {}
};

class stream_t
{
public:
    cudaStream_t stream;

public:
    const int gpu_id;

public:
    stream_t(int id, bool blocking) : gpu_id(id)
    {
        // printf("cuda create stream for gpu: %d, blocking: %d\n", id, blocking);
        if (blocking)
        {
            cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
        }
        else
        {
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        }
    }
    ~stream_t()
    {
        cudaStreamDestroy(stream);
    }
    inline operator decltype(stream)() const { return stream; }
    inline int id() const { return gpu_id; }
    // stream is equal to its gpu id
    inline operator int() const { return gpu_id; }

    inline void *Dmalloc(size_t sz) const
    {
        void *d_ptr;
        CUDA_OK(cudaMallocAsync(&d_ptr, sz, stream));
        return d_ptr;
    }
    inline void Dfree(void *d_ptr) const
    {
        CUDA_OK(cudaFreeAsync(d_ptr, stream));
    }

    template <typename T>
    inline void HtoD(T *dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        if (sz == sizeof(T))
            CUDA_OK(cudaMemcpyAsync(dst, src, nelems * sizeof(T),
                                    cudaMemcpyHostToDevice, stream));
        else
            CUDA_OK(cudaMemcpy2DAsync(dst, sizeof(T), src, sz,
                                      std::min(sizeof(T), sz), nelems,
                                      cudaMemcpyHostToDevice, stream));
    }
    template <typename T>
    inline void HtoD(T &dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        HtoD(&dst, src, nelems, sz);
    }
    template <typename T, typename U>
    inline void HtoD(T &dst, const std::vector<U> &src,
                     size_t sz = sizeof(T)) const
    {
        HtoD(&dst, &src[0], src.size(), sz);
    }
    template <typename T, typename U>
    inline void HtoD(T *dst, const std::vector<U> &src,
                     size_t sz = sizeof(T)) const
    {
        HtoD(dst, &src[0], src.size(), sz);
    }

    template <typename T>
    inline void DtoH(T *dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        if (sz == sizeof(T))
        {
            // printf("1D device to host, gpu_id: %d, stream: %d\n", gpu_id, stream);
            CUDA_OK(cudaMemcpyAsync(dst, src, nelems * sizeof(T),
                                    cudaMemcpyDeviceToHost, stream));
        }

        else
        {
            // printf("2D device to host, gpu_id: %d\n", gpu_id);
            CUDA_OK(cudaMemcpy2DAsync(dst, sizeof(T), src, sz,
                                      std::min(sizeof(T), sz), nelems,
                                      cudaMemcpyDeviceToHost, stream));
        }
    }

    template <typename T>
    inline void DtoD(T *dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        if (sz == sizeof(T))
            CUDA_OK(cudaMemcpyAsync(dst, src, nelems * sizeof(T),
                                    cudaMemcpyDeviceToDevice, stream));
        else
            CUDA_OK(cudaMemcpy2DAsync(dst, sizeof(T), src, sz,
                                      std::min(sizeof(T), sz), nelems,
                                      cudaMemcpyDeviceToDevice, stream));
    }

    template <typename T>
    inline void DtoH(T &dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        DtoH(&dst, src, nelems, sz);
    }
    template <typename T>
    inline void DtoH(std::vector<T> &dst, const void *src,
                     size_t sz = sizeof(T)) const
    {
        printf("copy to host size: %d\n", dst.size());
        DtoH(&dst[0], src, dst.size(), sz);
    }

    template <typename T>
    inline void DtoD(T &dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        DtoD(&dst, src, nelems, sz);
    }

    inline void sync() const
    {
        // printf("sync stream: %d, gpu_id: %d\n", stream, gpu_id);
        CUDA_OK(cudaStreamSynchronize(stream));
    }

    inline void notify(cudaHostFn_t cb, void *data)
    {
        CUDA_OK(cudaLaunchHostFunc(stream, cb, data));
    }

    template <class T>
    inline void notify(T &sema)
    {
        notify([](void *s)
               { reinterpret_cast<T *>(s)->notify(); },
               &sema);
    }

    inline void record(cudaEvent_t event)
    {
        CUDA_OK(cudaEventRecord(event, stream));
    }
    inline void wait(cudaEvent_t event)
    {
        CUDA_OK(cudaStreamWaitEvent(stream, event));
    }
};

class gpu_t
{
public:
    static const size_t FLIP_FLOP = 3;

private:
    int gpu_id;
    int cuda_id; // cuda id is determined by `cudaGetDevice`
    cudaDeviceProp prop;
    size_t total_mem;
    mutable stream_t zero = {gpu_id, true}; // the default stream, zero
    // in each gpu, there are three streams by default, non blocking
    mutable stream_t flipflop[FLIP_FLOP] = {{gpu_id, false}, {gpu_id, false}, {gpu_id, false}};

public:
    gpu_t(int id, int real_id, const cudaDeviceProp &p)
        : gpu_id(id), cuda_id(real_id), prop(p)
    {
        size_t freeMem;
        CUDA_OK(cudaMemGetInfo(&freeMem, &total_mem));
    }

    inline int cid() const { return cuda_id; }
    inline int id() const { return gpu_id; }
    inline operator int() const { return gpu_id; }
    inline const auto &props() const { return prop; }
    inline int sm_count() const { return prop.multiProcessorCount; }
    inline void select() const
    {
        // printf("select GPU id: %d\n", cuda_id);
        cudaSetDevice(cuda_id);
    }
    stream_t &operator[](size_t i) const { return flipflop[i % FLIP_FLOP]; }
    inline operator stream_t &() const { return zero; }
    inline operator cudaStream_t() const { return zero; }

    /*stream allocate memory, return the pointer to the allocation*/
    inline void *Dmalloc(size_t sz) const
    {
        void *d_ptr = zero.Dmalloc(sz);
        zero.sync();
        return d_ptr;
    }
    inline void Dfree(void *d_ptr) const
    {
        zero.Dfree(d_ptr);
        zero.sync();
    }

    template <typename T>
    inline void HtoD(T *dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        zero.HtoD(dst, src, nelems, sz);
    }
    template <typename T>
    inline void HtoD(T &dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        HtoD(&dst, src, nelems, sz);
    }

    /** copy from host vector `src` to device `dst`*/
    template <typename T, typename U>
    inline void HtoD(T &dst, const std::vector<U> &src,
                     size_t sz = sizeof(T)) const
    {
        HtoD(&dst, &src[0], src.size(), sz);
    }

    template <typename T>
    inline void DtoH(T *dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        zero.DtoH(dst, src, nelems, sz);
    }

    template <typename T>
    inline void DtoD(T *dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        zero.DtoD(dst, src, nelems, sz);
    }

    template <typename T>
    inline void DtoH(T &dst, const void *src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {
        DtoH(&dst, src, nelems, sz);
    }
    template <typename T>
    inline void DtoH(std::vector<T> &dst, const void *src,
                     size_t sz = sizeof(T)) const
    {
        DtoH(&dst[0], src, dst.size(), sz);
    }

    inline void sync() const
    {
        zero.sync(); // sync the default stream
        for (auto &f : flipflop)
        {
            f.sync();
        }
    }
};

template <typename T>
class gpu_ptr_t
{
    struct inner
    {
        T *ptr;
        std::atomic<size_t> ref_cnt;
        int real_id;
        inline inner(T *p) : ptr(p), ref_cnt(1)
        {
            cudaGetDevice(&real_id);
        }
    };
    inner *ptr;

public:
    gpu_ptr_t() : ptr(nullptr) {}
    gpu_ptr_t(T *p) { ptr = new inner(p); }
    gpu_ptr_t(const gpu_ptr_t &r) { *this = r; }
    ~gpu_ptr_t()
    {
        if (ptr && ptr->ref_cnt.fetch_sub(1, std::memory_order_seq_cst) == 1)
        {
            int current_id;
            cudaGetDevice(&current_id);
            if (current_id != ptr->real_id)
            {
                // printf("select GPU id: %d\n", ptr->real_id);
                cudaSetDevice(ptr->real_id);
            }
            cudaFree(ptr->ptr);
            if (current_id != ptr->real_id)
            {
                // printf("select GPU id: %d\n", current_id);
                cudaSetDevice(current_id);
            }
            delete ptr;
        }
    }

    gpu_ptr_t &operator=(const gpu_ptr_t &r)
    {
        if (this != &r)
            (ptr = r.ptr)->ref_cnt.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }
    gpu_ptr_t &operator=(gpu_ptr_t &&r) noexcept
    {
        if (this != &r)
        {
            ptr = r.ptr;
            r.ptr = nullptr;
        }
        return *this;
    }

    inline operator T *() const { return ptr->ptr; }

    // facilitate return by value through FFI, as gpu_ptr_t<T>::by_value.
    struct by_value
    {
        inner *ptr;
    };
    operator by_value() const
    {
        ptr->ref_cnt.fetch_add(1, std::memory_order_relaxed);
        return {ptr};
    }
    gpu_ptr_t(by_value v) { ptr = v.ptr; }
};

// A simple way to allocate a temporary device pointer without having to
// care about freeing it.
template <typename T>
class dev_ptr_t
{
public:
    T *d_ptr;

public:
    // int gpu_id;
    char *name;
    size_t n_elements;
    stream_t &stream;
    bool manual_drop;
    dev_ptr_t(size_t nelems) : d_ptr(nullptr)
    {
        if (nelems)
        {
            size_t n = (nelems + WARP_SZ - 1) & ((size_t)0 - WARP_SZ);
            CUDA_OK(cudaMalloc(&d_ptr, n * sizeof(T)));
        }
    }
    // TODO: `manual_drop` and `alloc` can be reduced to one flag
    dev_ptr_t(size_t nelems, stream_t &s, bool alloc, bool manual_drop = false) : d_ptr(nullptr), stream(s), manual_drop(manual_drop)
    {
        // printf("construct pointer on device: %d, nelems: %d, alloc: %d, manual_drop:%d\n",s.gpu_id, nelems, alloc, manual_drop);
        // manual_drop=false;
        if (nelems)
        {
            n_elements = (nelems + WARP_SZ - 1) & ((size_t)0 - WARP_SZ); // make n multiples of 32

            if (alloc)
            {
                this->alloc();
            }
        }
    }
    dev_ptr_t(const dev_ptr_t &r) = delete; // Copy constructor explicitly deleted
    void set_device_ptr(T *ptr)
    {
        d_ptr = ptr;
    }
    dev_ptr_t &operator=(const dev_ptr_t &r) = delete; // Copy assignment operator explicitly deleted
    ~dev_ptr_t()
    {

        if (d_ptr && !manual_drop)
        {
            // printf("drop device pointer,  on device: %d\n", stream.gpu_id);
            cudaFree((void *)d_ptr);
        }
    }

    inline operator const T *() const { return d_ptr; }
    inline operator T *() const { return d_ptr; }
    inline operator void *() const { return (void *)d_ptr; }
    inline const T &operator[](size_t i) const { return d_ptr[i]; }
    inline T &operator[](size_t i) { return d_ptr[i]; }

private:
    void alloc()
    {
        // should not reach this
        assert(d_ptr == nullptr);

        if (n_elements)
        {
            // printf("Try to alloc %ld elements %ld B\n", n_elements, n_elements * sizeof(T));
            CUDA_OK(cudaMallocAsync(&d_ptr, n_elements * sizeof(T), stream));
        }
    }
};

#endif  // __ZEKNOX_UTIL_GPU_T_CUH__
