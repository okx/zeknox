// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_UTIL_ALL_GPUS_CPP__
#define __SPPARK_UTIL_ALL_GPUS_CPP__

#include "gpu_t.cuh"
#include <cuda_runtime.h>

class gpus_t {
    std::vector<const gpu_t*> gpus;
public:
    gpus_t();
    ~gpus_t();
    static const auto& all()
    {
        static gpus_t all_gpus; // all_gpus will only be initated onetime throuout the lifetime of the program, as it is static
        return all_gpus.gpus;
    }
};

const gpu_t& select_gpu(int id);

const cudaDeviceProp& gpu_props(int id);

size_t ngpus();

const std::vector<const gpu_t*>& all_gpus();

extern "C" bool cuda_available();

int _ConvertSMVer2Cores(int major, int minor);

void list_all_gpus_prop();

#endif