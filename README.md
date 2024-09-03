# Description

This repo implements primitives used in Zero Knowledge Proofs accelerated with CUDA for Nvidia GPUs. In particular, this repo is used by OKX Plonky2 [fork](https://github.com/okx/plonky2).

The following primitives are implemented:

- Poseidon hashing over Goldilocks elements (in C/C++ and CUDA) - see [cuda/poseidon](cuda/poseidon).
- Poseidon hashing over BN254 (or BN128) elements (in C/C++ and CUDA) - see [cuda/poseidon](cuda/poseidon).
- Poseidon2 hashing over Goldilocks elements (in C/C++ and CUDA) - see [cuda/poseidon2](cuda/poseidon2).
- Keccak hashing over Goldilocks elements (in C/C++ and CUDA) - see [cuda/keccak](cuda/keccak).
- Monolith hashing over Goldilocks elements (in C/C++ and CUDA) - see [cuda/monolith](cuda/monolith).
- Merkle Tree building (Plonky2 version) using any of the above hashing methods - see [cuda/merkle](cuda/merkle).
- NTT (including LDE and transpose) over Goldilocks elements - see [cuda/ntt](cuda/ntt).
- MSM over Goldilocks elements - see [cuda/msm](cuda/msm).

## Algorithms
- **Goldilocks**: the CUDA implementation for Goldilocks field operations is taken from SupraNational's [sppark](https://github.com/supranational/sppark) project.

- **NTT**: much of the code has been taken from SupraNational's [sppark](https://github.com/supranational/sppark). Only a small portion of the code is actually requied for Plonky2 integration. As a purpose of learning, we implemented the algorithm from scratch, but it is mainly based on the work of SupraNational.

- **MSM**: based on pippenger algorithm, supporting both G1 and G2 curve. Currently, if the input points contains zero points, the kernel will return zero. Therefore, the user should set the corresponding scalar to zero to make the result valid.

# Building and Testing
## Prerequisites
- nvcc > 12.0 (CUDA Tookit). For the latest CUDA toolkit, please see [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). After installing CUDA, set NVCC environment variable:
```
export NVCC=/usr/local/cuda/bin/nvcc
```

- gcc/g++, make, gtest. To install these in Ubuntu, run:
```
sudo apt install gcc g++ make libgtest-dev
```

- rust. To install Rust, run:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Also, set Rust nighly:
```
rustup override set nightly
```

## Build CUDA library
```
cd cuda
make lib
```

## Build and run CUDA tests
Note: this requires an Nvidia GPU.

```
cd cuda
make tests.exe
./tests.exe
```

## Build and run Rust tests
```
./run_rust_tests.sh
```

## Build and run Rust benchmarks

For example:

```
cargo bench --bench gpu_fft
```



# Native (C/C++) usage
## Build
```
git submodule init
git submodule update
./build_gmp.sh host
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./package -DCURVE=${BN254,GOLDILOCKS} -DG2_ENABLED=OFF
make -j4 VERBOSE=1 & make install
```

to enable G2
```
-DG2_ENABLED=ON
```

## Test
- run test
```
export LD_LIBRARY_PATH=package/lib/ # if building a shared lib
./package/bin/test_bn128 --gtest_filter=*xxx*
```

## Remarks
- current cuda build time for G2_ENABLED is very slow. therefore G2_ENABLED is not enabled by default.

# License

Apache License, Version 2.0 [LICENSE](LICENSE)