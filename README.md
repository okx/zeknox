# ZEKNOX - ZEro KNOwledge Xcelerated

This repo implements primitives used in Zero Knowledge Proofs accelerated with CUDA for Nvidia GPUs. In particular, this repo is used by OKX Plonky2 [fork](https://github.com/okx/plonky2).

The following primitives are implemented:

- Poseidon hashing over Goldilocks elements (in C/C++ and CUDA) - see [native/poseidon](native/poseidon).
- Poseidon hashing over BN254 (or BN128) elements (in C/C++ and CUDA) - see [native/poseidon](native/poseidon).
- Poseidon2 hashing over Goldilocks elements (in C/C++ and CUDA) - see [native/poseidon2](native/poseidon2).
- Keccak hashing over Goldilocks elements (in C/C++ and CUDA) - see [native/keccak](native/keccak).
- Monolith hashing over Goldilocks elements (in C/C++ and CUDA) - see [native/monolith](native/monolith).
- Merkle Tree building (Plonky2 version) using any of the above hashing methods - see [native/merkle](native/merkle).
- NTT (including LDE and transpose) over Goldilocks field - see [native/ntt](native/ntt).
- MSM over BN254 - see [native/msm](native/msm).

# Building and Testing
## Prerequisites

- git submodules
```
git submodule update --init
```

- gcc/g++, make, gtest. To install these in Ubuntu, run:
```
sudo apt install -y gcc g++ clang make cmake libc++-dev libgtest-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

- nvcc > 12.0 (CUDA Tookit). For the latest CUDA toolkit, please see [https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit). After installing CUDA, set NVCC environment variable:
```
export NVCC=/usr/local/cuda/bin/nvcc
```

For example, to install CUDA 12.6:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
sudo apt-get install -y cuda-drivers
```

## Build CUDA library
```
mkdir -p native/build
cd native/build
cmake ..
make -j
```

Note: the steps above build the library with Goldilocks support, without MSM.

## Build and run CUDA tests
Note: this requires an Nvidia GPU.

```
mkdir -p native/build
cd native/build
cmake .. -DBUILD_TESTS=ON
make -j
./tests.exe
```

## Install the library
After building, run:
```
sudo make install
```

Note: you also need to have ``libblst.a`` built before running ``sudo make install``:
```
cd native/depends/blst
./build.sh
```

## Curve Parameters Generation
The curve/field parameters are generated by a template:
```
cd scripts
python3 new_curve_script.py configs/${field}.json
```

For Goldilocks field (see [details](native/README.md#goldilocks-field-parameters)), generate the parameters as:
```
cd scripts
python gen_field_params.py configs/gl64.json
# or (for compatibility with OxPolygonZero Plonky2)
python gen_field_params.py configs/gl64_v2.json
```
Then re-build the CUDA library as described above.

## FAQ

Please see our [FAQ page](FAQ.md).

# Examples and Benchmarks
Next, we present three examples of integrating this library to speedup ZK primitives and applications.

## E1. Plonky2

In Plonky2, we offload Merkle Tree building (with hashing) and Low Degree Extention (LDE) with Number Theoretic Transform (NTT) to a GPU (or multiple GPUs) (more details [here](native/README.md#algorithms-and-data-structures)). Next, we list the steps needed to build Plonky2 with GPU acceleration:

```
git clone https://github.com/okx/plonky2.git
cd plonky2
git checkout dev
rustup update
rustup override set nightly-x86_64-unknown-linux-gnu
cargo build --release --features=cuda
```

Next, we show benchmarking results for Merkle Tree building with Poseidon, Poseidon2, and Poseidon over BN254, comparing the CPU-only with the CPU+GPU execution. To run these benchmarks, simply:
```
git clone https://github.com/okx/plonky2.git
cd plonky2
git checkout dev
cd plonky2
cargo bench --bench=merkle
cargo bench --bench=merkle --features=cuda
```

The following results are from an GCP ``g2-standard-32`` instance with 32 vCPU of Intel Xeon type and one NVIDIA L4 GPU.

Hash | Leaves | CPU-only |	CPU+GPU | Speedup
--- | --- | --- | --- | ---
Poseidon | 8192	    | 26.8 ms  | 11.5 ms  | 2.3 X
Poseidon | 16384	| 53.4 ms  | 20.2 ms  | 2.6 X
Poseidon | 32768	| 111.1 ms | 44.8 ms  | 2.5 X
Poseidon2 | 8192    | 30.9 ms  | 8.4 ms   | 3.7 X
Poseidon2 | 16384   | 61.4 ms  | 16.6 ms  | 3.7 X
Poseidon2 | 32768   | 127.0 ms | 39.2 ms  | 3.2 X
Poseidon BN128 | 8192  | 404.7 ms  | 73.5 ms  | 5.5 X
Poseidon BN128 | 16384 | 809.4 ms  | 124.0 ms | 6.5 X
Poseidon BN128 | 32768 | 1618.4 ms | 239.9 ms | 6.7 X

Next, we show benchmarking results for LDE + MT building with Poseidon, comparing the CPU-only with the CPU+GPU execution. To run these benchmarks, simply:
```
git clone https://github.com/okx/plonky2.git
cd plonky2
git checkout dev
cd plonky2
cargo bench --bench=lde
cargo bench --bench=lde --features=cuda
```

LDE size (log) | CPU-only | CPU+GPU | Speedup
--- | --- | --- | ---
  13 |  6.5 ms | 3.1 ms | 2.1 X
  14 | 11.6 ms | 4.2 ms | 2.8 X
  15 | 22.0 ms | 6.0 ms | 3.7 X


## E2. zk_evm (Type 1 ZK EVM from 0xPolygonZero)

```
sudo apt install -y librust-openssl-dev bc
git clone https://github.com/okx/zk_evm.git
cd zk_evm
git checkout dev
cd scripts
./prove_stdio.sh ../artifacts/witness_b3_b6.json
./prove_stdio.sh ../artifacts/witness_b19807080.json
```

The following results are from an GCP g2-standard-32 instance with 32 vCPU of Intel Xeon type and one NVIDIA L4 GPU.

Input | CPU-only |	CPU+GPU | Speedup
--- | --- | --- | ---
witness_b3_b6.json     | 193.7 ms |  111.1 ms | 1.74 X
witness_b19807080.json | 294.6 ms |	 174.5 ms | 1.69 X

## E3. Proof-of-Reserves-v2 (OKX)

```
git clone https://github.com/okx/proof-of-reserves-v2.git
cd proof-of-reserves-v2.git
git checkout dev-dumi-v0.1.0
```
then follow the steps presented in the [README](https://github.com/okx/proof-of-reserves-v2/blob/dev-dumi-v0.1.0/README.md).

The following results are from an GCP g2-standard-32 instance with 32 vCPU of Intel Xeon type and one NVIDIA L4 GPU, for proving 1,310,720 accounts.

CPU-only |	CPU+GPU | Speedup
--- | --- | ---
 2834 s | 1377 s | 2.06 X

# License

Apache License, Version 2.0 [LICENSE](LICENSE)
