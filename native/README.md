# Build

We use ``cmake`` and ``make`` to build the static library and the tests.

```
cd native
rm -rf build
cmake -B build
cmake --build build -j
```

## cmake options

```
option(USE_CUDA "Enable CUDA code (ON by default)" ON)
set(CUDA_ARCH "86" CACHE STRING "CUDA architecture")
option(USE_AVX "Enable AVX acceleration" OFF)
option(USE_AVX512 "Enable AVX512 (and AVX) acceleration" OFF)
option(BUILD_TESTS "Build tests" OFF)
CURVE BN254    # indicate which curve to use, default is GL64; supported values are: BN254, etc
BUILD_MSM      # whether build MSM
G2_ENABLED     # whether enable msm on G2 curve; it takes very long to build; OFF by default
```

## Other flags

``FEATURE_GOLDILOCKS`` - enables Goldilocks field support in NTT (set by default)

``FEATURE_BN254`` - enables BN254 field support in NTT (unset by default)

``EXPOSE_C_INTERFACE`` - expose C interface to Rust (set by default)

``TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE`` - error messages are handled in the native code (set by default)

## CUDA arch

To get the CUDA capability of your GPU, run ``./configure.sh`` in this folder. Then, use the output in ``cmake``. Example output:

```
CUDA capability: 86
Usage: cmake .. -DCUDA_ARCH=86
```

# Tests

```
mkdir build
cd build
cmake .. -DBUILD_TESTS=ON
make -j4
./tests.exe
```

# Debug with VsCode + CUDA-GDB
1. Install [Nsight extension](https://marketplace.visualstudio.com/items?itemName=NVIDIA.nsight-vscode-edition)
2. Build tests in debug mode `./build-debug.sh`
3. Set breakpoints in the code [tests.cu](tests/tests.cu)
4. Enjoy VsCode debugging!

<img width="1775" alt="image" src="https://github.com/user-attachments/assets/39319eae-3de8-49a7-b5cd-5f1fad7d5531">

# Algorithms and Data Structures

## Merkle Tree

The original Plonky2 Merkle Tree data structure is a forest of trees where the hashes of the roots are stored in a vector called ``cap`` and the remaining hashes of all the trees are stored in a vector called ``digests``. By setting the cap height, we control the number of subtrees. For example, when the cap height is 0, we have only one tree.

The internal hashes (including the leaf hashes but excluding the root) are stored in a recursive data structure, such as: ``left recursive output || left child digest || right child digest || right recursive output``.
For example, we have a tree with 4 leaves, L1, L2, L3, L4, and Hi = Hash(Li), i = 1,4. Then, H12 = Hash(H1, H2) and H34 = Hash(H3, H4). The Merkle tree vector in Plonky2 is: ``H1 H2 H12 H23 H2 H3``.

In our version, we replaced this recursive structure with a linear structure. Using the above example, our Merkle tree vector is ``H12 H23 H1 H2 H3 H4``. Using 0-based indexing, the children of a hash at position ``i`` in the vector are at indices ``2*(i+1)`` and ``2*(i+1)+1``, respectively.

## Poseidon

We implemented Plonky2 version of Poseidon hashing in C++ and CUDA. We also modified 0xPolyginHermez [Goldilocks](https://github.com/0xPolygonHermez/goldilocks) Poseidon code (AVX and AVX512) to produce the same results as Plonky2 version. The key difference between Plonky2 (0xPolygonZero) and Goldilocks (0xPolygonHermez) is in the way the internal state (the sponge with 12 elements) is handled across multiple iterations (or permutations): in Plonky2, the old state remains the same and the new 8 elements of the input overwrite the first 8 elements of the state. In Goldilocks, the state is shifted by 8, and then the new 8 elements of the input overwrite the first 8 elements of the state.

## Goldilocks Field Parameters

In a [recent PR](https://github.com/0xPolygonZero/plonky2/pull/1579), 0xPolygonZero updated two Goldilocks field parameters (namely, ``MULTIPLICATIVE_GROUP_GENERATOR`` and ``POWER_OF_TWO_GENERATOR``). Previously, these two parameters have the values ``7`` and ``1753635133440165772``, respectively. Now, they are ``14293326489335486720`` and ``7277203076849721926``, respectively. Our implementation supports both variants (in [scripts/configs/gl64.json](../scripts/configs/gl64.json) and [scripts/configs/gl64_v2.json](../scripts/configs/gl64_v2.json), respectively). By default, we use the initial values. If one wants to use the newer values, please run:

```
cd scripts
python3 gen_field_params.py configs/gl64_v2.json
mkdir -p ../native/build
cd ../native/build
cmake .. -DBUILD_TESTS=ON
make -j4
```

To use the old values, please run:

```
cd scripts
python3 gen_field_params.py configs/gl64.json
mkdir -p ../native/build
cd ../native/build
cmake .. -DBUILD_TESTS=ON
make -j4
```

You can test the consistency with Plonky2 as:

```
cd wrappers/rust
cargo update
./run_rust_tests.sh
```
