# Build

We use ``cmake`` and ``make`` to build the static library and the tests.

```
mkdir build
cd build
cmake ..
make -j4
```

## cmake options

```
option(USE_CUDA "Enable CUDA code (ON by default)" ON)
set(CUDA_ARCH "86" CACHE STRING "CUDA architecture")
option(USE_AVX "Enable AVX acceleration" OFF)
option(USE_AVX512 "Enable AVX512 (and AVX) acceleration" OFF)
option(BUILD_TESTS "Build tests" OFF)
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
