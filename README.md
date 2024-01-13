# description

this repo implements cryptography arithmatic using cuda.

# algorithm support
- NTT
much of the code has been taken from https://github.com/supranational/sppark. only a small portion of the code is actually requied to our project; also, as a purpose of learning, we implemented the algorithm from scratch, but it is mainly based on the work of sppark.

# field support
- Goldilocks

# rust usage
**note** a gpu is required to run below tests
## run integration test
```
cargo test --test goldilocks_arithmetic -- test_goldilocks_exp_rust --exact --nocapture
cargo test --features=bn254 --test bn128_arithmetic -- test_bn128_add_rust --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_gl64_consistency_with_plonky2 --exact --nocapture
```
## run benchmarks
```
cargo bench --bench gpu_fft
```

## If got below issue
- no nvcc found
```
export NVCC=/usr/local/cuda/bin/nvcc
```


# cpp usage
## build
```
git submodule init
git submodule update
./build_gmp.sh host
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./package
make -j4 & make install
```

## run
- run example
```
export LD_LIBRARY_PATH=package/lib/
./package/bin/example
./package/bin/test_bn128 --gtest_filter=*xxx*
```

## factor, rt(ms)
2, 1862
4, 1895.108
6, 1911.426
10, 1794.383

10615314
1048576