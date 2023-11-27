# description

this repo implements cryptography arithmatic using cuda.

# field to support
- Goldilocks


# compile a single cuda file
```
nvcc -gencode arch=compute_70,code=sm_70 -g -G kernel.cu -o kernel
```

# run a test
```
cargo test --test goldilocks_arithmetic -- test_goldilocks_exp_rust --exact --nocapture
cargo test --test ntt -- test_ntt_gl64_consistency_with_plonky2 --exact --nocapture
```