# description

this repo implements cryptography arithmatic using cuda.

# algorithm support
- NTT
much of the code has been taken from https://github.com/supranational/sppark. only a small portion of the code is actually requied to our project; also, as a purpose of learning, we implemented the algorithm from scratch, but it is mainly based on the work of sppark.

# field to support
- Goldilocks

# run integration test
**note** a gpu is required to run below tests
```
cargo test --test goldilocks_arithmetic -- test_goldilocks_exp_rust --exact --nocapture
cargo test --test ntt -- test_ntt_gl64_consistency_with_plonky2 --exact --nocapture
```