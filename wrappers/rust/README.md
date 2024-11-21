# Run and Test

## Prerequisites
- rust. To install Rust, run:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Also, set Rust nighly:
```
rustup override set nightly
```

## Rust Tests
First, set the number of GPUs you want to run the tests on. You need at least one GPU in your test system.
```
export NUM_OF_GPUS=1
./run_rust_tests.sh
```

## Build and Run Benchmarks

For example:
```
export NUM_OF_GPUS=1
cargo bench --bench=gpu_fft_batch
cargo bench --bench=lde_batch
cargo bench --bench=transpose
```

