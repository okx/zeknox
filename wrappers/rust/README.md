# run and test
## prerequisites
- rust. To install Rust, run:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Also, set Rust nighly:
```
rustup override set nightly
```

## run Rust tests
```
./run_rust_tests.sh
```

## Build and run Rust benchmarks

For example:

```
cargo bench --bench=gpu_fft
cargo bench --bench=gpu_fft_batch
cargo bench --bench=lde_batch
cargo bench --bench=transpose
```

