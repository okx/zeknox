# Run and Test

## Prerequisites
- **rust**: to install Rust, run:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Also, set Rust nighly:
```
rustup override set nightly
```

- **libzeknox.a**: please download ``gl64-86-89-90-libzeknox-p2.a`` from the Github [release page](https://github.com/okx/zeknox/releases) for **zeknox_p2**, then run:
```
sudo cp gl64-86-89-90-libzeknox-p2.a /usr/local/lib/libzeknox.a
sudo cp libblst.a /usr/local/lib/
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

