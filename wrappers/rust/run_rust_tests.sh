#!/bin/sh -e
if [ -z "$NUM_OF_GPUS" ]; then
    echo "Please set NUM_OF_GPUS environment variable!"
    exit 1
fi
echo "Running the tests on ${NUM_OF_GPUS} GPU(s)."

cargo test --features=gl64 --test device -- test_get_number_of_gpus --exact --nocapture
cargo test --features=gl64 --test device -- test_list_devices_info_rs --exact --nocapture
cargo test --features=gl64 --test ntt -- test_intt_gl64_consistency_with_plonky2 --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_batch_gl64_consistency_with_plonky2 --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_batch_intt_batch_gl64_self_consistency --exact --nocapture
cargo test --features=gl64 --test ntt -- test_intt_batch_gl64_consistency_with_plonky2 --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_on_device --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_batch_on_device --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_batch_with_coset --exact --nocapture
cargo test --features=gl64 --test ntt -- test_compute_batched_lde_coeff_form --exact --nocapture
cargo test --features=gl64 --test ntt -- test_compute_batched_lde_value_form --exact --nocapture
cargo test --features=gl64 --test ntt -- test_compute_batched_lde_data_on_device_coeff_form --exact --nocapture
cargo test --features=gl64 --test ntt -- test_transpose_rev --exact --nocapture
cargo test --features=gl64 --test merkle_tree -- --exact --nocapture

if [ $NUM_OF_GPUS -gt 1 ]; then
    cargo test --features=gl64 --test ntt -- test_compute_batched_lde_multi_gpu_data_on_one_gpu_coeff_form --exact --nocapture
    cargo test --features=gl64 --test ntt -- test_compute_batched_lde_multi_gpu_data_on_one_gpu_value_form --exact --nocapture
fi
