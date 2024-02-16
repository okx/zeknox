#!/bin/sh -e
cargo test --features=gl64 --test gpu -- test_get_number_of_gpus --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_intt_gl64_self_consistency --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_gl64_consistency_with_plonky2 --exact --nocapture
cargo test --features=gl64 --test ntt -- test_intt_gl64_consistency_with_plonky2 --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_batch_gl64_consistency_with_plonky2 --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_batch_intt_batch_gl64_self_consistency --exact --nocapture
cargo test --features=gl64 --test ntt -- test_intt_batch_gl64_consistency_with_plonky2 --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_on_device --exact --nocapture
cargo test --features=gl64 --test ntt -- test_ntt_batch_on_device --exact --nocapture