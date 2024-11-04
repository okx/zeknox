// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
#[allow(unused_imports)]
extern crate criterion;
use cryptography_cuda::{
    device::memory::HostOrDeviceSlice, init_twiddle_factors_rs, ntt_batch, types::NTTConfig,
};
use rand::random;

const DEFAULT_GPU: usize = 0;
fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn ntt_batch_with_lg(batches: usize, log_ntt_size: usize) {
    let domain_size = 1usize << log_ntt_size;
    let total_elements = domain_size * batches;

    let start = std::time::Instant::now();

    let mut device_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU as i32, total_elements).unwrap();

    for i in 0..batches {
        let mut input: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

        let _ = device_data.copy_from_host_offset(
            input.as_mut_slice(),
            i * domain_size,
            (i + 1) * domain_size,
        );
    }

    let mut cfg = NTTConfig::default();
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    cfg.batches = batches as u32;

    ntt_batch(
        DEFAULT_GPU,
        device_data.as_mut_ptr(),
        log_ntt_size,
        cfg.clone(),
    );

    println!("total time spend: {:?}", start.elapsed());
}

fn main() {
    let start = std::time::Instant::now();
    println!("total time spend init context: {:?}", start.elapsed());
    let log_ntt_size = 19;
    init_twiddle_factors_rs(0, log_ntt_size);

    let batches = 10;

    ntt_batch_with_lg(batches, log_ntt_size);
}
