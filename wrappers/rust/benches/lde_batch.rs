extern crate criterion;
use std::env;

use criterion::{criterion_group, criterion_main, Criterion};
use zeknox::{
    device::memory::HostOrDeviceSlice, get_number_of_gpus_rs, init_coset_rs, init_twiddle_factors_rs, lde_batch_multi_gpu, types::NTTConfig
};
use plonky2_field::{
    goldilocks_field::GoldilocksField,
    types::{Field, PrimeField64},
};
use rand::random;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn init_twiddle_gpu(lg_domain_size: usize, ngpus: usize) {
    for i in 0..ngpus {
        init_twiddle_factors_rs(i, lg_domain_size);
    }
}

fn init_coset(ngpus: usize) {
    for i in 0..ngpus {
        init_coset_rs(i, 23, GoldilocksField::coset_shift().to_canonical_u64());
    }
}

fn bench_multi_gpu_lde_batch(c: &mut Criterion) {
    let ngpus1: usize = env::var("NUM_OF_GPUS").unwrap().parse::<usize>().unwrap();
    let ngpus2: usize = get_number_of_gpus_rs();
    let ngpus = std::cmp::min(ngpus1, ngpus2);

    assert!(ngpus > 0);

    let log_n_sizes: Vec<usize> = (16..=19).collect();
    let rate_bits = 3;
    let mut group = c.benchmark_group("LDE");

    init_coset(ngpus);

    for log_n_sizes in log_n_sizes {
        let batches = 50;
        let input_domain_size = 1usize << log_n_sizes;
        let output_domain_size = 1usize << (log_n_sizes + rate_bits);

        init_twiddle_gpu(log_n_sizes + rate_bits, ngpus);

        let total_num_input_elements = input_domain_size * batches;
        let total_num_output_elements = output_domain_size * batches;

        let mut cfg_lde = NTTConfig::default();
        cfg_lde.batches = batches as u32;
        cfg_lde.extension_rate_bits = rate_bits as u32;
        cfg_lde.with_coset = true;
        cfg_lde.are_inputs_on_device = false;
        cfg_lde.are_outputs_on_device = true;
        cfg_lde.is_multi_gpu = true;

        let mut gpu_inputs: Vec<u64> = (0..batches)
            .collect::<Vec<usize>>()
            .iter()
            .map(|_| {
                (0..input_domain_size)
                    .map(|_| random_fr())
                    .collect::<Vec<u64>>()
            })
            .flatten()
            .collect();

        let mut device_output_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(0, total_num_output_elements).unwrap();

        group.sample_size(10).bench_function(
            &format!(
                "Multi gpu lde on {} GPUs with lg_n size of 2^{}",
                ngpus,
                log_n_sizes
            ),
            |b| {
                b.iter(|| {
                    lde_batch_multi_gpu(
                        device_output_data.as_mut_ptr(),
                        gpu_inputs.as_mut_ptr(),
                        ngpus,
                        cfg_lde.clone(),
                        log_n_sizes,
                        total_num_input_elements,
                        total_num_output_elements,
                    )
                })
            },
        );

        group.sample_size(10).bench_function(
            &format!(
                "Multi gpu lde on 1 GPU with lg_n size of 2^{}",
                log_n_sizes
            ),
            |b| {
                b.iter(|| {
                    lde_batch_multi_gpu(
                        device_output_data.as_mut_ptr(),
                        gpu_inputs.as_mut_ptr(),
                        1,
                        cfg_lde.clone(),
                        log_n_sizes,
                        total_num_input_elements,
                        total_num_output_elements,
                    )
                })
            },
        );
    }
}

criterion_group!(benches, bench_multi_gpu_lde_batch);
criterion_main!(benches);
