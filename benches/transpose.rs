#[allow(dead_code)]

extern crate criterion;
use criterion::{criterion_group, criterion_main, Criterion};
#[cfg(not(feature = "no_cuda"))]
use cryptography_cuda::{
    device::memory::HostOrDeviceSlice, 
    naive_transpose_rev_batch, 
    transpose_rev_batch, 
    types::TransposeConfig
};

use rand::random;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

#[cfg(not(feature = "no_cuda"))]
fn bench_transpose_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transpose");
    let LOG_NTT_SIZES: Vec<usize> = (20..=20).collect();

    for log_ntt_size in LOG_NTT_SIZES {
        let domain_size = 1usize << log_ntt_size;
        let batches = 200;

        let total_elements = domain_size * batches;
        // let scalars: Vec<u64> = (0..(total_elements)).map(|_| random_fr()).collect();
        let mut device_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU.try_into().unwrap(), total_elements).unwrap();
        let mut device_data2: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU.try_into().unwrap(), total_elements).unwrap();

        let mut input: Vec<u64> = (0..batches * domain_size).map(|_| random_fr()).collect();
        let _ = device_data.copy_from_host_offset(input.as_mut_slice(), 0, batches * domain_size);

        let mut cfg = TransposeConfig::default();
        cfg.are_inputs_on_device = true;
        cfg.are_outputs_on_device = true;
        cfg.batches = batches as u32;

        group.sample_size(10).bench_function(
            &format!("Shared-Mem GPU transpose with n of size 2^{}", log_ntt_size),
            |b| {
                b.iter(|| {
                    transpose_rev_batch(
                        DEFAULT_GPU.try_into().unwrap(),
                        device_data.as_mut_ptr(),
                        device_data2.as_mut_ptr(),
                        log_ntt_size,
                        cfg.clone(),
                    )
                })
            },
        );
    }
}

#[cfg(not(feature = "no_cuda"))]
fn bench_naive_transpose_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transpose");
    let LOG_NTT_SIZES: Vec<usize> = (20..=20).collect();

    for log_ntt_size in LOG_NTT_SIZES {
        let domain_size = 1usize << log_ntt_size;
        let batches = 200;

        let total_elements = domain_size * batches;
        // let scalars: Vec<u64> = (0..(total_elements)).map(|_| random_fr()).collect();
        let mut device_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU.try_into().unwrap(), total_elements).unwrap();
        let mut device_data2: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU.try_into().unwrap(), total_elements).unwrap();

        let mut input: Vec<u64> = (0..batches * domain_size).map(|_| random_fr()).collect();
        let _ = device_data.copy_from_host_offset(input.as_mut_slice(), 0, batches * domain_size);

        let mut cfg = TransposeConfig::default();
        cfg.are_inputs_on_device = true;
        cfg.are_outputs_on_device = true;
        cfg.batches = batches as u32;

        group.sample_size(10).bench_function(
            &format!("Naive GPU transpose with n of size 2^{}", log_ntt_size),
            |b| {
                b.iter(|| {
                    naive_transpose_rev_batch(
                        DEFAULT_GPU.try_into().unwrap(),
                        device_data.as_mut_ptr(),
                        device_data2.as_mut_ptr(),
                        log_ntt_size,
                        cfg.clone(),
                    )
                })
            },
        );
    }
}

fn bench_transpose_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Transpose");
    let log_ntt_sizes: Vec<usize> = (19..=21).collect();

    for log_ntt_size in log_ntt_sizes {
        let domain_size = 1usize << log_ntt_size;
        let batches = 200;

        // let scalars: Vec<u64> = (0..(total_elements)).map(|_| random_fr()).collect();

        let mut cpu_buffer: Vec<Vec<u64>> = Vec::new();

        let input: Vec<u64> = (0..batches * domain_size).map(|_| random_fr()).collect();
        cpu_buffer.push(input);

        group.sample_size(10).bench_function(
            &format!("CPU transpose with n of size 2^{}", log_ntt_size),
            |b| b.iter(||  transpose(&cpu_buffer, domain_size))
        );
    }
}

pub fn transpose<T: Send + Sync + Copy>(matrix: &Vec<Vec<T>>, len: usize) -> Vec<Vec<T>> {
    (0..len)
        .into_par_iter()
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect()
}

#[cfg(not(feature = "no_cuda"))]
criterion_group!(
    benches,
    bench_transpose_gpu,
    bench_naive_transpose_gpu,
    bench_transpose_cpu
);
#[cfg(feature = "no_cuda")]
criterion_group!(
    benches,
    bench_transpose_cpu
);
criterion_main!(benches);
