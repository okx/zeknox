extern crate criterion;
use criterion::{criterion_group, criterion_main, Criterion};
use cryptography_cuda::{device::memory::HostOrDeviceSlice, init_twiddle_factors_rs, naive_transpose_rev_batch, transpose_rev_batch, types::NTTConfig};
use plonky2_field::{fft::fft, goldilocks_field::GoldilocksField, polynomial::PolynomialCoeffs, types::{Field, PrimeField64}};
use rand::random;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const DEFAULT_GPU: usize = 0;
fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn bench_transpose_gpu(c: &mut Criterion){
    let mut group = c.benchmark_group("Transpose");
    let LOG_NTT_SIZES: Vec<usize> = (15..=19).collect();

    for log_ntt_size in LOG_NTT_SIZES {
        let domain_size = 1usize << log_ntt_size;
        let batches = 200;

        let total_elements = domain_size * batches;
        // let scalars: Vec<u64> = (0..(total_elements)).map(|_| random_fr()).collect();
        let mut device_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(total_elements).unwrap();
        let mut device_data2: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(total_elements).unwrap();

        
        
        let mut input: Vec<u64> = (0..batches*domain_size).map(|_| random_fr()).collect();
        let _ = device_data.copy_from_host_offset(input.as_mut_slice(), 0, batches*domain_size);
        

        let mut cfg = NTTConfig::default();
        cfg.are_inputs_on_device = true;
        cfg.are_outputs_on_device = true;
        cfg.are_outputs_transposed = true;
        cfg.batches = batches as u32;

        group.sample_size(20).bench_function(
            &format!("Shared-Mem GPU transpose with n of size 2^{}", log_ntt_size),
            |b| b.iter(||  transpose_rev_batch(
                DEFAULT_GPU,
                device_data.as_mut_ptr(), 
                device_data2.as_mut_ptr(), 
                log_ntt_size, 
                cfg.clone())),
        );
    }
}

fn bench_naive_transpose_gpu(c: &mut Criterion){
    let mut group = c.benchmark_group("Transpose");
    let LOG_NTT_SIZES: Vec<usize> = (15..=19).collect();

    for log_ntt_size in LOG_NTT_SIZES {
        let domain_size = 1usize << log_ntt_size;
        let batches = 200;

        let total_elements = domain_size * batches;
        // let scalars: Vec<u64> = (0..(total_elements)).map(|_| random_fr()).collect();
        let mut device_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(total_elements).unwrap();
        let mut device_data2: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(total_elements).unwrap();

        
        
        let mut input: Vec<u64> = (0..batches*domain_size).map(|_| random_fr()).collect();
        let _ = device_data.copy_from_host_offset(input.as_mut_slice(), 0, batches*domain_size);
        

        let mut cfg = NTTConfig::default();
        cfg.are_inputs_on_device = true;
        cfg.are_outputs_on_device = true;
        cfg.are_outputs_transposed = true;
        cfg.batches = batches as u32;

        group.sample_size(20).bench_function(
            &format!("Naive GPU transpose with n of size 2^{}", log_ntt_size),
            |b| b.iter(||  naive_transpose_rev_batch(
                DEFAULT_GPU,
                device_data.as_mut_ptr(), 
                device_data2.as_mut_ptr(), 
                log_ntt_size, 
                cfg.clone())),
        );
    }
}

pub fn transpose<T: Send + Sync + Copy>(matrix: &Vec<Vec<T>>, len: usize) -> Vec<Vec<T>> {
    (0..len)
        .into_par_iter()
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect()
}

criterion_group!(ntt_benches, bench_transpose_gpu, bench_naive_transpose_gpu);
criterion_main!(ntt_benches);