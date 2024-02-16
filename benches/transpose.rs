extern crate criterion;
use criterion::{criterion_group, criterion_main, Criterion};
use cryptography_cuda::{device::memory::HostOrDeviceSlice, init_twiddle_factors_rs, ntt_batch, types::NTTConfig};
use plonky2_field::{fft::fft, goldilocks_field::GoldilocksField, polynomial::PolynomialCoeffs, types::{Field, PrimeField64}};
use rand::random;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const DEFAULT_GPU: usize = 0;
fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn bench_ntt_with_transpose_gpu(c: &mut Criterion){
    let mut group = c.benchmark_group("Transpose");
    let LOG_NTT_SIZES: Vec<usize> = (8..=12).collect();

    for log_ntt_size in LOG_NTT_SIZES {
        let domain_size = 1usize << log_ntt_size;
        let batches = 200;

        init_twiddle_factors_rs(0, log_ntt_size);

        let total_elements = domain_size * batches;
        // let scalars: Vec<u64> = (0..(total_elements)).map(|_| random_fr()).collect();

        let mut cpu_buffer: Vec<u64> = Vec::new();
        let mut device_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(total_elements).unwrap();
        let mut device_data2: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(total_elements).unwrap();
        for i in 0..batches{
            let mut input: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
            cpu_buffer.extend(input.iter());

            let _ = device_data.copy_from_host_offset(input.as_mut_slice(), i*domain_size, (i+1)*domain_size);
            let _ = device_data2.copy_from_host_offset(input.clone().as_mut_slice(), i*domain_size, (i+1)*domain_size);
        }

        let mut cfg = NTTConfig::default();
        cfg.are_inputs_on_device = true;
        cfg.are_outputs_on_device = true;
        cfg.are_outputs_transposed = true;
        cfg.batches = batches as u32;

        group.sample_size(20).bench_function(
            &format!("GPU transpose with n of size 2^{}", log_ntt_size),
            |b| b.iter(||  ntt_batch(DEFAULT_GPU, device_data.as_mut_ptr(), log_ntt_size, cfg.clone())),
        );

        group.sample_size(10).bench_function(
            &format!("CPU transpose with n of size 2^{}", log_ntt_size),
            |b| b.iter(||  cpu_ntt(domain_size, &cpu_buffer, batches)),
        );
    }

}

pub fn cpu_ntt(
    domain_size: usize,
    cpu_buffer: &Vec<u64>,
    batches: usize,
)->Vec<u64>{
    let mut cpu_res: Vec<Vec<u64>>  = Vec::new();
    for i in 0..batches{
        let coeffs = cpu_buffer[0..domain_size]
            .iter()
            .map(|i| GoldilocksField::from_canonical_u64(*i))
            .collect::<Vec<GoldilocksField>>();
        let coefficients = PolynomialCoeffs { coeffs};
        let points = fft(coefficients.clone());
        let cpu_results: Vec<u64> = points
            .values
            .iter()
            .map(|x| x.to_canonical_u64())
            .collect();
        cpu_res.push(cpu_results);
    }

    let cpu_results_tranposed = transpose(&cpu_res, domain_size);
    cpu_results_tranposed.into_iter().flatten().collect()
}

pub fn transpose<T: Send + Sync + Copy>(matrix: &Vec<Vec<T>>, len: usize) -> Vec<Vec<T>> {
    (0..len)
        .into_par_iter()
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect()
}

criterion_group!(ntt_benches, bench_ntt_with_transpose_gpu);
criterion_main!(ntt_benches);