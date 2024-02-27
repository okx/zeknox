extern crate criterion;
use criterion::{criterion_group, criterion_main, Criterion};
use cryptography_cuda::{device::memory::HostOrDeviceSlice, init_twiddle_factors_rs, ntt_batch, types::NTTConfig};
use plonky2_field::{fft::fft, goldilocks_field::GoldilocksField, polynomial::PolynomialCoeffs, types::{Field, PrimeField64}};
use rand::random;

const DEFAULT_GPU: usize = 0;
fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn bench_gpu_ntt_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("NTT");
    let LOG_NTT_SIZES: Vec<usize> = (10..=14).collect();

    for log_ntt_size in LOG_NTT_SIZES {
        let domain_size = 1usize << log_ntt_size;
        let batches = 200;

        init_twiddle_factors_rs(0, log_ntt_size);

        let total_elements = domain_size * batches;
        // let scalars: Vec<u64> = (0..(total_elements)).map(|_| random_fr()).collect();

        let mut cpu_buffer: Vec<u64> = Vec::new();
        let mut device_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU as i32, total_elements).unwrap();
        for i in 0..batches{
            let mut input: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
            cpu_buffer.extend(input.iter());

            let _ = device_data.copy_from_host_offset(input.as_mut_slice(), i*domain_size, (i+1)*domain_size);
        }

        let mut cfg = NTTConfig::default();
        cfg.are_inputs_on_device = true;
        cfg.are_outputs_on_device = true;
        cfg.batches = batches as u32;

        group.sample_size(20).bench_function(
            &format!("GPU FFT with n of size 2^{}", log_ntt_size),
            |b| b.iter(||  ntt_batch(DEFAULT_GPU, device_data.as_mut_ptr(), log_ntt_size, cfg.clone())),
        );

        group.sample_size(10).bench_function(
            &format!("CPU FFT with n of size 2^{}", log_ntt_size),
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
    for _i in 0..batches{
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

    cpu_res.into_iter().flatten().collect()
}


criterion_group!(benches, bench_gpu_ntt_batch);
criterion_main!(benches);
