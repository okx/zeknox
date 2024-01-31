use cryptography_cuda::{iNTT, init_twiddle_factors_rust, ntt_batch, types::*, NTT};
use icicle_cuda_runtime::{
    // memory::DeviceSlice,
    device_context::get_default_device_context,
    stream::CudaStream,
};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialValues;
use plonky2_field::{
    fft::fft,
    polynomial::PolynomialCoeffs,
    types::{Field, PrimeField64},
};
use rand::random;
use rayon::prelude::*;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn cpu_fft_concurrent(concurrent_nums: usize, log_ntt_size: usize) {
    let domain_size = 1usize << log_ntt_size;
    let total_size = domain_size * concurrent_nums;

    let mut gpu_buffer: Vec<u64> = (0..total_size).map(|_| random_fr()).collect();
    let start = std::time::Instant::now();
    let ret: Vec<PolynomialValues<GoldilocksField>> = (0..concurrent_nums)
        .into_par_iter()
        .map(|idx| {
            let coeffs = gpu_buffer[idx * domain_size..(idx + 1) * domain_size]
                .iter()
                .map(|i| GoldilocksField::from_canonical_u64(*i))
                .collect::<Vec<GoldilocksField>>();
            let coefficients = PolynomialCoeffs { coeffs };

            fft(coefficients.clone())
        })
        .collect();
    println!("total time spend cpu: {:?}", start.elapsed());
}

fn main() {
    cpu_fft_concurrent(300, 19);
}
