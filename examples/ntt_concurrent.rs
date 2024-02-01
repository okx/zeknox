#[cfg(not(feature = "no_cuda"))]
use cryptography_cuda::{iNTT, init_twiddle_factors_rust, ntt_batch, types::*, NTT};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialValues;
use plonky2_field::{
    fft::fft,
    polynomial::PolynomialCoeffs,
    types::{Field, PrimeField64},
};
use rand::random;
use rayon::prelude::*;

const DEFAULT_GPU:usize = 0;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn random_fr_n(n: usize) -> Vec<u64> {
    (0..n).into_iter().map(|_| random_fr()).collect()
}

fn cpu_fft_concurrent(concurrent_nums: usize, log_ntt_size: usize) {
    let domain_size = 1usize << log_ntt_size;
    let total_size = domain_size * concurrent_nums;

    let buffer: Vec<u64> = (0..total_size).map(|_| random_fr()).collect();
    let start = std::time::Instant::now();
    let _: Vec<PolynomialValues<GoldilocksField>> = (0..concurrent_nums)
        .into_par_iter()
        .map(|idx| {
            let coeffs = buffer[idx * domain_size..(idx + 1) * domain_size]
                .iter()
                .map(|i| GoldilocksField::from_canonical_u64(*i))
                .collect::<Vec<GoldilocksField>>();
            let coefficients = PolynomialCoeffs { coeffs };

            fft(coefficients.clone())
        })
        .collect();
    println!(
        "cpu fft of nums: {:?}, log_ntt_size: {:?}, total time spend: {:?}",
        concurrent_nums,
        log_ntt_size,
        start.elapsed()
    );
}

#[cfg(not(feature = "no_cuda"))]
fn gpu_fft_concurrent(device_id: usize, concurrent_nums: usize, log_ntt_size: usize) {
    let domain_size = 1usize << log_ntt_size;
    let gpu_buffers: Vec<Vec<u64>> = (0..concurrent_nums)
        .into_iter()
        .map(|_| random_fr_n(domain_size))
        .collect();

    let start = std::time::Instant::now();
    let _: Vec<_> = gpu_buffers
        .into_par_iter()
        .map(|mut gpu_buffer| NTT(device_id, &mut gpu_buffer, NTTInputOutputOrder::NN))
        .collect();
    println!(
        "gpu fft of nums: {:?}, log_ntt_size: {:?}, total time spend: {:?}",
        concurrent_nums,
        log_ntt_size,
        start.elapsed()
    );
}

#[cfg(not(feature = "no_cuda"))]
fn gpu_fft_concurrent_multiple_devices(
    device_nums: usize,
    concurrent_nums: usize,
    log_ntt_size: usize,
) {
    let domain_size = 1usize << log_ntt_size;
    let gpu_buffers: Vec<Vec<u64>> = (0..concurrent_nums)
        .into_iter()
        .map(|_| random_fr_n(domain_size))
        .collect();

    let start = std::time::Instant::now();
    let _: Vec<_> = gpu_buffers
        .into_par_iter()
        .enumerate()
        .map(|(idx, mut gpu_buffer)| {
            NTT(idx % device_nums, &mut gpu_buffer, NTTInputOutputOrder::NN)
        })
        .collect();
    println!(
        "gpu fft of nums: {:?}, log_ntt_size: {:?}, total time spend: {:?}",
        concurrent_nums,
        log_ntt_size,
        start.elapsed()
    );
}

#[cfg(not(feature = "no_cuda"))]
fn gpu_fft_batch(device_id: usize, batches: usize, log_ntt_size: usize) {
    let domain_size = 1usize << log_ntt_size;

    let mut gpu_buffer: Vec<u64> = (0..domain_size * batches).map(|_| random_fr()).collect();
    let start = std::time::Instant::now();
    ntt_batch(
        device_id,
        &mut gpu_buffer,
        NTTInputOutputOrder::NN,
        batches as u32,
        log_ntt_size,
    );
    println!(
        "gpu fft batch of nums: {:?}, log_ntt_size: {:?}, total time spend: {:?}",
        batches,
        log_ntt_size,
        start.elapsed()
    );
}

#[cfg(not(feature = "no_cuda"))]
fn gpu_fft_batch_multiple_devices(device_nums: usize, batches: usize, log_ntt_size: usize) {
    let domain_size = 1usize << log_ntt_size;
    let per_device_batch = batches.div_ceil(device_nums);
    println!("per_device_batch: {:?}", per_device_batch);

    let mut gpu_buffers: Vec<Vec<u64>> = (0..device_nums)
        .into_iter()
        .map(|_| random_fr_n(domain_size * per_device_batch))
        .collect();

    let mut cpu_buffers = gpu_buffers.clone();
    // println!("input: {:?}", gpu_buffers);
    let start = std::time::Instant::now();
    let _: Vec<_> = gpu_buffers
        .par_iter_mut()
        .enumerate()
        .map(|(device_id, mut input)| {
            // println!("invoking gpu: {:?}, input: {:?}", device_id, input);
            ntt_batch(
                device_id,
                &mut input,
                NTTInputOutputOrder::NN,
                per_device_batch as u32,
                log_ntt_size,
            );
            // println!("invoking gpu: {:?}, output: {:?}", device_id, input);

        })
        .collect();

    // println!("after ntt: {:?}", gpu_buffers);

    // println!("cpu input: {:?}", cpu_buffers[1]);
    let coeffs = cpu_buffers[0][0..domain_size]
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let coefficients = PolynomialCoeffs { coeffs };

    let ret = fft(coefficients.clone());
    // println!("cpu ret: {:?}", ret);

    println!(
        "gpu fft batch of nums: {:?}, log_ntt_size: {:?}, total time spend: {:?}",
        batches,
        log_ntt_size,
        start.elapsed()
    );
}

fn main() {
    let nums = 200;
    let log_ntt_size = 23;

    let num_devices = 4;
    cpu_fft_concurrent(nums, log_ntt_size);

    #[cfg(not(feature = "no_cuda"))]
    let mut device_id = 0;
    while (device_id < num_devices) {
        // println!("init for device: {:?}", device_id);
        init_twiddle_factors_rust(device_id, log_ntt_size);
        device_id = device_id + 1;
    }

    // #[cfg(not(feature = "no_cuda"))]
    // gpu_fft_concurrent(DEFAULT_GPU, nums, log_ntt_size);
    // #[cfg(not(feature = "no_cuda"))]
    // gpu_fft_batch(DEFAULT_GPU, nums, log_ntt_size);

    // #[cfg(not(feature = "no_cuda"))]
    // gpu_fft_concurrent_multiple_devices(num_devices, nums, log_ntt_size);
    #[cfg(not(feature = "no_cuda"))]
    gpu_fft_batch_multiple_devices(num_devices, nums, log_ntt_size);
}
