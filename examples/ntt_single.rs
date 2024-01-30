use cryptography_cuda::{iNTT, init_twiddle_factors_rust, ntt_batch, types::*, NTT};
use icicle_cuda_runtime::{
    // memory::DeviceSlice,
    device_context::get_default_device_context,
    stream::CudaStream,
};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::{
    fft::fft,
    polynomial::PolynomialCoeffs,
    types::{Field, PrimeField64},
};
use rand::random;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

const DEFAULT_GPU: usize = 0;

fn cpu_fft(log_ntt_size: usize) {
    let domain_size = 1usize << log_ntt_size;

    let mut gpu_buffer: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let coeffs = gpu_buffer
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let coefficients = PolynomialCoeffs { coeffs };
    let start = std::time::Instant::now();
    fft(coefficients.clone());
    println!("total time spend cpu: {:?}", start.elapsed());
}

fn gpu_fft(log_ntt_size: usize) {
    let domain_size = 1usize << log_ntt_size;

    let v: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let mut gpu_buffer = v.clone();
    let start = std::time::Instant::now();
    NTT(DEFAULT_GPU, &mut gpu_buffer, NTTInputOutputOrder::NN);
    println!("total time spend gpu: {:?}", start.elapsed());
}

fn main() {
    let start = std::time::Instant::now();
    let stream = CudaStream::create().unwrap();
    println!("total time spend init context: {:?}", start.elapsed());
    let log_ntt_size = 19;
    init_twiddle_factors_rust(0, log_ntt_size);

    gpu_fft(2);

    println!("after warm up");
    let mut i = 0;
    while (i < 5) {
        gpu_fft(log_ntt_size);
        i = i + 1;
    }
}
