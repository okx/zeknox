#[cfg(not(feature = "no_cuda"))]
use cryptography_cuda::{init_twiddle_factors_rs, ntt_batch, intt_batch, types::*};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialValues;
use plonky2_field::{
    fft::{fft, ifft},
    polynomial::PolynomialCoeffs,
    types::{Field, PrimeField64},
};
use rand::random;
use rayon::prelude::*;

const DEFAULT_GPU:usize = 0;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr & 0xfffffffeffffffff
}

fn random_fr_n(n: usize) -> Vec<u64> {
    (0..n).into_par_iter().map(|_| random_fr()).collect()
}

fn cpu_fft_concurrent(domain_size: usize, buffers: &[Vec<u64>], inverse: bool) -> Vec<Vec<u64>> {
    let start = std::time::Instant::now();

    let result: Vec<_> = buffers.par_iter().map(|buffer|{
        buffer.par_chunks(domain_size).flat_map(|chunk|{
            let coeffs = chunk
                .iter()
                .map(|i| GoldilocksField::from_canonical_u64(*i))
                .collect::<Vec<GoldilocksField>>();
            if inverse {
                let values = PolynomialValues { values: coeffs };
                ifft(values).coeffs.iter().map(|x| x.to_canonical_u64()).collect::<Vec<_>>()
            } else {
                let coefficients = PolynomialCoeffs { coeffs };
                fft(coefficients).values.iter().map(|x| x.to_canonical_u64()).collect::<Vec<_>>()
            }

        }).collect::<Vec<_>>()
    }).collect();
    println!(
        "cpu fft total time spend: {:?}",
        start.elapsed()
    );

    result
}

#[cfg(not(feature = "no_cuda"))]
fn gpu_fft_batch_multiple_devices(log_ntt_size: usize, buffers: &mut [Vec<u64>], inverse: bool) {
    let start = std::time::Instant::now();
    let _: Vec<_> = buffers.par_iter_mut().enumerate().map(|(device_id, buffer)|{
        let batches = (buffer.len()/(1<<log_ntt_size)) as u32;
        if inverse {
            intt_batch(
                device_id,
                buffer,
                NTTInputOutputOrder::NN,
                batches,
                log_ntt_size,
            );
        } else {
            ntt_batch(
                device_id,
                buffer,
                NTTInputOutputOrder::NN,
                batches,
                log_ntt_size,
            );
        }

    }).collect();

    println!(
        "gpu fft batch of nums: {:?}, log_ntt_size: {:?}, total time spend: {:?}",
        buffers.len(),
        log_ntt_size,
        start.elapsed()
    );
}

fn main() {
    let num_devices = 4;
    let total_cols = 201;
    let log_ntt_size = 23;
    let domain_size = 1<<log_ntt_size;
    let inverse = true;

    let batch_size = total_cols / num_devices;
    let last_batch_size = batch_size + total_cols%num_devices;

    let mut gpu_buffers: Vec<Vec<u64>> = (0..num_devices)
        .into_par_iter()
        .map(|i| {
            let size = if i == num_devices -1 {
                last_batch_size * domain_size
            } else {
                batch_size * domain_size
            };
            random_fr_n(size)
        })
        .collect();

    let cpu_result = cpu_fft_concurrent(domain_size, &gpu_buffers, inverse);

    #[cfg(not(feature = "no_cuda"))]
        let _ : Vec<_> = (0..num_devices).into_par_iter().map(|device_id|{
        init_twiddle_factors_rs(device_id, log_ntt_size);
        //init_twiddle_factors_rs(device_id, log_ntt_size+1);
    }).collect();


    #[cfg(not(feature = "no_cuda"))]
    gpu_fft_batch_multiple_devices(log_ntt_size, &mut gpu_buffers, inverse);

    if cpu_result == gpu_buffers {
        println!("success")
    } else {
        println!("failed")
    }
    //assert_eq!(cpu_result, gpu_buffers);
}
