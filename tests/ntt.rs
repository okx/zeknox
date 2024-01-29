use cryptography_cuda::{iNTT, ntt_batch, types::*, NTT};
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

#[test]
fn test_ntt_gl64_self_consistency() {
    for lg_domain_size in 1..28 {
        let domain_size = 1usize << lg_domain_size;

        let v: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

        let mut gpu_buffer = v.clone();

        NTT(DEFAULT_GPU, &mut gpu_buffer, NTTInputOutputOrder::NN);

        iNTT(DEFAULT_GPU, &mut gpu_buffer, NTTInputOutputOrder::NN);

        assert_eq!(v, gpu_buffer);
    }
}

#[test]
fn test_ntt_gl64_consistency_with_plonky2() {
    for lg_domain_size in 1..20 {
        let domain_size = 1usize << lg_domain_size;

        let v: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
        let mut gpu_buffer = v.clone();
        NTT(DEFAULT_GPU, &mut gpu_buffer, NTTInputOutputOrder::NN);

        let plonky2_ntt_input = v.clone();
        let coeffs = plonky2_ntt_input
            .iter()
            .map(|i| GoldilocksField::from_canonical_u64(*i))
            .collect::<Vec<GoldilocksField>>();
        let coefficients = PolynomialCoeffs { coeffs };
        let points = fft(coefficients.clone());
        let cpu_results: Vec<u64> = points.values.iter().map(|x| x.to_canonical_u64()).collect();

        assert_eq!(gpu_buffer, cpu_results);
    }
}

#[test]
fn test_ntt_batch_gl64_consistency_with_plonky2() {
    let lg_domain_size: u32 = 19;
    let domain_size = 1usize << lg_domain_size;

    let v1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let v2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut gpu_buffer = v1.clone();
    gpu_buffer.extend(v2.iter());
    ntt_batch(
        DEFAULT_GPU,
        &mut gpu_buffer,
        NTTInputOutputOrder::NN,
        2,
        lg_domain_size,
    );

    let plonky2_ntt_input1 = v1.clone();
    let coeffs1 = plonky2_ntt_input1
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let coefficients1 = PolynomialCoeffs { coeffs: coeffs1 };
    let points1 = fft(coefficients1.clone());
    let cpu_results1: Vec<u64> = points1
        .values
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    assert_eq!(gpu_buffer[0..1 << lg_domain_size], cpu_results1);

    let plonky2_ntt_input2 = v2.clone();
    let coeffs2 = plonky2_ntt_input2
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let coefficients2 = PolynomialCoeffs { coeffs: coeffs2 };
    let points2 = fft(coefficients2.clone());
    let cpu_results2: Vec<u64> = points2
        .values
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    assert_eq!(
        gpu_buffer[1 << lg_domain_size..(1 << lg_domain_size) * 2],
        cpu_results2
    );
}
