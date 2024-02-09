use std::ops::Mul;

use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::packed::PackedField;
use plonky2_field::polynomial::PolynomialValues;
use plonky2_field::{
    fft::fft,
    polynomial::PolynomialCoeffs,
    types::{Field, PrimeField64, Sample},
};
use rand::random;
use rayon::prelude::*;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

static mut g_lehmer64_state: u128 = 0xAAAAAAAAAAAAAAAA;
const multiplier: u128 = 0xda942042e4dd58b5;
fn lehmer64() -> u64 {
    unsafe {
        let lehmer64_state = g_lehmer64_state.mul(multiplier);
        g_lehmer64_state = (lehmer64_state >> 64) as u128;
        g_lehmer64_state as u64
    }
}

fn random_fr_n(n: usize) -> Vec<u64> {
    (0..n).into_iter().map(|_| random_fr()).collect()
}
/// Computes `log_2(n)`, panicking if `n` is not a power of two.
pub fn log2_strict(n: usize) -> usize {
    let res = n.trailing_zeros();
    assert!(n.wrapping_shr(res) == 1, "Not a power of two: {n}");
    // Tell the optimizer about the semantics of `log2_strict`. i.e. it can replace `n` with
    // `1 << res` and vice versa.
    // assume(n == 1 << res);
    res as usize
}

fn main() {
    let rate_bits = 3;
    let degree = 1 << 17;
    let log_n = log2_strict(degree) + rate_bits;
    println!("log_n: {:?}", log_n);
    let domain_size = 1 << log_n;
    let batches = 136;
    let total_size = batches * domain_size;

    let pols = (0..batches)
        .map(|i| {
            let mut cpu_buffer: Vec<GoldilocksField> = (0..degree)
                .map(|_| GoldilocksField::from_canonical_u64(lehmer64()))
                .collect();
            let coeff_pol = PolynomialCoeffs { coeffs: cpu_buffer };
            coeff_pol
        })
        .collect::<Vec<PolynomialCoeffs<GoldilocksField>>>();

    let start = std::time::Instant::now();
    // let _ = pols
    //     .par_iter()
    //     .map(|p| {
    //         assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
    //         p.lde(rate_bits)
    //             .coset_fft_with_options(GoldilocksField::coset_shift(), Some(rate_bits), None)
    //             .values
    //     })
    //     .collect::<Vec<_>>();
    // println!("total non batch : {:?}, ", start_non_batch.elapsed());
    let mut shifts = Vec::<GoldilocksField>::with_capacity(domain_size);
    let mut base = GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR;
    let mut current = GoldilocksField::ONES;
    println!("base: {:?}, current: {:?}", base, current);
    let mut i = 0;
    while i < domain_size {
        shifts.push(current);
        current = current * base;
    }
    println!("total elapsted in shifts: {:?}, ", start.elapsed());
    let _ = pols.par_iter().map(|p| {
        let p_extended = p.lde(rate_bits);
        let modified_poly: PolynomialCoeffs<GoldilocksField> =shifts.iter()
            .zip(&p_extended.coeffs)
            .map(|(&r, &c)| r * c)
            .collect::<Vec<_>>()
            .into();
    }).collect::<Vec<()>>();
    println!("total data transform elapsted : {:?}, ", start.elapsed());
}
