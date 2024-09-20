#[allow(dead_code)]
#[allow(unused_imports)]
extern crate criterion;
use criterion::{criterion_group, criterion_main, Criterion};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::{fft::fft, polynomial::PolynomialCoeffs, types::Field};
use rand::random;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn bench_cpu_ntt(c: &mut Criterion) {
    let log_ntt_sizes: Vec<usize> = (13..=19).collect();
    let mut group = c.benchmark_group("NTT");
    for log_ntt_size in log_ntt_sizes {
        let domain_size = 1usize << log_ntt_size;

        let gpu_buffer: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
        let coeffs = gpu_buffer
            .iter()
            .map(|i| GoldilocksField::from_canonical_u64(*i))
            .collect::<Vec<GoldilocksField>>();
        let coefficients = PolynomialCoeffs { coeffs };

        group.sample_size(20).bench_function(
            &format!("GoldilocksField NTT of size 2^{}", log_ntt_size),
            |b| b.iter(|| fft(coefficients.clone())),
        );
    }
}

criterion_group!(benches, bench_cpu_ntt);
criterion_main!(benches);
