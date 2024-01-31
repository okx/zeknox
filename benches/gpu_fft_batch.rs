extern crate criterion;
use criterion::{criterion_group, criterion_main, Criterion};
use cryptography_cuda::{init_twiddle_factors_rust, ntt_batch, types::NTTInputOutputOrder, NTT};
use rand::random;

const DEFAULT_GPU: usize = 0;
fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn bench_gpu_ntt(c: &mut Criterion) {
    let batches = 260;
    let LOG_NTT_SIZES: Vec<usize> = (16..=19).collect();
    let mut group = c.benchmark_group("NTT");

    for log_n in LOG_NTT_SIZES.clone() {
        init_twiddle_factors_rust(0, log_n as usize);
    }
    for log_ntt_size in LOG_NTT_SIZES {
        let domain_size = 1usize << log_ntt_size;

        let mut gpu_buffer: Vec<u64> = (0..domain_size * batches).map(|_| random_fr()).collect();

        group.sample_size(20).bench_function(
            &format!("GoldilocksField NTT batch of size 2^{}", log_ntt_size),
            |b| {
                b.iter(|| {
                    ntt_batch(
                        DEFAULT_GPU,
                        &mut gpu_buffer,
                        NTTInputOutputOrder::NN,
                        batches as u32,
                        log_ntt_size,
                    )
                })
            },
        );
    }
}

criterion_group!(ntt_benches, bench_gpu_ntt);
criterion_main!(ntt_benches);
