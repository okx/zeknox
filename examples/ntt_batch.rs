extern crate criterion;
use cryptography_cuda::{ntt_batch, types::NTTInputOutputOrder};
use rand::random;

const DEFAULT_GPU: usize = 0;
fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn main() {
    let batches = 300;
    let log_ntt_size = 19;
    let domain_size = 1usize << log_ntt_size;

    let mut gpu_buffer: Vec<u64> = (0..domain_size * batches).map(|_| random_fr()).collect();
    let start = std::time::Instant::now();
    ntt_batch(
        DEFAULT_GPU,
        &mut gpu_buffer,
        NTTInputOutputOrder::NN,
        batches as u32,
        log_ntt_size,
    );
    println!("total time spend: {:?}", start.elapsed());
}
