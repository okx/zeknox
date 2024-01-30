extern crate criterion;
use cryptography_cuda::{ntt_batch,init_twiddle_factors_rust, types::NTTInputOutputOrder};
use rand::random;
use icicle_cuda_runtime::{
    stream::CudaStream,
    device_context::get_default_device_context
};
use rustacuda::memory::DeviceSlice;
// memory::DeviceSlice,
const DEFAULT_GPU: usize = 0;
fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn ntt_batch_with_lg(batches: usize, log_ntt_size:u32) {

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

fn main() {
    let start = std::time::Instant::now();
    let stream = CudaStream::create().unwrap();
    println!("total time spend init context: {:?}", start.elapsed());

    init_twiddle_factors_rust();

    let batches = 300;
    let log_ntt_size = 19;
    // ntt_batch_with_lg(1, 19);
    ntt_batch_with_lg(batches, log_ntt_size);
    // ntt_batch_with_lg(batches, log_ntt_size);
    // cpu_fft(log_ntt_size);
    // println!("after warm up");
    // let mut i = 0;
    // while(i<20){
    //     ntt_batch_with_lg(batches, log_ntt_size);
    //     i=i+1;
    // }
}
