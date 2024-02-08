use cryptography_cuda::{
    device::stream::CudaStream,
    device::memory::HostOrDeviceSlice,
};
use rand::random;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn random_fr_n(n: usize) -> Vec<u64> {
    (0..n).into_iter().map(|_| random_fr()).collect()
}

fn main() {
    let log_size = 10;
    let size = 1 << log_size;
    let scalars = HostOrDeviceSlice::Host(random_fr_n(size));

    let mut device_data: HostOrDeviceSlice<'_, u64> = HostOrDeviceSlice::cuda_malloc(size).unwrap();
    // println!("device_data: {:?}", device_data);
}