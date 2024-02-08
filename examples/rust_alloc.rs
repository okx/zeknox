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
    let log_ntt_size = 10;
    let domain_size = 1usize << log_ntt_size;

    let scalars: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    // let scalars = HostOrDeviceSlice::Host(random_fr_n(size));

    let mut device_data: HostOrDeviceSlice<'_, u64> = HostOrDeviceSlice::cuda_malloc(domain_size).unwrap();
    let ret = device_data.copy_from_host(&scalars);
    println!("device_data: {:?}", ret);
}