use cryptography_cuda::{
    device::stream::CudaStream,
    device::memory::HostOrDeviceSlice,
};
use cryptography_cuda::{intt, init_twiddle_factors_rs, intt_batch, ntt_batch, types::*, ntt,get_number_of_gpus_rs};
use rand::random;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn random_fr_n(n: usize) -> Vec<u64> {
    (0..n).into_iter().map(|_| random_fr()).collect()
}

fn main() {
    let lg_domain_size = 10;
    let domain_size = 1usize << lg_domain_size;

    init_twiddle_factors_rs(0, lg_domain_size);

    let scalars: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut device_data: HostOrDeviceSlice<'_, u64> = HostOrDeviceSlice::cuda_malloc(domain_size).unwrap();
    let ret = device_data.copy_from_host(&scalars);

    let mut cfg = NTTConfig::default();
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    ntt_batch(0, device_data.as_mut_ptr(), lg_domain_size, cfg.clone());
    intt_batch(0, device_data.as_mut_ptr(), lg_domain_size, cfg.clone() );

    let mut host_output = vec![0;domain_size];
    // println!("start copy to host");
    device_data.copy_to_host(host_output.as_mut_slice()).unwrap();
    assert_eq!(host_output, scalars.as_slice());


}