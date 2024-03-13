#[warn(dead_code)]


use cryptography_cuda::{list_devices_info_rs, get_number_of_gpus_rs};


/// example output
/// Device 0 - Tesla V100-SXM2-16GB
/// CUDA multi processor count: 80   CUDA Cores: 5120
#[test]
fn test_list_devices_info_rs() {
    list_devices_info_rs()
}

#[test]
fn test_get_number_of_gpus() {
    let nums = get_number_of_gpus_rs();
    assert!(nums>=0);
}