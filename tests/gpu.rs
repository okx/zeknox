use cryptography_cuda::get_number_of_gpus_rs;

#[test]
fn test_get_number_of_gpus() {
    let nums = get_number_of_gpus_rs();
    assert!(nums>=0);
}