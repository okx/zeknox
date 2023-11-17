use std::ops::Add;

use cryptography_cuda::{goldilocks_add_rust, check_cuda_available, mul_rust};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::types::Field;
#[test]
fn test_goldilocks_add_rust() {

    let mut a:u64 = 0xffffffff00000000;
    let mut b:u64 = 0x00000001ffffffff;
    let gpu_ret = goldilocks_add_rust(&mut a, &mut b);


    let mut a_cpu = GoldilocksField::from_canonical_u64(a);
    let b_cpu = GoldilocksField::from_canonical_u64(b);
    let cpu_ret = a_cpu.add(b_cpu);

    assert_eq!(gpu_ret, cpu_ret.0);
}

#[test]
fn test_cuda_available() {
    let available = check_cuda_available();
    println!("available: {:?}", available);
}


#[test]
fn test_mul() {
    let result = mul_rust();
    println!("result: {:?}", result);
}