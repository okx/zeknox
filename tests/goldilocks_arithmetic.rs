use cryptography_cuda::{
    check_cuda_available, goldilocks_add_rust, goldilocks_mul_rust, goldilocks_sub_rust,
};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::types::{Field, PrimeField64};
use std::ops::{Add, Mul, Sub};

#[test]
fn test_goldilocks_add_rust() {
    // (c, carry) = a + b; (tmp, borrow) = c - MOD;

    // case 1, carry = false && borrow = false (c> MOD); should use tmp (c - MOD)
    let mut a:u64 = 0xfffffffc_00000000;
    let mut b:u64 = 0x00000003_00000002;
    let gpu_ret = goldilocks_add_rust(&mut a, &mut b);
    assert_eq!(gpu_ret, 1);
    let a_cpu = GoldilocksField::from_canonical_u64(a);
    let b_cpu = GoldilocksField::from_canonical_u64(b);
    let cpu_ret = a_cpu.add(b_cpu).to_canonical_u64();
    assert_eq!(gpu_ret, cpu_ret);

    // case 2, carry = false && borrow = true (c < MOD); should use c
    let mut a:u64 = 0xfffffff0_00000000;
    let mut b:u64 = 0x0000000f_00000000;
    let gpu_ret = goldilocks_add_rust(&mut a, &mut b);
    assert_eq!(gpu_ret, 0xffffffff_00000000);
    let a_cpu = GoldilocksField::from_canonical_u64(a);
    let b_cpu = GoldilocksField::from_canonical_u64(b);
    let cpu_ret = a_cpu.add(b_cpu).to_canonical_u64();
    assert_eq!(gpu_ret, cpu_ret);

    // case 3, carry = true && borrow = true (c < MOD); should use tmp
    let mut a: u64 = 0xffffffff_00000000;
    let mut b: u64 = 0x0000000e_ffffffff;
    let gpu_ret = goldilocks_add_rust(&mut a, &mut b);
    assert_eq!(gpu_ret, 64424509438); // 0000000e_fffffffe
    let a_cpu = GoldilocksField::from_canonical_u64(a);
    let b_cpu = GoldilocksField::from_canonical_u64(b);
    let cpu_ret = a_cpu.add(b_cpu).to_canonical_u64();
    assert_eq!(gpu_ret, cpu_ret);

    // TODO: case 4, carry = true && borrow = false (c > MOD); should use c
    // let mut a: u64 = 0xffffffff_f0000fff;
    // let mut b: u64 = 0xffffffff_f0000fff;
    // let gpu_ret = goldilocks_add_rust(&mut a, &mut b);
    // println!("gpu ret: {:?}", gpu_ret);
    // let a_cpu = GoldilocksField::from_noncanonical_u64(a).to_canonical();
    // let b_cpu = GoldilocksField::from_noncanonical_u64(b).to_canonical();
    // let cpu_ret = a_cpu.add(b_cpu).to_canonical();
    // println!("a_cpu: {:?}, b_cpu: {:?}, cpu_ret: {:?}",a_cpu, b_cpu, cpu_ret);
}

#[test]
fn test_goldilocks_sub_rust() {
    let mut a: u64 = 0x00000001ffffffff;
    let mut b: u64 = 0x0000000efffffff1;
    let gpu_ret = goldilocks_sub_rust(&mut a, &mut b);

    let a_cpu = GoldilocksField::from_canonical_u64(a);
    let b_cpu = GoldilocksField::from_canonical_u64(b);
    let cpu_ret = a_cpu.sub(b_cpu);
    assert_eq!(gpu_ret, cpu_ret.0);
}

#[test]
fn test_goldilocks_mul_rust() {
    let mut a: u64 = 0x0000000011ffffff;
    let mut b: u64 = 0x0000000011fffff1;
    let gpu_ret = goldilocks_mul_rust(&mut a, &mut b);

    let a_cpu = GoldilocksField::from_canonical_u64(a);
    let b_cpu = GoldilocksField::from_canonical_u64(b);
    let cpu_ret = a_cpu.mul(b_cpu).to_canonical_u64();
    assert_eq!(gpu_ret, cpu_ret);
}

#[test]
fn test_cuda_available() {
    let available = check_cuda_available();
    println!("available: {:?}", available);
}
