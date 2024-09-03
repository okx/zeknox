use cryptography_cuda::{
    bn128_add_rust, bn128_lshift_rust, bn128_mul_rust, bn128_rshift_rust, bn128_sub_rust,
};

// P = 21888242871839275222246405745257275088696311157297823662689037894645226208583 (0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47)

#[test]
fn test_bn128_add_rust() {
    // no overflow
    let mut a: [u32; 8] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000001,
    ];
    let mut b: [u32; 8] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000010,
    ];
    let gpu_ret = bn128_add_rust(&mut a, &mut b);
    assert_eq!(gpu_ret[0], 0x00000000);
    assert_eq!(gpu_ret[7], 0x00000011);

    // with overflow, but no reduction performed (in canonical form)
    let mut a: [u32; 8] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0xf0000001,
    ];
    let mut b: [u32; 8] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0xf0000010,
    ];
    // add result is [0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,0x00000000,0xe0000010];
    let gpu_ret = bn128_add_rust(&mut a, &mut b);
    assert_eq!(gpu_ret[0], 0x00000000);
    assert_eq!(gpu_ret[7], 0xe0000011);
}

#[test]
fn test_bn128_sub_rust() {
    // without borrow
    let mut a: [u32; 8] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0xf0000011,
    ];
    let mut b: [u32; 8] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0xf0000001,
    ];
    let gpu_ret = bn128_sub_rust(&mut a, &mut b);
    assert_eq!(gpu_ret[0], 0x00000000);
    assert_eq!(gpu_ret[7], 0x00000010);

    // no borrow
    let mut a: [u32; 8] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ];
    let mut b: [u32; 8] = [
        0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000001,
    ];
    let gpu_ret = bn128_sub_rust(&mut a, &mut b);
    println!("ret: {:?}", gpu_ret);
    assert_eq!(gpu_ret[0], 0xd87cfd47);
    assert_eq!(gpu_ret[7], 0x30644e71);
}

#[test]
fn test_bn128_mul_rust() {
    // without borrow
    let mut a: [u32; 8] = [
        0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ];
    let mut b: [u32; 8] = [
        0x00000011, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        0x00000000,
    ];
    let gpu_ret = bn128_mul_rust(&mut a, &mut b);
    println!("result: {:?}", gpu_ret);
    // assert_eq!(gpu_ret[0],0x00000000 );
    // assert_eq!(gpu_ret[7],0x00000010 );
}

#[test]
fn test_bn128_lshift_rust() {
    // no reduction by MOD
    let mut a: [u32; 8] = [
        0x10010000, 0x00100000, 0x01000000, 0x00010000, 0x00001000, 0x00000100, 0x00000010,
        0x00000001,
    ];
    let mut l: u32 = 2;
    let gpu_ret = bn128_lshift_rust(&mut a, &mut l);
    assert_eq!(gpu_ret[0], 0x40040000);
    assert_eq!(gpu_ret[1], 0x00400000);
    assert_eq!(gpu_ret[2], 0x04000000);
    assert_eq!(gpu_ret[3], 0x00040000);
    assert_eq!(gpu_ret[4], 0x00004000);
    assert_eq!(gpu_ret[5], 0x00000400);
    assert_eq!(gpu_ret[6], 0x00000040);
    assert_eq!(gpu_ret[7], 0x00000004);

    // need reduction by MOD
    let mut a: [u32; 8] = [
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffff11,
    ];
    let mut l: u32 = 1;
    let gpu_ret = bn128_lshift_rust(&mut a, &mut l);
    println!("gpu: {:?}", gpu_ret);
}

// TODO: add assertion
#[test]
fn test_bn128_rshift_rust() {
    // no reduction by MOD
    // let mut a:[u32;8] = [0x10010000, 0x00100000, 0x01000000, 0x00010000, 0x00001000, 0x00000100,0x00000010,0x00000001];
    // let mut l: u32 = 2;
    // let gpu_ret = bn128_rshift_rust(&mut a, &mut l);
    // assert_eq!(gpu_ret[0],0x40040000 );
    // assert_eq!(gpu_ret[1],0x00400000 );
    // assert_eq!(gpu_ret[2],0x04000000 );
    // assert_eq!(gpu_ret[3],0x00040000 );
    // assert_eq!(gpu_ret[4],0x00004000 );
    // assert_eq!(gpu_ret[5],0x00000400 );
    // assert_eq!(gpu_ret[6],0x00000040 );
    // assert_eq!(gpu_ret[7],0x00000004 );

    // shift right of odd value
    let mut a: [u32; 8] = [
        0x00000001, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x00000000,
    ];
    let mut l: u32 = 1;
    let gpu_ret = bn128_rshift_rust(&mut a, &mut l);
    println!("gpu: {:?}", gpu_ret);
}
