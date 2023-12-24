pub mod error;
pub mod types;

extern "C" {

    fn cuda_available() -> bool;

    fn compute_ntt(
        device_id: usize,
        inout: *mut core::ffi::c_void,
        lg_domain_size: u32,
        ntt_order: types::NTTInputOutputOrder,
        ntt_direction: types::NTTDirection,
        ntt_type: types::NTTType,
    ) -> error::Error;

    fn goldilocks_add(result: *mut u64, alloc: *mut u64, resbult: *mut u64) -> ();

    fn goldilocks_sub(result: *mut u64, alloc: *mut u64, resbult: *mut u64) -> ();

    fn goldilocks_mul(result: *mut u64, alloc: *mut u64, resbult: *mut u64) -> ();

    fn goldilocks_exp(result: *mut u64, base: *mut u64, pow: *mut u32) -> ();

    fn goldilocks_inverse(result: *mut u64, alloc: *mut u64) -> ();

    fn goldilocks_rshift(result: *mut u64, val: *mut u64, r: *mut u32) -> ();

    fn bn128_add(result: *mut [u32; 8], val: *mut [u32; 8], r: *mut [u32; 8]) -> ();
    fn bn128_sub(result: *mut [u32; 8], val: *mut [u32; 8], r: *mut [u32; 8]) -> ();
    fn bn128_mul(result: *mut [u32; 8], val: *mut [u32; 8], r: *mut [u32; 8]) -> ();
    fn bn128_lshift(result: *mut [u32; 8], val: *mut [u32; 8], l: *mut u32) -> ();
    fn bn128_rshift(result: *mut [u32; 8], val: *mut [u32; 8], r: *mut u32) -> ();
}

#[allow(non_snake_case)]
pub fn bn128_add_rust(a: &mut [u32; 8], b: &mut [u32; 8]) -> [u32; 8] {
    let mut result: [u32; 8] = [0; 8];
    unsafe { bn128_add(&mut result, a, b) };

    result
}

#[allow(non_snake_case)]
pub fn bn128_sub_rust(a: &mut [u32; 8], b: &mut [u32; 8]) -> [u32; 8] {
    let mut result: [u32; 8] = [0; 8];
    unsafe { bn128_sub(&mut result, a, b) };

    result
}

#[allow(non_snake_case)]
pub fn bn128_mul_rust(a: &mut [u32; 8], b: &mut [u32; 8]) -> [u32; 8] {
    let mut result: [u32; 8] = [0; 8];
    unsafe { bn128_mul(&mut result, a, b) };

    result
}

#[allow(non_snake_case)]
pub fn bn128_lshift_rust(a: &mut [u32; 8], l: &mut u32) -> [u32; 8] {
    let mut result: [u32; 8] = [0; 8];
    unsafe { bn128_lshift(&mut result, a, l) };

    result
}

#[allow(non_snake_case)]
pub fn bn128_rshift_rust(a: &mut [u32; 8], l: &mut u32) -> [u32; 8] {
    let mut result: [u32; 8] = [0; 8];
    unsafe { bn128_rshift(&mut result, a, l) };

    result
}

#[allow(non_snake_case)]
pub fn goldilocks_add_rust(a: &mut u64, b: &mut u64) -> u64 {
    let mut result: u64 = 0;
    unsafe { goldilocks_add(&mut result, a, b) };

    result
}

#[allow(non_snake_case)]
pub fn goldilocks_sub_rust(a: &mut u64, b: &mut u64) -> u64 {
    let mut result: u64 = 0;
    unsafe { goldilocks_sub(&mut result, a, b) };

    result
}

#[allow(non_snake_case)]
pub fn goldilocks_mul_rust(a: &mut u64, b: &mut u64) -> u64 {
    let mut result: u64 = 0;
    unsafe { goldilocks_mul(&mut result, a, b) };

    result
}

#[allow(non_snake_case)]
pub fn goldilocks_inverse_rust(a: &mut u64) -> u64 {
    let mut result: u64 = 0;
    unsafe { goldilocks_inverse(&mut result, a) };

    result
}

#[allow(non_snake_case)]
pub fn goldilocks_rshift_rust(a: &mut u64, r: &mut u32) -> u64 {
    let mut result: u64 = 0;
    unsafe { goldilocks_rshift(&mut result, a, r) };

    result
}

#[allow(non_snake_case)]
pub fn goldilocks_exp_rust(a: &mut u64, r: &mut u32) -> u64 {
    let mut result: u64 = 0;
    unsafe { goldilocks_exp(&mut result, a, r) };

    result
}

pub fn check_cuda_available() -> bool {
    unsafe { cuda_available() }
}

#[allow(non_snake_case)]
pub fn NTT<T>(device_id: usize, inout: &mut [T], order: types::NTTInputOutputOrder) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut core::ffi::c_void,
            len.trailing_zeros(),
            order,
            types::NTTDirection::Forward,
            types::NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

#[allow(non_snake_case)]
pub fn iNTT<T>(device_id: usize, inout: &mut [T], order: types::NTTInputOutputOrder) {
    let len = inout.len();
    if (len & (len - 1)) != 0 {
        panic!("inout.len() is not power of 2");
    }

    let err = unsafe {
        compute_ntt(
            device_id,
            inout.as_mut_ptr() as *mut core::ffi::c_void,
            len.trailing_zeros(),
            order,
            types::NTTDirection::Inverse,
            types::NTTType::Standard,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}
