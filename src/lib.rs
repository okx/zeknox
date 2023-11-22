#[repr(C)]
pub struct Error {
    pub code: i32,
    str: Option<core::ptr::NonNull<i8>>, // just strdup("string") from C/C++
}

extern "C" {
    fn goldilocks_add(result: *mut u64, alloc: *mut u64, resbult: *mut u64) -> ();

    fn goldilocks_sub(result: *mut u64, alloc: *mut u64, resbult: *mut u64) -> ();

    fn goldilocks_mul(result: *mut u64, alloc: *mut u64, resbult: *mut u64) -> ();

    fn goldilocks_inverse(result: *mut u64, alloc: *mut u64) -> ();

    fn goldilocks_rshift(result: *mut u64, val: *mut u64, r: *mut u32) -> ();

    fn cuda_available() -> bool;
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

pub fn check_cuda_available() -> bool {
    unsafe { cuda_available() }
}


#[cfg(test)]
mod tests {
    
    #[test]
    fn test_sub() {
        let x: u8 = 1;
        let y: u8 = 24;
        let (ret, borrow) = x.overflowing_sub(y);
        println!("ret: {:?}, borrow: {:?}", ret, borrow);
    }
}