#[repr(C)]
pub struct Error {
    pub code: i32,
    str: Option<core::ptr::NonNull<i8>>, // just strdup("string") from C/C++
}

extern "C" {
    fn goldilocks_add(
        result: *mut u64,
        alloc: *mut u64,
        resbult: *mut u64,
    ) -> ();

    fn mul(
        result: *mut u32,
    ) -> ();



    fn  cuda_available() -> bool;
}

/// Compute an in-place NTT on the input data.
#[allow(non_snake_case)]
pub fn goldilocks_add_rust(a: &mut u64, b: &mut u64) -> u64 {
    let mut result: u64 = 0;
    unsafe {
        println!("a: {:?}, b: {:?}", *a, *b);
        goldilocks_add(
            &mut result ,
            a ,
            b ,
        )
    };

    result
}

pub fn check_cuda_available() -> bool {
    unsafe {
        cuda_available()
    }
}

pub fn mul_rust() -> u32 {
    let mut result: u32 = 0;
    unsafe {
        mul(&mut result);
    }
    result
}
