// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use types::{NTTConfig, TransposeConfig};

#[cfg(feature = "cuda")]
pub mod device;
pub mod error;
pub mod types;

extern "C" {

    fn list_devices_info() -> error::Error;
    fn get_number_of_gpus(ngpus: *mut usize) -> error::Error;

    fn init_twiddle_factors(device_id: usize, lg_n: usize) -> error::Error;
    fn init_coset(device_id: usize, lg_n: usize, coset_gen: u64) -> error::Error;

    fn init_cuda();

    fn init_cuda_degree(max_degree: usize);

    fn compute_batched_ntt(
        device_id: usize,
        inout: *mut core::ffi::c_void,
        lg_domain_size: usize,
        ntt_direction: types::NTTDirection,
        cfg: types::NTTConfig,
    ) -> error::Error;

    fn compute_batched_lde(
        device_id: usize,
        output: *mut core::ffi::c_void,
        input: *mut core::ffi::c_void,
        lg_domain_size: usize,
        ntt_direction: types::NTTDirection,
        cfg: types::NTTConfig,
    ) -> error::Error;

    fn compute_batched_lde_multi_gpu(
        output: *mut core::ffi::c_void,
        input: *mut core::ffi::c_void,
        num_gpu: usize,
        ntt_direction: types::NTTDirection,
        cfg: types::NTTConfig,
        lg_domain_size: usize,
        total_num_input_elements: usize,
        total_num_output_elements: usize,
    ) -> error::Error;

    fn compute_transpose_rev(
        device_id: i32,
        output: *mut core::ffi::c_void,
        input: *mut core::ffi::c_void,
        lg_n: usize,
        cfg: types::TransposeConfig,
    ) -> error::Error;

    pub fn fill_digests_buf_linear_gpu_with_gpu_ptr(
        digests_buf_gpu_ptr: *mut ::std::os::raw::c_void,
        cap_buf_gpu_ptr: *mut ::std::os::raw::c_void,
        leaves_buf_gpu_ptr: *mut ::std::os::raw::c_void,
        digests_buf_size: u64,
        cap_buf_size: u64,
        leaves_buf_size: u64,
        leaf_size: u64,
        cap_height: u64,
        hash_type: u64,
        gpu_id: u64,
    );

    pub fn fill_digests_buf_linear_multigpu_with_gpu_ptr(
        digests_buf_gpu_ptr: *mut ::std::os::raw::c_void,
        cap_buf_gpu_ptr: *mut ::std::os::raw::c_void,
        leaves_buf_gpu_ptr: *mut ::std::os::raw::c_void,
        digests_buf_size: u64,
        cap_buf_size: u64,
        leaves_buf_size: u64,
        leaf_size: u64,
        cap_height: u64,
        hash_type: u64,
    );

    pub fn fill_digests_buf_linear_cpu(
        digests_buf_ptr: *mut ::std::os::raw::c_void,
        cap_buf_ptr: *mut ::std::os::raw::c_void,
        leaves_buf_ptr: *const ::std::os::raw::c_void,
        digests_buf_size: u64,
        cap_buf_size: u64,
        leaves_buf_size: u64,
        leaf_size: u64,
        cap_height: u64,
        hash_type: u64,
    );
}

pub fn list_devices_info_rs() {
    unsafe {
        list_devices_info();
    }
}

pub fn get_number_of_gpus_rs() -> usize {
    let mut nums = 0;
    let err = unsafe { get_number_of_gpus(&mut nums) };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
    return nums;
}

pub fn lde_batch<T>(
    device_id: usize,
    output: *mut T,  // &mut [T],
    input: *const T, // &mut [T],
    log_n_size: usize,
    cfg: NTTConfig,
) {
    let err = unsafe {
        compute_batched_lde(
            device_id,
            output as *mut core::ffi::c_void,
            input as *mut core::ffi::c_void,
            log_n_size,
            types::NTTDirection::Forward,
            cfg,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn lde_batch_multi_gpu<T>(
    output: *mut T,  // &mut [T],
    input: *const T, // &mut [T],
    num_gpu: usize,
    cfg: NTTConfig,
    log_n_size: usize,
    total_num_input_elements: usize,
    total_num_output_elements: usize,
) {
    let err = unsafe {
        // println!("In compute_batched_lde_multi_gpu {:?}", total_num_input_elements);
        compute_batched_lde_multi_gpu(
            output as *mut core::ffi::c_void,
            input as *mut core::ffi::c_void,
            num_gpu,
            types::NTTDirection::Forward,
            cfg,
            log_n_size,
            total_num_input_elements,
            total_num_output_elements,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn ntt_batch<T>(
    device_id: usize,
    inout: *mut T, // &mut [T],
    log_n_size: usize,
    cfg: NTTConfig,
) {
    let err = unsafe {
        compute_batched_ntt(
            device_id,
            // inout.as_mut_ptr() as *mut core::ffi::c_void,
            inout as *mut core::ffi::c_void,
            log_n_size,
            types::NTTDirection::Forward,
            cfg,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn intt_batch<T>(device_id: usize, inout: *mut T, log_n_size: usize, cfg: NTTConfig) {
    let err = unsafe {
        compute_batched_ntt(
            device_id,
            inout as *mut core::ffi::c_void,
            log_n_size,
            types::NTTDirection::Inverse,
            cfg,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn transpose_rev_batch<T>(
    device_id: i32,
    output: *mut T,  // &mut [T],
    input: *const T, // &mut [T],
    log_n_size: usize,
    cfg: TransposeConfig,
) {
    let err = unsafe {
        compute_transpose_rev(
            device_id,
            output as *mut core::ffi::c_void,
            input as *mut core::ffi::c_void,
            log_n_size,
            cfg,
        )
    };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn init_twiddle_factors_rs(device_id: usize, lg_n: usize) {
    let err = unsafe { init_twiddle_factors(device_id, lg_n) };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn init_coset_rs(device_id: usize, lg_n: usize, coset_gen: u64) {
    let err = unsafe { init_coset(device_id, lg_n, coset_gen) };

    if err.code != 0 {
        panic!("{}", String::from(err));
    }
}

pub fn init_cuda_rs() {
    unsafe {
        init_cuda();
    }
}

pub fn init_cuda_degree_rs(max_degree: usize) {
    unsafe {
        init_cuda_degree(max_degree);
    }
}
