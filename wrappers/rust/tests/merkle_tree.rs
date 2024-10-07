use core::mem::MaybeUninit;
use core::slice;

use cryptography_cuda::{
    device::{memory::HostOrDeviceSlice, stream::CudaStream},
    fill_digests_buf_linear_gpu_with_gpu_ptr, fill_digests_buf_linear_multigpu_with_gpu_ptr,
};
use plonky2::{
    hash::{
        poseidon_bn128::PoseidonBN128GoldilocksConfig,
        hash_types::{RichField, NUM_HASH_OUT_ELTS},
        merkle_tree::MerkleTree
    },
    plonk::config::{
        GenericConfig, Hasher, HasherType, Poseidon2GoldilocksConfig,
        PoseidonGoldilocksConfig,
    },
};
use plonky2_field::types::Field;

fn random_data<F: RichField>(n: usize, k: usize) -> Vec<Vec<F>> {
    (0..n).map(|_| F::rand_vec(k)).collect()
}

fn fill_digests_buf_gpu_ptr<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves_ptr: *const F,
    leaves_len: usize,
    leaf_len: usize,
    cap_height: usize,
    gpu_id: u64,
) {
    let digests_count: u64 = digests_buf.len().try_into().unwrap();
    let leaves_count: u64 = leaves_len.try_into().unwrap();
    let caps_count: u64 = cap_buf.len().try_into().unwrap();
    let cap_height: u64 = cap_height.try_into().unwrap();
    let leaf_size: u64 = leaf_len.try_into().unwrap();

    // if digests_buf is empty (size 0), just allocate a few bytes to avoid errors
    let digests_size = if digests_buf.len() == 0 {
        NUM_HASH_OUT_ELTS
    } else {
        digests_buf.len() * NUM_HASH_OUT_ELTS
    };
    let caps_size = if cap_buf.len() == 0 {
        NUM_HASH_OUT_ELTS
    } else {
        cap_buf.len() * NUM_HASH_OUT_ELTS
    };

    let mut gpu_digests_buf: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(gpu_id as i32, digests_size).unwrap();
    let mut gpu_cap_buf: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(gpu_id as i32, caps_size).unwrap();

    unsafe {
        let num_gpus: usize = std::env::var("NUM_OF_GPUS")
            .expect("NUM_OF_GPUS should be set")
            .parse()
            .unwrap();
        if leaves_count >= (1 << 12)
            && cap_height > 0
            && num_gpus > 1
            && H::HASHER_TYPE == HasherType::PoseidonBN128
        {
            // println!("Multi GPU");
            fill_digests_buf_linear_multigpu_with_gpu_ptr(
                gpu_digests_buf.as_mut_ptr() as *mut core::ffi::c_void,
                gpu_cap_buf.as_mut_ptr() as *mut core::ffi::c_void,
                leaves_ptr as *mut core::ffi::c_void,
                digests_count,
                caps_count,
                leaves_count,
                leaf_size,
                cap_height,
                H::HASHER_TYPE as u64,
            );
        } else {
            // println!("Single GPU");
            fill_digests_buf_linear_gpu_with_gpu_ptr(
                gpu_digests_buf.as_mut_ptr() as *mut core::ffi::c_void,
                gpu_cap_buf.as_mut_ptr() as *mut core::ffi::c_void,
                leaves_ptr as *mut core::ffi::c_void,
                digests_count,
                caps_count,
                leaves_count,
                leaf_size,
                cap_height,
                H::HASHER_TYPE as u64,
                gpu_id,
            );
        }
    }

    let stream1 = CudaStream::create().unwrap();
    let stream2 = CudaStream::create().unwrap();

    gpu_digests_buf
        .copy_to_host_ptr_async(
            digests_buf.as_mut_ptr() as *mut core::ffi::c_void,
            digests_size,
            &stream1,
        )
        .expect("copy digests");
    gpu_cap_buf
        .copy_to_host_ptr_async(
            cap_buf.as_mut_ptr() as *mut core::ffi::c_void,
            caps_size,
            &stream2,
        )
        .expect("copy caps");
    stream1.synchronize().expect("cuda sync");
    stream2.synchronize().expect("cuda sync");
    stream1.destroy().expect("cuda stream destroy");
    stream2.destroy().expect("cuda stream destroy");
}

fn fill_digests_buf_gpu<F: RichField, H: Hasher<F>>(
    digests_buf: &mut [MaybeUninit<H::Hash>],
    cap_buf: &mut [MaybeUninit<H::Hash>],
    leaves: &Vec<F>,
    leaf_size: usize,
    cap_height: usize,
) {
    let leaves_count = leaves.len() / leaf_size;

    let gpu_id = 0;

    let mut gpu_leaves_buf: HostOrDeviceSlice<'_, F> =
        HostOrDeviceSlice::cuda_malloc(gpu_id as i32, leaves.len()).unwrap();

    let _ = gpu_leaves_buf.copy_from_host(leaves.as_slice());

    fill_digests_buf_gpu_ptr::<F, H>(
        digests_buf,
        cap_buf,
        gpu_leaves_buf.as_mut_ptr(),
        leaves_count,
        leaf_size,
        cap_height,
        gpu_id,
    );
}

fn capacity_up_to_mut<T>(v: &mut Vec<T>, len: usize) -> &mut [MaybeUninit<T>] {
    assert!(v.capacity() >= len);
    let v_ptr = v.as_mut_ptr().cast::<MaybeUninit<T>>();
    unsafe {
        // SAFETY: `v_ptr` is a valid pointer to a buffer of length at least `len`. Upon return, the
        // lifetime will be bound to that of `v`. The underlying memory will not be deallocated as
        // we hold the sole mutable reference to `v`. The contents of the slice may be
        // uninitialized, but the `MaybeUninit` makes it safe.
        slice::from_raw_parts_mut(v_ptr, len)
    }
}

fn test_merkle_trees_consistency_with_plonky2<C>()
where C: GenericConfig<2>
{
    let leaves_count = 1 << 12; // number of leaves
    let leaf_size = 7; // leaf size
    let cap_height = 1; // cap height
    let leaves_2d = random_data(leaves_count, leaf_size);

    // MT on CPU from Plonky2
    let mt = MerkleTree::<C::F, C::Hasher>::new_from_2d(leaves_2d.clone(), cap_height);

    // MT on GPU
    let zeros = vec![C::F::ZERO; leaf_size];
    let mut leaves_1d: Vec<C::F> = Vec::with_capacity(leaves_count * leaf_size);
    for idx in 0..leaves_count {
        if leaves_2d[idx].len() == 0 {
            leaves_1d.extend(zeros.clone());
        } else {
            leaves_1d.extend(leaves_2d[idx].clone());
        }
    }

    let num_digests = 2 * (leaves_count - (1 << cap_height));
    let mut digests: Vec<<C::Hasher as Hasher<C::F>>::Hash> = Vec::with_capacity(num_digests);

    let len_cap = 1 << cap_height;
    let mut cap: Vec<<C::Hasher as Hasher<C::F>>::Hash> = Vec::with_capacity(len_cap);

    let mut digests_buf = capacity_up_to_mut(&mut digests, num_digests);
    let mut cap_buf = capacity_up_to_mut(&mut cap, len_cap);

    fill_digests_buf_gpu::<C::F, C::Hasher>(
        &mut digests_buf,
        &mut cap_buf,
        &leaves_1d,
        leaf_size,
        cap_height,
    );

    mt.digests.iter().zip(digests.iter()).for_each(|(d1, d2)| {
        assert_eq!(d1, d2);
    });

    mt.cap.0.iter().zip(cap.iter()).for_each(|(d1, d2)| {
        assert_eq!(d1, d2);
    });
}

#[test]
fn test_merkle_trees_poseidon_g64_consistency_with_plonky2() {
    test_merkle_trees_consistency_with_plonky2::<PoseidonGoldilocksConfig>();
}

#[test]
fn test_merkle_trees_poseidon2_g64_consistency_with_plonky2() {
    test_merkle_trees_consistency_with_plonky2::<Poseidon2GoldilocksConfig>();
}

#[test]
fn test_merkle_trees_poseidon_bn128_consistency_with_plonky2() {
    test_merkle_trees_consistency_with_plonky2::<PoseidonBN128GoldilocksConfig>();
}