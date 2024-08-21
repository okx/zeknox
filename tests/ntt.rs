use std::ops::Mul;

use cryptography_cuda::device::memory::HostOrDeviceSlice;
use cryptography_cuda::{
    init_coset_rs, init_twiddle_factors_rs, intt, intt_batch, lde_batch, lde_batch_multi_gpu,
    naive_transpose_rev_batch, ntt, ntt_batch, transpose_rev_batch, types::*,
};
use plonky2_field::fft::{fft, ifft};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_field::types::{Field, PrimeField64};
use rand::random;

use crate::utils::transpose_and_rev;

pub mod utils;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

const DEFAULT_GPU: i32 = 0;
const DEFAULT_GPU2: i32 = 1;
const DEFAULT_GPU3: i32 = 2;
const DEFAULT_GPU4: i32 = 3;

#[test]
fn test_ntt_intt_gl64_self_consistency() {
    for lg_domain_size in 1..19 {
        let domain_size = 1usize << lg_domain_size;

        let v: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

        let mut gpu_buffer = v.clone();

        ntt(
            DEFAULT_GPU as usize,
            &mut gpu_buffer,
            NTTInputOutputOrder::NN,
        );

        intt(
            DEFAULT_GPU as usize,
            &mut gpu_buffer,
            NTTInputOutputOrder::NN,
        );

        assert_eq!(v, gpu_buffer);
    }
}

#[test]
fn test_ntt_gl64_consistency_with_plonky2() {
    for lg_domain_size in 1..20 {
        let domain_size = 1usize << lg_domain_size;

        let v: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
        let mut gpu_buffer = v.clone();
        ntt(
            DEFAULT_GPU as usize,
            &mut gpu_buffer,
            NTTInputOutputOrder::NN,
        );

        let plonky2_ntt_input = v.clone();
        let coeffs = plonky2_ntt_input
            .iter()
            .map(|i| GoldilocksField::from_canonical_u64(*i))
            .collect::<Vec<GoldilocksField>>();
        let coefficients = PolynomialCoeffs { coeffs };
        let points = fft(coefficients.clone());
        let cpu_results: Vec<u64> = points.values.iter().map(|x| x.to_canonical_u64()).collect();

        assert_eq!(gpu_buffer, cpu_results);
    }
}

#[test]
fn test_intt_gl64_consistency_with_plonky2() {
    for lg_domain_size in 10..20 {
        let domain_size = 1usize << lg_domain_size;

        let v: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
        let mut gpu_buffer = v.clone();
        intt(
            DEFAULT_GPU as usize,
            &mut gpu_buffer,
            NTTInputOutputOrder::NN,
        );

        let plonky2_ntt_input = v.clone();
        let values = plonky2_ntt_input
            .iter()
            .map(|i| GoldilocksField::from_canonical_u64(*i))
            .collect::<Vec<GoldilocksField>>();
        let values = PolynomialValues { values };
        let points = ifft(values.clone());
        let cpu_results: Vec<u64> = points.coeffs.iter().map(|x| x.to_canonical_u64()).collect();

        assert_eq!(gpu_buffer, cpu_results);
    }
}

#[test]
fn test_ntt_batch_gl64_consistency_with_plonky2() {
    let lg_domain_size: usize = 4;
    init_twiddle_factors_rs(DEFAULT_GPU as usize, lg_domain_size);
    let domain_size = 1usize << lg_domain_size;

    let v1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let v2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut gpu_buffer = v1.clone();
    gpu_buffer.extend(v2.iter());

    let mut cfg = NTTConfig::default();
    cfg.batches = 2;
    ntt_batch(
        DEFAULT_GPU as usize,
        gpu_buffer.as_mut_ptr(),
        lg_domain_size,
        cfg,
    );

    let plonky2_ntt_input1 = v1.clone();
    let coeffs1 = plonky2_ntt_input1
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let coefficients1 = PolynomialCoeffs { coeffs: coeffs1 };
    let points1 = fft(coefficients1.clone());
    let cpu_results1: Vec<u64> = points1
        .values
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    assert_eq!(gpu_buffer[0..1 << lg_domain_size], cpu_results1);

    let plonky2_ntt_input2 = v2.clone();
    let coeffs2 = plonky2_ntt_input2
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let coefficients2 = PolynomialCoeffs { coeffs: coeffs2 };
    let points2 = fft(coefficients2.clone());
    let cpu_results2: Vec<u64> = points2
        .values
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    assert_eq!(
        gpu_buffer[1 << lg_domain_size..(1 << lg_domain_size) * 2],
        cpu_results2
    );
}

#[test]
fn test_ntt_batch_intt_batch_gl64_self_consistency() {
    let lg_domain_size: usize = 10;
    init_twiddle_factors_rs(DEFAULT_GPU as usize, lg_domain_size);
    let domain_size = 1usize << lg_domain_size;

    let v1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let mut gpu_buffer = v1.clone();

    let cfg = NTTConfig::default();
    ntt_batch(
        DEFAULT_GPU as usize,
        gpu_buffer.as_mut_ptr(),
        lg_domain_size,
        cfg.clone(),
    );

    intt_batch(
        DEFAULT_GPU as usize,
        gpu_buffer.as_mut_ptr(),
        lg_domain_size,
        cfg.clone(),
    );
    assert_eq!(v1, gpu_buffer);
}

#[test]
fn test_intt_batch_gl64_consistency_with_plonky2() {
    let lg_domain_size: usize = 4;
    init_twiddle_factors_rs(DEFAULT_GPU as usize, lg_domain_size);

    let batches = 2;
    let domain_size = 1usize << lg_domain_size;

    let input1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let input2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut gpu_buffer = input1.clone();
    gpu_buffer.extend(input2.iter());

    let mut cfg = NTTConfig::default();
    cfg.batches = batches;
    intt_batch(
        DEFAULT_GPU as usize,
        gpu_buffer.as_mut_ptr(),
        lg_domain_size,
        cfg,
    );

    let plonky2_intt_input1 = input1.clone();
    let values1 = plonky2_intt_input1
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let polys = PolynomialValues { values: values1 };
    let points1 = ifft(polys.clone());
    let cpu_results1: Vec<u64> = points1
        .coeffs
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    assert_eq!(gpu_buffer[0..1 << lg_domain_size], cpu_results1);

    let plonky2_intt_input2 = input2.clone();
    let values2 = plonky2_intt_input2
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let polys = PolynomialValues { values: values2 };
    let points2 = ifft(polys.clone());
    let cpu_results2: Vec<u64> = points2
        .coeffs
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    assert_eq!(
        gpu_buffer[1 << lg_domain_size..(1 << lg_domain_size) * 2],
        cpu_results2
    );
}

#[test]
fn test_ntt_on_device() {
    let lg_domain_size = 10;
    let domain_size = 1usize << lg_domain_size;

    init_twiddle_factors_rs(0, lg_domain_size);

    let scalars: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut device_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, domain_size).unwrap();
    let _ret = device_data.copy_from_host(&scalars);

    let mut cfg = NTTConfig::default();
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    ntt_batch(0, device_data.as_mut_ptr(), lg_domain_size, cfg.clone());
    intt_batch(0, device_data.as_mut_ptr(), lg_domain_size, cfg.clone());

    let mut host_output = vec![0; domain_size];
    // println!("start copy to host");
    device_data
        .copy_to_host(host_output.as_mut_slice(), domain_size)
        .unwrap();
    assert_eq!(host_output, scalars.as_slice());
}

#[test]
fn test_ntt_batch_on_device() {
    let lg_domain_size = 10;
    let domain_size = 1usize << lg_domain_size;
    let batches = 2;

    init_twiddle_factors_rs(0, lg_domain_size);

    let total_elements = domain_size * batches;
    // let scalars: Vec<u64> = (0..(total_elements)).map(|_| random_fr()).collect();

    let mut input1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let mut input2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut cpu_buffer = input1.clone();
    cpu_buffer.extend(input2.iter());

    let mut device_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, total_elements).unwrap();
    let _ = device_data.copy_from_host_offset(input1.as_mut_slice(), 0, domain_size);
    let _ = device_data.copy_from_host_offset(input2.as_mut_slice(), domain_size, domain_size);
    // let ret = device_data.copy_from_host(&scalars);

    let mut cfg = NTTConfig::default();
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    cfg.batches = batches as u32;
    // println!("device data len: {:?}", device_data.len());
    ntt_batch(0, device_data.as_mut_ptr(), lg_domain_size, cfg.clone());

    let mut host_output = vec![0; total_elements];
    // println!("start copy to host");
    device_data
        .copy_to_host(host_output.as_mut_slice(), total_elements)
        .unwrap();
    // println!("host output: {:?}", host_output);

    let coeffs1 = cpu_buffer[0..domain_size]
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let coefficients1 = PolynomialCoeffs { coeffs: coeffs1 };
    let points1 = fft(coefficients1.clone());
    let cpu_results1: Vec<u64> = points1
        .values
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    assert_eq!(host_output[0..domain_size], cpu_results1);

    let coeffs2 = cpu_buffer[domain_size..total_elements]
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let coefficients2 = PolynomialCoeffs { coeffs: coeffs2 };
    let points2 = fft(coefficients2.clone());
    let cpu_results2: Vec<u64> = points2
        .values
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect();

    assert_eq!(
        host_output[1 << lg_domain_size..(1 << lg_domain_size) * 2],
        cpu_results2
    );
}

#[test]
fn test_naive_transpose_rev() {
    let lg_domain_size = 22;
    let domain_size = 1usize << lg_domain_size;
    let batches = 85;

    let total_elements = domain_size * batches;
    // let scalars: Vec<u64> = (0..(total_elements)).map(|_| random_fr()).collect();

    let mut input1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let mut input2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut cpu_buffer: Vec<Vec<u64>> = Vec::new();
    for _k in 0..batches / 2 {
        cpu_buffer.push(input1.clone());
        cpu_buffer.push(input2.clone());
    }
    if batches % 2 == 1 {
        cpu_buffer.push(input1.clone());
    }

    let mut device_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, total_elements).unwrap();
    let mut device_data2: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, total_elements).unwrap();
    for k in 0..batches / 2 {
        let _ = device_data.copy_from_host_offset(
            input1.as_mut_slice(),
            2 * k * domain_size,
            domain_size,
        );
        let _ = device_data.copy_from_host_offset(
            input2.as_mut_slice(),
            (2 * k + 1) * domain_size,
            domain_size,
        );
    }
    if batches % 2 == 1 {
        let _ = device_data.copy_from_host_offset(
            input1.as_mut_slice(),
            (batches - 1) * domain_size,
            domain_size,
        );
    }

    let mut cfg = TransposeConfig::default();
    cfg.batches = batches as u32;
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    // println!("device data len: {:?}", device_data.len());
    naive_transpose_rev_batch(
        DEFAULT_GPU,
        device_data2.as_mut_ptr(),
        device_data.as_mut_ptr(),
        lg_domain_size,
        cfg.clone(),
    );

    let mut host_output = vec![0; total_elements];
    // println!("start copy to host");
    device_data2
        .copy_to_host(host_output.as_mut_slice(), total_elements)
        .unwrap();

    let cpu_results_tranposed = transpose_and_rev(&cpu_buffer, lg_domain_size);
    let cpu_results: Vec<u64> = cpu_results_tranposed.into_iter().flatten().collect();
    assert_eq!(host_output, cpu_results);
}

#[test]
fn test_transpose_rev() {
    let lg_domain_size = 20;
    let domain_size = 1usize << lg_domain_size;
    let batches = 85;

    let total_elements = domain_size * batches;

    let mut input1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let mut input2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut cpu_buffer: Vec<Vec<u64>> = Vec::new();
    for _k in 0..batches / 2 {
        cpu_buffer.push(input1.clone());
        cpu_buffer.push(input2.clone());
    }
    if batches % 2 == 1 {
        cpu_buffer.push(input1.clone());
    }

    let mut device_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, total_elements).unwrap();
    let mut device_data2: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, total_elements).unwrap();
    for k in 0..batches / 2 {
        let _ = device_data.copy_from_host_offset(
            input1.as_mut_slice(),
            2 * k * domain_size,
            domain_size,
        );
        let _ = device_data.copy_from_host_offset(
            input2.as_mut_slice(),
            (2 * k + 1) * domain_size,
            domain_size,
        );
    }
    if batches % 2 == 1 {
        let _ = device_data.copy_from_host_offset(
            input1.as_mut_slice(),
            (batches - 1) * domain_size,
            domain_size,
        );
    }

    let mut cfg = TransposeConfig::default();
    cfg.batches = batches as u32;
    cfg.are_inputs_on_device = true;
    cfg.are_outputs_on_device = true;
    // println!("device data len: {:?}", device_data.len());
    transpose_rev_batch(
        DEFAULT_GPU,
        device_data2.as_mut_ptr(),
        device_data.as_mut_ptr(),
        lg_domain_size,
        cfg.clone(),
    );

    let mut host_output = vec![0; total_elements];
    // println!("start copy to host");
    device_data2
        .copy_to_host(host_output.as_mut_slice(), total_elements)
        .unwrap();

    let cpu_results_tranposed = transpose_and_rev(&cpu_buffer, lg_domain_size);
    let cpu_results: Vec<u64> = cpu_results_tranposed.into_iter().flatten().collect();

    assert_eq!(host_output, cpu_results);
}

#[test]
fn test_ntt_batch_with_coset() {
    let lg_domain_size = 4;
    let domain_size = 1usize << lg_domain_size;
    // let batches = 2;

    init_twiddle_factors_rs(DEFAULT_GPU as usize, lg_domain_size);
    init_coset_rs(
        DEFAULT_GPU as usize,
        lg_domain_size,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );

    let v1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let v2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut gpu_buffer = v1.clone();
    gpu_buffer.extend(v2.iter());

    let mut cfg = NTTConfig::default();
    cfg.with_coset = true;
    cfg.batches = 2;
    ntt_batch(
        DEFAULT_GPU as usize,
        gpu_buffer.as_mut_ptr(),
        lg_domain_size,
        cfg.clone(),
    );

    let cpu_buffer = v1.clone();

    let modified_poly: PolynomialCoeffs<GoldilocksField> = GoldilocksField::coset_shift()
        .powers()
        .zip(cpu_buffer)
        .map(|(r, c)| GoldilocksField::from_canonical_u64(c).mul(r))
        .collect::<Vec<_>>()
        .into();
    let ret = modified_poly
        .fft_with_options(None, None)
        .values
        .iter()
        .map(|v| v.to_canonical_u64())
        .collect::<Vec<u64>>();

    assert_eq!(gpu_buffer[0..domain_size], ret);

    let cpu_buffer2 = v2.clone();
    let modified_poly2: PolynomialCoeffs<GoldilocksField> = GoldilocksField::coset_shift()
        .powers()
        .zip(cpu_buffer2)
        .map(|(r, c)| GoldilocksField::from_canonical_u64(c).mul(r))
        .collect::<Vec<_>>()
        .into();
    let ret2 = modified_poly2
        .fft_with_options(None, None)
        .values
        .iter()
        .map(|v| v.to_canonical_u64())
        .collect::<Vec<u64>>();
    assert_eq!(gpu_buffer[domain_size..2 * domain_size], ret2);
}

#[test]
fn test_compute_batched_lde() {
    let lg_n: usize = 1;
    let rate_bits = 1;
    let lg_domain_size = lg_n + rate_bits;
    let batches = 2;
    init_twiddle_factors_rs(DEFAULT_GPU as usize, lg_domain_size);
    init_coset_rs(
        DEFAULT_GPU as usize,
        lg_domain_size,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );

    let input_size = 1usize << lg_n;

    let inputs = (0..batches)
        .into_iter()
        .map(|_| {
            let cpu_buffer: Vec<u64> = (0..input_size).map(|_| random_fr()).collect();
            let cpu_coeffs = cpu_buffer
                .iter()
                .map(|i| GoldilocksField::from_canonical_u64(*i))
                .collect::<Vec<GoldilocksField>>();
            let cpu_poly: PolynomialCoeffs<GoldilocksField> =
                PolynomialCoeffs { coeffs: cpu_coeffs };
            cpu_poly
        })
        .collect::<Vec<PolynomialCoeffs<GoldilocksField>>>();

    let mut cpu_polys_coeffs: Vec<GoldilocksField> = inputs
        .iter()
        .flat_map(|p| {
            let p_extended = p.lde(rate_bits);
            let modified_poly: PolynomialCoeffs<GoldilocksField> = GoldilocksField::coset_shift()
                .powers()
                .zip(&p_extended.coeffs)
                .map(|(r, &c)| r * c)
                .collect::<Vec<_>>()
                .into();
            modified_poly.coeffs
        })
        .collect();

    let mut cfg = NTTConfig::default();
    cfg.batches = batches as u32;
    // println!("ntt config {:?}", cfg);
    ntt_batch(
        DEFAULT_GPU as usize,
        cpu_polys_coeffs.as_mut_ptr(),
        lg_domain_size,
        cfg,
    );

    let cpu_outputs = cpu_polys_coeffs
        .iter()
        .map(|x| x.to_canonical_u64())
        .collect::<Vec<u64>>();

    // println!("inputs: {:?}", inputs);
    let mut gpu_buffer: Vec<u64> = inputs
        .clone()
        .into_iter()
        .flat_map(|p| {
            let coeffs = p
                .coeffs
                .iter()
                .map(|x| x.to_canonical_u64())
                .collect::<Vec<u64>>();
            return coeffs;
        })
        .collect();
    let mut cfg_lde = NTTConfig::default();
    cfg_lde.batches = batches as u32;
    cfg_lde.extension_rate_bits = rate_bits as u32;
    cfg_lde.with_coset = true;

    let mut gpu_lde_output = vec![0; (1 << lg_domain_size) * batches];
    lde_batch(
        DEFAULT_GPU as usize,
        gpu_lde_output.as_mut_ptr(),
        gpu_buffer.as_mut_ptr(),
        lg_n,
        cfg_lde,
    );
    assert_eq!(cpu_outputs, gpu_lde_output);
}

#[test]
fn test_compute_batched_lde_data_on_device() {
    let lg_n: usize = 17;
    let rate_bits = 3;
    let lg_domain_size = lg_n + rate_bits;
    let input_domain_size = 1usize << lg_n;
    let output_domain_size = 1usize << (lg_n + rate_bits);
    let batches = 2;

    init_twiddle_factors_rs(DEFAULT_GPU as usize, lg_domain_size);
    init_coset_rs(
        DEFAULT_GPU as usize,
        lg_domain_size,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );

    let total_num_input_elements = input_domain_size * batches;
    let total_num_output_elements = output_domain_size * batches;

    let host_inputs = (0..batches)
        .collect::<Vec<usize>>()
        .iter()
        .map(|_| (0..input_domain_size).map(|_| random_fr()).collect())
        .collect::<Vec<Vec<u64>>>();

    let mut device_input_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, total_num_input_elements).unwrap();

    host_inputs.iter().enumerate().for_each(|(i, p)| {
        let _ = device_input_data.copy_from_host_offset(
            p.as_slice(),
            input_domain_size * i,
            input_domain_size,
        );
    });

    // lde rust allocate to gpu prior to api call
    let mut device_output_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, total_num_output_elements).unwrap();

    let mut cfg_lde = NTTConfig::default();
    cfg_lde.batches = batches as u32;
    cfg_lde.extension_rate_bits = rate_bits as u32;
    cfg_lde.with_coset = true;
    cfg_lde.are_inputs_on_device = true;
    cfg_lde.are_outputs_on_device = true;

    lde_batch(
        DEFAULT_GPU as usize,
        device_output_data.as_mut_ptr(),
        device_input_data.as_mut_ptr(),
        lg_n,
        cfg_lde,
    );

    let mut host_output_first = vec![0; output_domain_size];
    let mut host_output_last = vec![0; output_domain_size];

    let _ = device_output_data.copy_to_host_offset(
        host_output_first.as_mut_slice(),
        0,
        output_domain_size,
    );
    let _ = device_output_data.copy_to_host_offset(
        host_output_last.as_mut_slice(),
        output_domain_size * (batches - 1),
        output_domain_size,
    );

    // lde gpu copy from host during api call
    let mut cfg_lde_copy = NTTConfig::default();
    cfg_lde_copy.batches = batches as u32;
    cfg_lde_copy.extension_rate_bits = rate_bits as u32;
    cfg_lde_copy.with_coset = true;

    let mut gpu_lde_output_copy = vec![0; total_num_output_elements];
    let mut lde_copy_buffer = host_inputs
        .into_iter()
        .flat_map(|p| p)
        .collect::<Vec<u64>>();
    lde_batch(
        DEFAULT_GPU as usize,
        gpu_lde_output_copy.as_mut_ptr(),
        lde_copy_buffer.as_mut_ptr(),
        lg_n,
        cfg_lde_copy,
    );
    assert_eq!(
        gpu_lde_output_copy[0..output_domain_size],
        host_output_first
    );
    assert_eq!(
        gpu_lde_output_copy[output_domain_size * (batches - 1)..output_domain_size * batches],
        host_output_last
    );
}

#[test]
fn test_compute_batched_lde_multi_gpu_data_on_one_gpu() {
    let lg_n: usize = 17;
    let rate_bits = 3;
    let lg_domain_size = lg_n + rate_bits;
    let input_domain_size = 1usize << lg_n;
    let output_domain_size = 1usize << (lg_n + rate_bits);
    let batches = 200;

    init_twiddle_factors_rs(DEFAULT_GPU as usize, lg_domain_size);
    init_twiddle_factors_rs(DEFAULT_GPU2 as usize, lg_domain_size);
    init_twiddle_factors_rs(DEFAULT_GPU3 as usize, lg_domain_size);
    init_twiddle_factors_rs(DEFAULT_GPU4 as usize, lg_domain_size);

    init_coset_rs(
        DEFAULT_GPU as usize,
        lg_domain_size,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );
    init_coset_rs(
        DEFAULT_GPU2 as usize,
        lg_domain_size,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );

    init_coset_rs(
        DEFAULT_GPU3 as usize,
        lg_domain_size,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );

    init_coset_rs(
        DEFAULT_GPU4 as usize,
        lg_domain_size,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );

    for i in 0..10 {
        println!("Starting test: {:?}", i + 1);
        let total_num_input_elements = input_domain_size * batches;
        let total_num_output_elements = output_domain_size * batches;

        let mut host_inputs: Vec<u64> = (0..batches)
            .collect::<Vec<usize>>()
            .iter()
            .map(|_| {
                (0..input_domain_size)
                    .map(|_| random_fr())
                    .collect::<Vec<u64>>()
            })
            .flatten()
            .collect();

        // println!("Length of inputs: {:?}", host_inputs.len());

        // lde rust allocate to gpu prior to api call
        let mut device_output_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, total_num_output_elements).unwrap();

        let mut cfg_lde = NTTConfig::default();
        cfg_lde.batches = batches as u32;
        cfg_lde.extension_rate_bits = rate_bits as u32;
        cfg_lde.with_coset = true;
        cfg_lde.are_inputs_on_device = true;
        cfg_lde.are_outputs_on_device = true;
        cfg_lde.is_multi_gpu = true;

        lde_batch_multi_gpu(
            device_output_data.as_mut_ptr(),
            host_inputs.as_mut_ptr(),
            4,
            cfg_lde.clone(),
            lg_n,
            total_num_input_elements,
            total_num_output_elements,
        );

        // println!("LDE completed");

        let mut host_output = vec![0; batches * output_domain_size];

        let _ = device_output_data.copy_to_host_offset(
            host_output.as_mut_slice(),
            0,
            output_domain_size * batches,
        );
        println!("LDE completed, copied from gpu");

        // lde gpu copy from host during api call
        let mut cfg_lde_copy = NTTConfig::default();
        cfg_lde_copy.batches = batches as u32;
        cfg_lde_copy.extension_rate_bits = rate_bits as u32;
        cfg_lde_copy.with_coset = true;

        let mut gpu_lde_output_copy = vec![0; total_num_output_elements];
        let mut lde_copy_buffer = host_inputs.clone();

        lde_batch(
            DEFAULT_GPU as usize,
            gpu_lde_output_copy.as_mut_ptr(),
            lde_copy_buffer.as_mut_ptr(),
            lg_n,
            cfg_lde_copy,
        );

        assert_eq!(gpu_lde_output_copy, host_output);
    }
}
