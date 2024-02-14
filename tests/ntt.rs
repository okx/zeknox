use std::ops::Mul;

use cryptography_cuda::{device::memory::HostOrDeviceSlice, device::stream::CudaStream};
use cryptography_cuda::{
    get_number_of_gpus_rs, init_coset_rs, init_twiddle_factors_rs, intt, intt_batch, lde_batch,
    ntt, ntt_batch, types::*,
};
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialValues;
use plonky2_field::{
    fft::{fft, ifft},
    polynomial::PolynomialCoeffs,
    types::{Field, PrimeField64},
};
use rand::random;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

const DEFAULT_GPU: usize = 0;

#[test]
fn test_ntt_intt_gl64_self_consistency() {
    for lg_domain_size in 1..19 {
        let domain_size = 1usize << lg_domain_size;

        let v: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

        let mut gpu_buffer = v.clone();

        ntt(DEFAULT_GPU, &mut gpu_buffer, NTTInputOutputOrder::NN);

        intt(DEFAULT_GPU, &mut gpu_buffer, NTTInputOutputOrder::NN);

        assert_eq!(v, gpu_buffer);
    }
}

#[test]
fn test_ntt_gl64_consistency_with_plonky2() {
    for lg_domain_size in 1..20 {
        let domain_size = 1usize << lg_domain_size;

        let v: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
        let mut gpu_buffer = v.clone();
        ntt(DEFAULT_GPU, &mut gpu_buffer, NTTInputOutputOrder::NN);

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
        intt(DEFAULT_GPU, &mut gpu_buffer, NTTInputOutputOrder::NN);

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
    init_twiddle_factors_rs(DEFAULT_GPU, lg_domain_size);
    let domain_size = 1usize << lg_domain_size;

    let v1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let v2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut gpu_buffer = v1.clone();
    gpu_buffer.extend(v2.iter());

    let mut cfg = NTTConfig::default();
    cfg.batches = 2;
    ntt_batch(DEFAULT_GPU, gpu_buffer.as_mut_ptr(), lg_domain_size, cfg);

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
    init_twiddle_factors_rs(DEFAULT_GPU, lg_domain_size);
    let domain_size = 1usize << lg_domain_size;

    let v1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let v2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut gpu_buffer = v1.clone();

    let mut cfg = NTTConfig::default();
    ntt_batch(
        DEFAULT_GPU,
        gpu_buffer.as_mut_ptr(),
        lg_domain_size,
        cfg.clone(),
    );

    intt_batch(
        DEFAULT_GPU,
        gpu_buffer.as_mut_ptr(),
        lg_domain_size,
        cfg.clone(),
    );
    assert_eq!(v1, gpu_buffer);
}

#[test]
fn test_intt_batch_gl64_consistency_with_plonky2() {
    let lg_domain_size: usize = 4;
    init_twiddle_factors_rs(DEFAULT_GPU, lg_domain_size);

    let batches = 2;
    let domain_size = 1usize << lg_domain_size;

    let input1: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let input2: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();

    let mut gpu_buffer = input1.clone();
    gpu_buffer.extend(input2.iter());

    let mut cfg = NTTConfig::default();
    cfg.batches = batches;
    intt_batch(DEFAULT_GPU, gpu_buffer.as_mut_ptr(), lg_domain_size, cfg);

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
        HostOrDeviceSlice::cuda_malloc(domain_size).unwrap();
    let ret = device_data.copy_from_host(&scalars);

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
    let lg_domain_size = 4;
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
        HostOrDeviceSlice::cuda_malloc(total_elements).unwrap();
    device_data.copy_from_host_offset(input1.as_mut_slice(), 0, domain_size);
    device_data.copy_from_host_offset(input2.as_mut_slice(), domain_size, domain_size);
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
fn test_ntt_batch_with_coset() {
    let lg_domain_size = 4;
    let domain_size = 1usize << lg_domain_size;
    // let batches = 2;

    init_twiddle_factors_rs(DEFAULT_GPU, lg_domain_size);
    init_coset_rs(
        DEFAULT_GPU,
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
        DEFAULT_GPU,
        gpu_buffer.as_mut_ptr(),
        lg_domain_size,
        cfg.clone(),
    );

    let mut cpu_buffer = v1.clone();

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

    let mut cpu_buffer2 = v2.clone();
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
    let lg_n: usize = 2;
    let rate_bits = 2;
    let lg_domain_size = lg_n + rate_bits;
    init_twiddle_factors_rs(DEFAULT_GPU, lg_domain_size);
    init_coset_rs(DEFAULT_GPU, lg_domain_size, GoldilocksField::coset_shift().to_canonical_u64());

    let input_size = 1usize << lg_n;
    let mut input: Vec<u64> = (0..input_size).map(|_| random_fr()).collect();

    let cpu_buffer = input.clone();
    let cpu_coeffs = cpu_buffer
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let cpu_poly: PolynomialCoeffs<GoldilocksField> = PolynomialCoeffs { coeffs: cpu_coeffs };
    // println!("poly: {:?}", cpu_poly);
    let poly_extend = cpu_poly.lde(rate_bits);
    // println!("extended: {:?}", poly_extend);
    let mut modified_poly: PolynomialCoeffs<GoldilocksField> = GoldilocksField::coset_shift()
        .powers()
        .zip(&poly_extend.coeffs)
        .map(|(r, &c)| r * c)
        .collect::<Vec<_>>()
        .into();
    // println!("modified_poly: {:?}", modified_poly);

    let mut gpu_buffer = modified_poly.coeffs.iter().map(|x| x.to_canonical_u64()).collect::<Vec<u64>>();
    let mut cfg_ntt = NTTConfig::default();
    // println!("gpu ntt input: {:?}", gpu_buffer);
    ntt_batch(DEFAULT_GPU, gpu_buffer.as_mut_ptr(), lg_domain_size, cfg_ntt);

    // println!("gpu ntt output: {:?}", gpu_buffer);

    let mut cfg_lde = NTTConfig::default();
    cfg_lde.batches = 1 as u32;
    // cfg_lde.are_outputs_on_device = true;
    cfg_lde.extension_rate_bits = rate_bits as u32;
    cfg_lde.with_coset = true;

    let mut gpu_lde_buffer = input.clone();
    let mut gpu_lde_output = vec![0; 1 << lg_domain_size];
    lde_batch(
        DEFAULT_GPU,
        gpu_lde_output.as_mut_ptr(),
        gpu_lde_buffer.as_mut_ptr(),
        lg_n,
        cfg_lde,
    );
    // println!("gpu lde output, len: {:?}, {:?}", gpu_lde_output.len(), gpu_lde_output);
    assert_eq!(gpu_lde_output, gpu_buffer);
}
