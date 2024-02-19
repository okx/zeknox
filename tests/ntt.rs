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
fn test_ntt_batch_transposed_on_device() {
    let lg_domain_size = 12;
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
    cfg.are_outputs_transposed = true;
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

    let cpu_results_matrix = [cpu_results1, cpu_results2];
    let cpu_results_tranposed = transpose(&cpu_results_matrix);
    let cpu_results: Vec<u64> = cpu_results_tranposed.into_iter().flatten().collect();
    assert_eq!(host_output, cpu_results);

}

pub fn transpose<T: Send + Sync + Copy>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    let len = matrix[0].len();
    (0..len)
        .into_iter()
        .map(|i| matrix.iter().map(|row| row[i]).collect())
        .collect()
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
    let batches = 2;
    init_twiddle_factors_rs(DEFAULT_GPU, lg_domain_size);
    init_coset_rs(
        DEFAULT_GPU,
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
    cfg.are_outputs_transposed = true;
    cfg.batches = batches as u32;
    // println!("ntt config {:?}", cfg);
    ntt_batch(
        DEFAULT_GPU,
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
    cfg_lde.are_outputs_transposed = true;

    let mut gpu_lde_output = vec![0; (1 << lg_domain_size) * batches];
    lde_batch(
        DEFAULT_GPU,
        gpu_lde_output.as_mut_ptr(),
        gpu_buffer.as_mut_ptr(),
        lg_n,
        cfg_lde,
    );
    assert_eq!(cpu_outputs, gpu_lde_output);
}

// todo
#[test]
fn test_compute_batched_lde_data_on_device() {
    let lg_n: usize = 2;
    let rate_bits = 2;
    let lg_domain_size = lg_n + rate_bits;
    let input_domain_size = 1usize << lg_n;
    let output_domain_size = 1usize << (lg_n + rate_bits);
    let batches = 2;

    init_twiddle_factors_rs(DEFAULT_GPU, lg_domain_size);
    init_coset_rs(
        DEFAULT_GPU,
        lg_domain_size,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );

    let total_num_input_elements = input_domain_size * batches;
    let total_num_output_elements = output_domain_size * batches;

    let mut input1: Vec<u64> = (0..input_domain_size).map(|_| random_fr()).collect();
    let mut input2: Vec<u64> = (0..input_domain_size).map(|_| random_fr()).collect();


    let mut device_input_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc((total_num_input_elements)).unwrap();
    device_input_data.copy_from_host_offset(input1.as_mut_slice(), 0, input_domain_size);
    device_input_data.copy_from_host_offset(
        input2.as_mut_slice(),
        input_domain_size,
        input_domain_size,
    );

    let mut device_output_data: HostOrDeviceSlice<'_, u64> =
        HostOrDeviceSlice::cuda_malloc(total_num_output_elements).unwrap();


    let mut cfg_lde = NTTConfig::default();
    cfg_lde.batches = batches as u32;
    cfg_lde.extension_rate_bits = rate_bits as u32;
    cfg_lde.with_coset = true;
    cfg_lde.are_inputs_on_device = true;
    cfg_lde.are_outputs_on_device = true;

    lde_batch(
        DEFAULT_GPU,
        device_output_data.as_mut_ptr(),
        device_input_data.as_mut_ptr(),
        lg_n,
        cfg_lde,
    );

    let mut host_output1 = vec![0; output_domain_size];
    let mut host_output2 = vec![0; output_domain_size];

    device_output_data.copy_to_host_offset(host_output1.as_mut_slice(), 0, output_domain_size);
    device_output_data.copy_to_host_offset(host_output2.as_mut_slice() , output_domain_size, output_domain_size);

    // println!("host_output1: {:?}", host_output1);
    // println!("host_output2: {:?}", host_output2);

    // lde copy data
    let mut cfg_lde_copy = NTTConfig::default();
    cfg_lde_copy.batches = batches as u32;
    cfg_lde_copy.extension_rate_bits = rate_bits as u32;
    cfg_lde_copy.with_coset = true;

    let mut gpu_lde_output_copy = vec![0; total_num_output_elements];
    let mut lde_copy_buffer = input1.clone();
    lde_copy_buffer.extend(input2.iter());
    lde_batch(
        DEFAULT_GPU,
        gpu_lde_output_copy.as_mut_ptr(),
        lde_copy_buffer.as_mut_ptr(),
        lg_n,
        cfg_lde_copy,
    );
    // println!("gpu_lde_output_copy: {:?}", gpu_lde_output_copy);
    assert_eq!(gpu_lde_output_copy[0..output_domain_size], host_output1);
    assert_eq!(gpu_lde_output_copy[output_domain_size..output_domain_size*2], host_output2);

}
