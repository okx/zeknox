extern crate criterion;
use criterion::{criterion_group, criterion_main, Criterion};
use cryptography_cuda::{device::memory::HostOrDeviceSlice, init_coset_rs, init_twiddle_factors_rs, lde_batch_multi_gpu, ntt, types::{NTTConfig, NTTInputOutputOrder}};
use plonky2_field::{goldilocks_field::GoldilocksField, types::{Field, PrimeField64}};
use rand::random;

const DEFAULT_GPU: i32 = 0;
const DEFAULT_GPU2: i32 = 1;
const DEFAULT_GPU3: i32 = 2;
const DEFAULT_GPU4: i32 = 3;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn init_twiddle_gpu(lg_domain_size: usize){
    init_twiddle_factors_rs(DEFAULT_GPU as usize, lg_domain_size);
    init_twiddle_factors_rs(DEFAULT_GPU2 as usize, lg_domain_size);
    init_twiddle_factors_rs(DEFAULT_GPU3 as usize, lg_domain_size);
    init_twiddle_factors_rs(DEFAULT_GPU4 as usize, lg_domain_size);
}

fn init_coset(){
    init_coset_rs(
        DEFAULT_GPU as usize,
        23,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );
    init_coset_rs(
        DEFAULT_GPU2 as usize,
        23,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );
    init_coset_rs(
        DEFAULT_GPU3 as usize,
        23,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );
    init_coset_rs(
        DEFAULT_GPU4 as usize,
        23,
        GoldilocksField::coset_shift().to_canonical_u64(),
    );
}

fn bench_multi_gpu_lde_batch(c: &mut Criterion) {
    let mut LOG_N_SIZES: Vec<usize> = (16..=19).collect();
    let RATE_BITS = 3;
    let mut group = c.benchmark_group("LDE");
    init_coset();

    for log_n_sizes in LOG_N_SIZES {
        let batches = 50;
        let input_domain_size = 1usize << log_n_sizes;
        let output_domain_size =  1usize << (log_n_sizes + RATE_BITS);

        init_twiddle_gpu(log_n_sizes + RATE_BITS);

        let total_num_input_elements = input_domain_size * batches;
        let total_num_output_elements = output_domain_size * batches;

        let mut cfg_lde = NTTConfig::default();
        cfg_lde.batches = batches as u32;
        cfg_lde.extension_rate_bits = RATE_BITS as u32;
        cfg_lde.with_coset = true;
        cfg_lde.are_inputs_on_device = false;
        cfg_lde.are_outputs_on_device = true;
        cfg_lde.is_multi_gpu = true;

        let mut gpu_inputs: Vec<u64> = (0..batches).collect::<Vec<usize>>()
            .iter()
            .map(|_| (0..input_domain_size).map(|_| random_fr()).collect::<Vec<u64>>())
            .flatten().collect();

        let mut device_output_data: HostOrDeviceSlice<'_, u64> =
            HostOrDeviceSlice::cuda_malloc(DEFAULT_GPU, total_num_output_elements).unwrap();

        group.sample_size(10).bench_function(
            &format!("Multi gpu lde on 4 gpus with lg_n size of 2^{}", log_n_sizes),
            |b| b.iter(||  lde_batch_multi_gpu(
                device_output_data.as_mut_ptr(),
                gpu_inputs.as_mut_ptr(),
                4,    
                cfg_lde.clone(),
                log_n_sizes, 
                total_num_input_elements,
                total_num_output_elements,
            )),
        );

        group.sample_size(10).bench_function(
            &format!("Multi gpu lde on 1 gpu1 with lg_n size of 2^{}", log_n_sizes),
            |b| b.iter(||  lde_batch_multi_gpu(
                device_output_data.as_mut_ptr(),
                gpu_inputs.as_mut_ptr(),
                1,    
                cfg_lde.clone(),
                log_n_sizes, 
                total_num_input_elements,
                total_num_output_elements,
            )),
        );
    }
}

criterion_group!(benches, bench_multi_gpu_lde_batch);
criterion_main!(benches);
