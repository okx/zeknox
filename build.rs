
use std::env;
use std::path::PathBuf;
#[macro_use]
extern crate rustacuda;

use rustacuda::device::DeviceAttribute;
use rustacuda::prelude::*;

fn get_device_arch() -> String {
    rustacuda::init(CudaFlags::empty()).expect("unable to init");

    let device = Device::get_device(0).expect("error in get device");
    let compute_major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor).unwrap();
    let compute_minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor).unwrap();
    let cuda_arch = format!("sm_{}{}", compute_major, compute_minor);
    println!("cuda arch is: {:?}", cuda_arch);
    cuda_arch
}

fn feature_check() -> String {
    let fr_s = [
        "gl64",
    ];
    let fr_s_as_features: Vec<String> = (0..fr_s.len())
        .map(|i| format!("CARGO_FEATURE_{}", fr_s[i].to_uppercase()))
        .collect();

    let mut fr_counter = 0;
    for fr_feature in fr_s_as_features.iter() {
        fr_counter += env::var(&fr_feature).is_ok() as i32;
    }

    match fr_counter {
        0 => String::from("FEATURE_GOLDILOCKS"),  // use gl64 as default
        1 => {
            let mut fr = "";
            if cfg!(feature = "gl64") {
                fr = "FEATURE_GOLDILOCKS";
            }
            String::from(fr)
        },
        _ => panic!("Multiple fields are not supported, please select only one."),
    }
}

fn main() {
    if cfg!(target_os = "windows") && !cfg!(target_env = "msvc") {
        panic!("unsupported compiler");
    }

    let fr = feature_check();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut base_dir = manifest_dir.join("cuda");

    // pass DEP_CRYPTOGRAPHY_CUDA_* variables to dependents
    println!("cargo:ROOT={}", base_dir.to_string_lossy());

    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };
    if nvcc.is_ok() {
        let cuda_version = std::process::Command::new(nvcc.unwrap())
            .arg("--version")
            .output()
            .expect("impossible");
        if !cuda_version.status.success() {
            panic!("{:?}", cuda_version);
        }
        let cuda_version = String::from_utf8(cuda_version.stdout).unwrap();
        let x = cuda_version
            .find("release ")
            .expect("can't find \"release X.Y,\" in --version output")
            + 8;
        let y = cuda_version[x..]
            .find(",")
            .expect("can't parse \"release X.Y,\" in --version output");
        let v = cuda_version[x..x + y].parse::<f32>().unwrap();
        if v < 11.4 {
            panic!("Unsupported CUDA version {} < 11.4", v);
        }

        let util_dir = base_dir.join("util");
        let mut nvcc = cc::Build::new();
        let cuda_arch = get_device_arch();
        nvcc.cuda(true);
        nvcc.flag(&format!("-arch={}", cuda_arch));  // check [Virtual Architectures](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architectures)
        #[cfg(not(target_env = "msvc"))]
        // nvcc.flag("-Xcompiler").flag("-Wno-unused-function");
        nvcc.define("TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE", None);
        nvcc.define(&fr, None);
        if cfg!(feature = "gl64") {
            nvcc.define("GL64_NO_REDUCTION_KLUDGE", None);
        }
        nvcc.include(base_dir);
        nvcc.file("src/lib.cu")
            .file(util_dir.join("all_gpus.cpp"))
            .compile("cryptography_cuda");
        println!("cargo:rerun-if-changed=src/lib.cu");
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }
    println!("cargo:rerun-if-env-changed=NVCC");
}
