#[cfg(not(feature="no_cuda"))]
use std::env;
#[cfg(not(feature="no_cuda"))]
use std::fs;
#[cfg(not(feature="no_cuda"))]
use std::path::PathBuf;
#[cfg(not(feature="no_cuda"))]
extern crate rustacuda;
#[cfg(not(feature="no_cuda"))]
use rustacuda::device::DeviceAttribute;
#[cfg(not(feature="no_cuda"))]
use rustacuda::prelude::*;

// based on: https://github.com/matter-labs/z-prize-msm-gpu/blob/main/bellman-cuda-rust/cudart-sys/build.rs
#[cfg(not(feature="no_cuda"))]
fn build_device_wrapper() {
    let cuda_runtime_api_path = PathBuf::from("/usr/local/cuda/include")
        .join("cuda_runtime_api.h")
        .to_string_lossy()
        .to_string();
    println!("cargo:rustc-link-search=native={}",  "/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed={}", cuda_runtime_api_path);

    let bindings = bindgen::Builder::default()
    .header(cuda_runtime_api_path)
    .size_t_is_usize(true)
    .generate_comments(false)
    .layout_tests(false)
    .allowlist_type("cudaError")
    .rustified_enum("cudaError")
    .must_use_type("cudaError")
    // device management
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html
    .allowlist_function("cudaSetDevice")
    // error handling
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
    .allowlist_function("cudaGetLastError")
    // stream management
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
    .allowlist_function("cudaStreamCreate")
    .allowlist_var("cudaStreamDefault")
    .allowlist_var("cudaStreamNonBlocking")
    .allowlist_function("cudaStreamCreateWithFlags")
    .allowlist_function("cudaStreamDestroy")
    .allowlist_function("cudaStreamQuery")
    .allowlist_function("cudaStreamSynchronize")
    .allowlist_var("cudaEventWaitDefault")
    .allowlist_var("cudaEventWaitExternal")
    .allowlist_function("cudaStreamWaitEvent")
    // memory management
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
    .allowlist_function("cudaFree")
    .allowlist_function("cudaMalloc")
    .allowlist_function("cudaMemcpy")
    .allowlist_function("cudaMemcpyAsync")
    .allowlist_function("cudaMemset")
    .allowlist_function("cudaMemsetAsync")
    .rustified_enum("cudaMemcpyKind")
    // Stream Ordered Memory Allocator
    // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html
    .allowlist_function("cudaFreeAsync")
    .allowlist_function("cudaMallocAsync")
    //
    .allowlist_function("cudaMemGetInfo")
    //
    .generate()
    .expect("Unable to generate bindings");

    fs::write(PathBuf::from("src/device").join("bindings.rs"), bindings.to_string()).expect("Couldn't write bindings!");
}

#[cfg(not(feature="no_cuda"))]
fn get_device_arch() -> String {
    rustacuda::init(CudaFlags::empty()).expect("unable to init");

    let device = Device::get_device(0).expect("error in get device");
    let compute_major = device.get_attribute(DeviceAttribute::ComputeCapabilityMajor).unwrap();
    let compute_minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor).unwrap();
    let cuda_arch = format!("sm_{}{}", compute_major, compute_minor);
    println!("cuda arch is: {:?}", cuda_arch);
    cuda_arch
}

#[cfg(not(feature="no_cuda"))]
fn feature_check() -> String {
    let fr_s = [
        "gl64",
        "bn254"
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
            } else if cfg!(feature = "bn254") {
                fr = "FEATURE_BN254"
            }
            String::from(fr)
        },
        _ => panic!("Multiple fields are not supported, please select only one."),
    }
}

#[cfg(not(feature="no_cuda"))]
fn build_cuda() {
    if cfg!(target_os = "windows") && !cfg!(target_env = "msvc") {
        panic!("unsupported compiler");
    }

    build_device_wrapper();

    let fr = feature_check();
    println!("feature: {:?}", fr);

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let base_dir = manifest_dir.join("cuda");

    // pass DEP_CRYPTOGRAPHY_CUDA_* variables to dependents
    println!("cargo:ROOT={}", base_dir.to_string_lossy());

    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    let nvcc = match env::var("NVCC") {
        Ok(var) => {
            println!("nvcc env var: {:?}", var);
            which::which(var)
        },
        Err(_) => {
            println!("no nvcc env var set use nvcc");
            which::which("nvcc")
        },
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
        nvcc.define("EXPOSE_C_INTERFACE", None);
        if cfg!(feature = "gl64") {
            nvcc.define("GL64_NO_REDUCTION_KLUDGE", None);
        }

        nvcc.include(base_dir);
        // required for parling curve, such as bls12_381, bn254, etc.
        // cargo dependency blst will set DEP_BLST_C_SRC
        if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
            println!("blst_c_src directory: {:?}", include); // ~/.cargo/registry/src/index.crates.io-6f17d22bba15001f/blst-0.3.11/blst/src
            nvcc.include(include);
        }

        nvcc.file("src/lib.cu")
            .file(util_dir.join("all_gpus.cpp"))
            .compile("cryptography_cuda");

        println!("cargo:rerun-if-changed=src/lib.cu");
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    } else {
        panic!("no nvcc found");
    }
    println!("cargo:rerun-if-env-changed=NVCC");
}

#[cfg(not(feature="no_cuda"))]
fn build_lib() {
    use std::process::Command;

    let pwd = env::current_dir().unwrap();
    let libdir = pwd.join("cuda");
    let header_file = libdir.join("merkle/merkle.h");
    let src_file = libdir.join("merkle/merkle.cu");
    let lib_file = libdir.join("libcryptocuda.a");

    if !lib_file.exists()
    {
        assert!(env::set_current_dir(&libdir).is_ok());
        Command::new("make")
        .arg("lib")
        .output()
        .expect("failed to execute process");
        assert!(env::set_current_dir(&pwd).is_ok());
    }

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}", libdir.to_str().unwrap());

    // Shared lib.
    // println!("cargo:rustc-link-lib=cryptocuda");

    // Static lib
    println!("cargo:rustc-link-lib=static=cryptocuda");
    println!("cargo:rustc-link-lib=gomp");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed={},{}", header_file.to_str().unwrap(), src_file.to_str().unwrap());
}

fn main() {
    #[cfg(not(feature="no_cuda"))]
    build_cuda();
    #[cfg(not(feature="no_cuda"))]
    build_lib();
}
