#[cfg(not(feature = "no_cuda"))]
use std::env;
#[cfg(not(feature = "no_cuda"))]
use std::fs;
#[cfg(not(feature = "no_cuda"))]
use std::path::PathBuf;
#[cfg(not(feature = "no_cuda"))]
extern crate rustacuda;

// based on: https://github.com/matter-labs/z-prize-msm-gpu/blob/main/bellman-cuda-rust/cudart-sys/build.rs
#[cfg(not(feature = "no_cuda"))]
fn build_device_wrapper() {
    let cuda_runtime_api_path = PathBuf::from("/usr/local/cuda/include")
        .join("cuda_runtime_api.h")
        .to_string_lossy()
        .to_string();
    println!("cargo:rustc-link-search=native={}", "/usr/local/cuda/lib64");
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

    fs::write(
        PathBuf::from("src/device").join("bindings.rs"),
        bindings.to_string(),
    )
    .expect("Couldn't write bindings!");
}

#[cfg(not(feature = "no_cuda"))]
fn build_lib() {
    use std::process::Command;

    let rootdirstr = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let rootdir = PathBuf::from(rootdirstr);
    let parent = rootdir.parent().unwrap().parent().unwrap();
    let srcdir = parent.join("native");
    let libdir = srcdir.join("build");
    let libfile = libdir.join("libcryptocuda.a");

    if !libfile.exists() {
        assert!(env::set_current_dir(&srcdir).is_ok());
        Command::new("rm")
            .args(["-r", "-f", "build"])
            .output()
            .expect("failed to execute process");
        Command::new("mkdir")
            .arg("build")
            .output()
            .expect("failed to execute process");
        assert!(env::set_current_dir(&libdir).is_ok());
        Command::new("cmake")
            .arg("..")
            .output()
            .expect("failed to execute process");
        Command::new("make")
            .output()
            .expect("failed to execute process");
        assert!(env::set_current_dir(&rootdir).is_ok());

        println!("{:?}", libdir);
    }

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}", libdir.to_str().unwrap());

    // Static lib
    println!("cargo:rustc-link-search=native={}", "/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=static=cryptocuda");
    println!("cargo:rustc-link-lib=gomp");
}

fn main() {
    #[cfg(not(feature = "no_cuda"))]
    build_device_wrapper();
    #[cfg(not(feature = "no_cuda"))]
    build_lib();
}
