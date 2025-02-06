// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[cfg(not(feature = "no_cuda"))]
extern crate rustacuda;

#[cfg(not(feature = "no_cuda"))]
fn build_lib() {
    println!("cargo:rustc-link-search={}", "/usr/local/lib");
    println!("cargo:rustc-link-search=native={}", "/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=static=zeknox");
    println!("cargo:rustc-link-lib=gomp");
}

fn main() {
    #[cfg(not(feature = "no_cuda"))]
    build_lib();
}
