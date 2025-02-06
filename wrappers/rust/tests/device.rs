// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[warn(dead_code)]
use zeknox_p2::{get_number_of_gpus_rs, list_devices_info_rs};

/// example output
/// Device 0 - Tesla V100-SXM2-16GB
/// CUDA multi processor count: 80   CUDA Cores: 5120
#[test]
fn test_list_devices_info_rs() {
    list_devices_info_rs()
}

#[test]
fn test_get_number_of_gpus() {
    let _ = get_number_of_gpus_rs();
}
