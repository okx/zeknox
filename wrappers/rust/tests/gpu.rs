// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use cryptography_cuda::get_number_of_gpus_rs;

#[test]
fn test_get_number_of_gpus() {
    let _ = get_number_of_gpus_rs();
}
