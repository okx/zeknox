// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef _HASHER_CUH_
#define _HASHER_CUH_

#include <stdint.h>
#include <cstring>

#include "types/int_types.h"

#ifdef USE_CUDA
#include "ff/goldilocks.hpp"
#else
#include "poseidon/goldilocks.hpp"
#endif

class Hasher {};

#define NUM_HASH_OUT_ELTS 4

#endif // _HASHER_CUH_
