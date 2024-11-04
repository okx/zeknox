"""
Copyright 2024 OKX Group
Licensed under the Apache License, Version 2.0, see LICENSE for details.
SPDX-License-Identifier: Apache-2.0
"""

import os
# import blst
import itertools
from time import time
# import numpy as np
# from fft import fft, expand_root_of_unity
import random
from typing import Tuple, List

from arithmetic import (
    get_root_of_unity,
    count_trailing_zeros,
    div
)

import csv
from curve_parameters import (
    MODULUS,
    # P1_INF,
    PRIMITIVE_ROOT
)



# 8x reduced size for testing, for full set use n = 512 and m = 4096
n = 1 << 0
m = 1 << 3
MAX_DEGREE_POLY = MODULUS-1

# def test_fft():
#     # random input
#     D_in = [[random.randint(1, MAX_DEGREE_POLY) for _ in range(m)] for _ in range(n)]

#     # saved input
#     with open(f'test_data/D_in_{n}_{m}.csv') as file:
#         reader = csv.reader(file)
#         D_in = [[int(i, 16) for i in row] for row in reader]
#     start = time()
#     # Coefficient form
#     C_rows = [fft(D_in[i], MODULUS, get_root_of_unity(m), inv=False)
#               for i in range(n)]
#     print(C_rows)
#     for x in C_rows[0]:
#         print(hex(x))
#     end = time()
#     print((end - start) * 1000)
#     # print(hex(get_root_of_unity(m)))

def get_omega(rou:int, k):
    return pow(rou, 2**(32-k), MODULUS)

if __name__ == "__main__":
    # test_fft()
    rou = get_root_of_unity(2**32)
    print(rou)
    # omega1=get_omega(rou, 2)
    # print(f"omega_2 {count_trailing_zeros(42)}")
    # print(1<<32)
    # out = div(MODULUS, 1<<32)
    # # out2 = pow(PRIMITIVE_ROOT, out, MODULUS)
    # print(out)

