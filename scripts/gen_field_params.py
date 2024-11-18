"""
Copyright 2024 OKX Group
Licensed under the Apache License, Version 2.0, see LICENSE for details.
SPDX-License-Identifier: Apache-2.0
"""

import json
import sys

def to_hex(val, length):
    return "0x{:016x}".format(val)

def inv(a, modulus):
    """
    Modular inverse using eGCD algorithm
    """
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % modulus, modulus
    while low > 1:
        r = high // low
        nm, new = hm - lm * r, high - low * r
        lm, low, hm, high = nm, new, lm, low
    return lm % modulus

def gen_omegas(name, size, root_of_unity, modulus, generator):
    limb_size = 8
    if size > 0:
        omegas = []
        omegas_inv = []
        for k in range(size):
            if k == 0:
                om = root_of_unity
            else:
                om = pow(om, 2, modulus)
            omegas.append(om)
            omegas_inv.append(pow(om, -1, modulus))
        omegas.reverse()
        omegas_inv.reverse()

        omega = "static constexpr uint64_t OMEGA[{:d}] = {{\n".format(size + 1)
        omega_inv = "static constexpr uint64_t OMEGA_INV[{}] = {{\n".format(size + 1)
        inv_logs = "static constexpr uint64_t DOMAIN_SIZE_INV[{}] = {{\n".format(size + 1)

        omega += "\t" + to_hex(1,limb_size) + ",\n"
        omega_inv += "\t" + to_hex(1,limb_size) + ",\n"
        inv_logs += "\t" + to_hex(1,limb_size) + ",\n"

        for k in range(size):
            omega += "\t" + to_hex(omegas[k],limb_size) + ",\n"
            omega_inv += "\t" + to_hex(omegas_inv[k],limb_size) + ",\n"
            inv_logs += "\t" + to_hex(pow(int(pow(2,k+1)), -1, modulus),limb_size) + ",\n"

        omega += "};\n\n"
        omega_inv += "};\n\n"
        inv_logs += "};\n\n"

    with open("../native/ff/{}_params.hpp".format(name), "w") as f:
        f.write("// Copyright 2024 OKX Group\n")
        f.write("// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n")
        f.write("// SPDX-License-Identifier: Apache-2.0\n\n")
        f.write("/* This file is generated by gen_field_params.py */\n\n")
        f.write("#ifndef __PARAMS_{}_HPP__\n".format(name.upper()))
        f.write("#define __PARAMS_{}_HPP__\n\n".format(name.upper()))
        f.write("#include <cstdint>\n\n")
        f.write("const uint64_t GROUP_GENERATOR = {};\n".format(to_hex(generator, limb_size)))
        f.write("const uint64_t GROUP_GENERATOR_INV = {};\n\n".format(to_hex(inv(generator, modulus), limb_size)))
        f.write(omega)
        f.write(omega_inv)
        f.write(inv_logs)
        f.write("#endif // __PARAMS_{}_HPP__\n".format(name.upper()))
        f.close()
        print("Generated params_{}.hpp".format(name))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gen_field_params.py <config.json>")
        exit(1)

    config = None
    with open(sys.argv[1]) as json_file:
        config = json.load(json_file)

    gen_omegas(config["curve_name"], int(config["ntt_size"]), int(config["root_of_unity"]), int(config["modulus_p"]), int(config["group_gen"]))