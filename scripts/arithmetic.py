import csv
from curve_parameters import (
    MODULUS,
    PRIMITIVE_ROOT,
)

def count_trailing_zeros(x):
    return (x & -x).bit_length() - 1

def inverse_goldilock(v, MODULUS_GOLDILOCK):
    (u, v) = (MODULUS_GOLDILOCK, v)
    (t0, t1) = (0, 1)
    twos = count_trailing_zeros(v)
    print(f"twos: {twos}")
    v >>= twos
    print(f"v: {v}")
    twos += 96 # 2^96 = -1
    while u != v:
        u -= v
        t0 += t1
        count = count_trailing_zeros(u)
        u >>= count
        t1 <<= count
        twos += count
        if u < v:
            (u, v) = (v, u)
            (t0, t1) = (t1, t0)
            twos += 96 # 2^96 = -1
    print(f"t0: {t0}")
    print(f"left_shift: {(191 * twos) % 192}")
    return t0 << ((191 * twos) % 192) # 2^-1 = 2^191, 2^192 = 1

def get_root_of_unity(order: int) -> int:
    """
    Returns a root of unity of order "order"
    """
    assert (MODULUS - 1) % order == 0
    return pow(PRIMITIVE_ROOT, (MODULUS - 1) // order, MODULUS)


def inv(a):
    """
    Modular inverse using eGCD algorithm
    """
    if a == 0:
        return 0
    lm, hm = 1, 0
    low, high = a % MODULUS, MODULUS
    while low > 1:
        r = high // low
        nm, new = hm - lm * r, high - low * r
        lm, low, hm, high = nm, new, lm, low
    return lm % MODULUS

def div(x, y):
    return x * inv(y) % MODULUS


if __name__ == "__main__":
    # test_fft()
    # rou = get_root_of_unity(2**32)
    # print(rou)
    # a = 0x185629dcda58878c
    # print(a)
    # print(f"inverse_goldilock {hex(get_root_of_unity(1<<31))}")
    inv_a = inv(1<<256)
    print(inv_a)