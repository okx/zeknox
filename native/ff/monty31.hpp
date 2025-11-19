#ifndef __MONTY_HPP__
#define __MONTY_HPP__

#include "utils/cuda_utils.cuh"
#include "types/int_types.h"

template <typename MP>
DEVICE INLINE u32 monty_reduce(u64 x)
{
    u64 t = (x * (u64)MP::MONTY_MU) & (u64)MP::MONTY_MASK;
    u64 u = t * (u64)MP::PRIME;

    u64 over = (u > x);
    u64 x_sub_u = x - u;
    u32 x_sub_u_hi = (u32)(x_sub_u >> MP::MONTY_BITS);
    u32 corr = over ? MP::PRIME : 0;
    return x_sub_u_hi + corr;
}

template <typename MP>
class MontyField31
{
private:
    u32 val;

public:
    DEVICE MontyField31() { this->val = 0; };

    DEVICE MontyField31(u32 x, bool is_monty = false) {
        if (is_monty)
            this->val = x;
        else
            this->val = to_monty(x);
    };

    static DEVICE INLINE u32 to_monty(u32 x)
    {
        return (u32)(((u64)x << MP::MONTY_BITS) % (u64)MP::PRIME);
    }

    static DEVICE INLINE u32 from_monty(u32 x)
    {
        return monty_reduce<MP>((u64)x);
    }

    static DEVICE INLINE MontyField31 Zero()
    {
        return MontyField31(0);
    }

    DEVICE INLINE u32 value() const
    {
        return this->val;
    }

    DEVICE INLINE u64 to_u64() const
    {
        return (u64)from_monty(val);
    }

    DEVICE MontyField31 operator+(const MontyField31 &rhs)
    {
        u64 sum = (u64)val + (u64)rhs.value();
        u32 v = sum < (u64)MP::PRIME ? (u32)sum : (u32)(sum - MP::PRIME);
        return MontyField31(v, true);
    }

    DEVICE MontyField31 operator+=(const MontyField31 &rhs)
    {
        u64 sum = (u64)val + (u64)rhs.value();
        this->val = sum < (u64)MP::PRIME ? (u32)sum : (u32)(sum - MP::PRIME);
        return *this;
    }

    DEVICE MontyField31 operator-(const MontyField31 &rhs)
    {
        u32 over = val < rhs.value();
        u32 diff = over ? (u32)((u64)val + (u64)MP::PRIME - rhs.value()) : val - rhs.value();
        return MontyField31(diff, true);
    }

    DEVICE MontyField31 operator-=(const MontyField31 &rhs)
    {
        u32 over = val < rhs.value();
        this->diff = over ? (u32)((u64)val + (u64)MP::PRIME - rhs.value()) : val - rhs.value();
        return *this;
    }

    DEVICE MontyField31 operator*(const MontyField31 &rhs)
    {
        u64 prod = (u64)val * (u64)rhs.value();
        return MontyField31(monty_reduce<MP>(prod), true);
    }

    DEVICE MontyField31 operator*=(const MontyField31 &rhs)
    {
        u64 prod = (u64)val * (u64)rhs.value();
        this-> val = monty_reduce<MP>(prod);
        return *this;
    }
};

class BabyBearParameters
{
public:
    // The Baby Bear prime: 2^31 - 2^27 + 1.
    // This is the unique 31-bit prime with the highest possible 2 adicity (27).
    static const u32 PRIME = 0x78000001; // 2013265921
    static const u32 MONTY_BITS = 32;
    static const u32 MONTY_MU = 0x88000001;
    static const u32 MONTY_MASK = 0xFFFFFFFF;
};

#endif // __MONTY_HPP__