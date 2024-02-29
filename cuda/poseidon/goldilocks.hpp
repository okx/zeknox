#ifndef __GOLDILOCKS_H__
#define __GOLDILOCKS_H__

#include "int_types.h"

#ifdef TESTING
#include <stdio.h>
#endif

class GoldilocksField
{
private:
    u64 val;

public:
    static const u64 ORDER = 0xFFFFFFFF00000001ul; // 18446744069414584321

    static const u64 EPSILON = 4294967295; // ((u64)1 << 32) - 1 = 2^64 % ORDER

    static const u64 EPSILON2 = 18446744069414584320ul; // 2^96 % ORDER

    GoldilocksField()
    {
        this->val = 0;
    }

    GoldilocksField(u64 val)
    {
        assert(val < ORDER);
        this->val = val;
    }

    static GoldilocksField Zero()
    {
        return GoldilocksField(0);
    };

    static GoldilocksField One()
    {
        return GoldilocksField(1);
    }

    static GoldilocksField Two()
    {
        return GoldilocksField(2);
    }

    static GoldilocksField NegOne()
    {
        return GoldilocksField(ORDER - 1);
    };

    u64 get_val() const
    {
        return this->val;
    }

    /*
     * if x + y >= ORDER => x + y - ORDER
     * we avoid doing x + y because it may overflow 64 bits!
     * - the condition x + y < ORDER <=> x < ORDER - y
     * - x + y - ORDER <=> y - (ORDER - x)
     */
    static inline u64 modulo_add(u64 x, u64 y)
    {
        return (x < ORDER - y) ? x + y : y - (ORDER - x);
    }

    /*
     * we assume x, y < ORDER
     * if x < y, we need ORDER - y + x
     */
    static inline u64 modulo_sub(u64 x, u64 y)
    {
        return (x > y) ? x - y : x + (ORDER - y);
    }

    /*
     * this does modulo multuply only for x, y < 2^32 such that it does not overflow 64 bits!
     */
    static inline u64 modulo_mul(u64 x, u64 y)
    {
        assert(x < (u64)1 << 32);
        assert(y < (u64)1 << 32);
        x = x * y;
        return (x > ORDER) ? x - ORDER : x;
    }

    static GoldilocksField from_noncanonical_u96(u64 n_lo, u32 n_hi)
    {
        u64 t1 = (u64)n_hi * EPSILON;
        return GoldilocksField(modulo_add(n_lo, t1));
    }

    static GoldilocksField from_canonical_u64(u64 x)
    {
        assert(x < ORDER);
        return GoldilocksField(x);
    }

    u64 to_noncanonical_u64()
    {
        return this->val;
    }

    GoldilocksField operator+(const GoldilocksField &rhs)
    {
        return GoldilocksField(modulo_add(this->val, rhs.get_val()));
    }

    GoldilocksField add_canonical_u64(const u64 &rhs)
    {
        // Default implementation.
        return *this + GoldilocksField::from_canonical_u64(rhs);
    }

    /*
     * This is based on x = x_hi_hi * 2^96 + x_hi_lo * 2^64 + x_lo (x_lo is 64 bits)
     * Note that EPSILON is 0xFFFFFFFF, that's why we can do &
     */
    inline static u64 reduce128(u128 x)
    {
        u64 x_lo = (u64)(x & 0xFFFFFFFFFFFFFFFF);
        u64 x_hi = (u64)(x >> 64);

        u64 x_hi_hi = x_hi >> 32;
        u64 x_hi_lo = x_hi & EPSILON;

        u64 t0 = modulo_sub(x_lo, x_hi_hi);
        u64 t1 = x_hi_lo * EPSILON;
        u64 t2 = modulo_add(t0, t1);

        return t2;
    }

    inline static u64 mul(u64 a, u64 b)
    {
        // a * b = a1*b1*2^64 + (a1*b0 + a0*b1)*2^32 + a0*b0
        const u64 a0 = a & 0xFFFFFFFF;
        const u64 a1 = a >> 32;
        const u64 b0 = b & 0xFFFFFFFF;
        const u64 b1 = b >> 32;
        const u64 M = GoldilocksField::ORDER;

        // a0 * b0 mod M
        u64 res = modulo_mul(a0, b0);
        assert(res < M);

        // (a1*b0 + a0*b1)*2^32 mod M
        u64 tmp1 = modulo_mul(a0, b1);
        assert(tmp1 < M);
        u64 tmp2 = modulo_mul(a1, b0);
        assert(tmp2 < M);
        u64 tmp = modulo_add(tmp1, tmp2);
        res = modulo_add(res, tmp);
        assert(res < M);

        // a1 * b1 * 2^64 mod M
        tmp1 = modulo_mul(a1, b1);
        assert(tmp1 < M);

        // TODO - split tmp1

        return res;
    }

    inline static u64 mulv2(u64 a, u64 b)
    {
        // a * b = a1*b1*2^64 + (a1*b0 + a0*b1)*2^32 + a0*b0
        const u64 a0 = a & 0xFFFFFFFF;
        const u64 a1 = a >> 32;
        const u64 b0 = b & 0xFFFFFFFF;
        const u64 b1 = b >> 32;
        const u64 M = GoldilocksField::ORDER;

        // a0 * b0 mod M
        u64 res = (a0 * b0) % M;
        assert(res < M);

        // (a1*b0 + a0*b1)*2^32 mod M
        u64 tmp = (a0 * b1 + a1 * b0) % M;
        assert(tmp < M);
        assert(tmp < ((u64)1 << 32));
        tmp = (tmp << 32) % M;
        assert(tmp < M);

        res = (res + tmp) % M;
        assert(res < M);

        // a1 * b1 * 2^64 mod M
        tmp = (a1 * b1) % M;
        assert(tmp < M);

        tmp = (tmp * EPSILON) % M;
        assert(tmp < M);

        res = (res + tmp) % M;
        assert(res < M);

        return res;
    }

    inline static u64 mulmod(u64 a, u64 b)
    {
        return (u64)(((u128)a * (u128)b) % GoldilocksField::ORDER);
    }

    GoldilocksField operator*(const GoldilocksField &rhs) const
    {
        // return GoldilocksField(reduce128((u128)this->val * (u128)rhs.get_val()));
        // return GoldilocksField(GoldilocksField::mulmod(this->val, rhs.get_val()));
        // u64 v1 = GoldilocksField::mul(this->val, rhs.get_val());
        u64 v1 = GoldilocksField::reduce128((u128)this->val * (u128)rhs.get_val());
        // u64 v2 = GoldilocksField::mulmod(this->val, rhs.get_val());
        // if (v1 != v2) {
        //      printf("Diff for %lu * %lu\n", this->val, rhs.get_val());
        // }
        // return GoldilocksField(v2);
        return GoldilocksField(v1);
    }

    inline GoldilocksField multiply_accumulate(GoldilocksField x, GoldilocksField y)
    {
        return *this + x * y;
    }

    inline static GoldilocksField from_noncanonical_u128(u128 n)
    {
        return GoldilocksField(GoldilocksField::reduce128(n));
    }
};

#endif // __GOLDILOCKS_H__