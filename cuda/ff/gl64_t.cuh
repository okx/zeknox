// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CRYPTO_FF_GL64_T_CUH__
#define __CRYPTO_FF_GL64_T_CUH__
#ifdef __NVCC__
#include <functional>
#include <cstdint>
namespace gl64_device
{
    static __device__ __constant__ /*const*/ uint32_t W = 0xffffffffU;
}

#ifdef __CUDA_ARCH__
#define inline __device__ __forceinline__
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif

static constexpr uint64_t inv_logs[32] = {
    0x7fffffff80000001,
    0xbfffffff40000001,
    0xdfffffff20000001,
    0xefffffff10000001,
    0xf7ffffff08000001,
    0xfbffffff04000001,
    0xfdffffff02000001,
    0xfeffffff01000001,
    0xff7fffff00800001,
    0xffbfffff00400001,
    0xffdfffff00200001,
    0xffefffff00100001,
    0xfff7ffff00080001,
    0xfffbffff00040001,
    0xfffdffff00020001,
    0xfffeffff00010001,
    0xffff7fff00008001,
    0xffffbfff00004001,
    0xffffdfff00002001,
    0xffffefff00001001,
    0xfffff7ff00000801,
    0xfffffbff00000401,
    0xfffffdff00000201,
    0xfffffeff00000101,
    0xffffff7f00000081,
    0xffffffbf00000041,
    0xffffffdf00000021,
    0xffffffef00000011,
    0xfffffff700000009,
    0xfffffffb00000005,
    0xfffffffd00000003,
    0xfffffffe00000002};

static constexpr uint64_t omegas[32] = {
    0xffffffff00000000,
    0x0001000000000000,
    0xfffffffeff000001,
    0xefffffff00000001,
    0x00003fffffffc000,
    0x0000008000000000,
    0xf80007ff08000001,
    0xbf79143ce60ca966,
    0x1905d02a5c411f4e,
    0x9d8f2ad78bfed972,
    0x0653b4801da1c8cf,
    0xf2c35199959dfcb6,
    0x1544ef2335d17997,
    0xe0ee099310bba1e2,
    0xf6b2cffe2306baac,
    0x54df9630bf79450e,
    0xabd0a6e8aa3d8a0e,
    0x81281a7b05f9beac,
    0xfbd41c6b8caa3302,
    0x30ba2ecd5e93e76d,
    0xf502aef532322654,
    0x4b2a18ade67246b5,
    0xea9d5a1336fbc98b,
    0x86cdcc31c307e171,
    0x4bbaf5976ecfefd8,
    0xed41d05b78d6e286,
    0x10d78dd8915a171d,
    0x59049500004a4485,
    0xdfa8c93ba46d2666,
    0x7e9bd009b86a0845,
    0x400a7f755588e659,
    0x185629dcda58878c}

static constexpr uint64_t omegas_inv[32] = {0xffffffff00000000, 0xfffeffff00000001, 0x000000ffffffff00, 0x0000001000000000, 0xfffffffefffc0001, 0xfdffffff00000001, 0xffefffff00000011, 0x1d62e30fa4a4eeb0, 0x3de19c67cf496a74, 0x3b9ae9d1d8d87589, 0x76a40e0866a8e50d, 0x9af01e431fbd6ea0, 0x3712791d9eb0314a, 0x409730a1895adfb6, 0x158ee068c8241329, 0x6d341b1c9a04ed19, 0xcc9e5a57b8343b3f, 0x22e1fbf03f8b95d6, 0x46a23c48234c7df9, 0xef8856969fe6ed7b, 0xa52008ac564a2368, 0xd46e5a4c36458c11, 0x4bb9aee372cf655e, 0x10eb845263814db7, 0xc01f93fc71bb0b9b, 0xea52f593bb20759a, 0x91f3853f38e675d9, 0x3ea7eab8d8857184, 0xe4d14a114454645d, 0xe2434909eec4f00b, 0x95c0ec9a7ab50701, 0x76b6b635b6fc8719}

class gl64_t
{
public:
    uint64_t val;

public:
    using mem_t = gl64_t;
    static const uint32_t degree = 1;
    static const unsigned nbits = 64;
    static const uint64_t MOD = 0xffffffff00000001U;
    static constexpr size_t __device__ bit_length() { return 64; }

    inline uint64_t &operator[](size_t i) { return val; }
    inline const uint64_t &operator[](size_t i) const { return val; }
    inline size_t len() const { return 1; }

    inline gl64_t() {}
    inline gl64_t(const uint64_t a)
    {
        val = a;
        to();
    }
    inline gl64_t(const uint64_t *p)
    {
        val = *p;
        to();
    }

    static inline const gl64_t one()
    {
        gl64_t ret;
        ret.val = 1;
        return ret;
    }



    static inline gl64_t omega(uint32_t logn) {
         if (logn == 0)
        {
            return one();
        }
         return omegas[logn-1];
    }
    static inline gl64_t omega_inv(uint32_t logn) {
          if (logn == 0)
        {
            return one();
        }
         return omegas_inv[logn-1];
    }

        static inline const gl64_t inv_log_size(uint32_t logn)
    {
        if (logn == 0)
        {
            return one();
        }
        return inv_logs[logn-1];
    }

    inline operator uint64_t() const
    {
        auto ret = *this;
        ret.from();
        return ret.val;
    }

    inline void to() { reduce(); }
    inline void from() {}

    // conditionally select. return a if sel_a != 0, else b
    static inline gl64_t csel(const gl64_t &a, const gl64_t &b, int sel_a)
    {
        gl64_t ret;
        asm("{ .reg.pred %sel_a;");
        asm("setp.ne.s32 %sel_a, %0, 0;" : : "r"(sel_a));
        asm("selp.u64 %0, %1, %2, %sel_a;" : "=l"(ret.val) : "l"(a.val), "l"(b.val));
        asm("}");
        return ret;
    }

    // addition
    inline gl64_t &operator+=(const gl64_t &b)
    {
        from();
        uint64_t tmp;
        uint32_t carry;
        // printf("val: %lu, b.val: %lu \n", val, b.val);
        asm("add.cc.u64 %0, %0, %2; addc.u32 %1, 0, 0;"
            : "+l"(val), "=r"(carry)
            : "l"(b.val));
        // printf("val: %lu, b.val: %lu, carry: %u \n", val, b.val, carry);
        asm("{ .reg.pred %top;");
        asm("sub.cc.u64 %0, %2, %3; subc.u32 %1, %1, 0;"
            : "=l"(tmp), "+r"(carry)
            : "l"(val), "l"(MOD));
        // printf("tmp: %lu, val: %lu, carry: %u \n", tmp, val, carry);
        asm("setp.eq.u32 %top, %0, 0;" ::"r"(carry));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp)); // if carry is zero, move tmp to val
        asm("}");

        return *this;
    }
    friend inline gl64_t operator+(gl64_t a, const gl64_t &b)
    {
        return a += b;
    }

    // subtraction
    inline gl64_t &operator-=(const gl64_t &b)
    {
        uint64_t tmp;
        uint32_t borrow;
        asm("{ .reg.pred %top;");
        asm("sub.cc.u64 %0, %0, %2; subc.u32 %1, 0, 0;"
            : "+l"(val), "=r"(borrow)
            : "l"(b.val));

        asm("add.u64 %0, %1, %2;" : "=l"(tmp) : "l"(val), "l"(MOD));
        asm("setp.ne.u32 %top, %0, 0;" ::"r"(borrow));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp)); // use tmp value if borrow != 0
        asm("}");

        return *this;
    }
    friend inline gl64_t operator-(gl64_t a, const gl64_t &b)
    {
        return a -= b;
    }

    // multiplication
    inline gl64_t &operator*=(const gl64_t &a)
    {
        mul(a);
        return *this;
    }
    friend inline gl64_t operator*(gl64_t a, const gl64_t &b)
    {
        a.mul(b);
        a.to();
        return a;
    }
    friend inline gl64_t sqr(gl64_t a)
    {
        return a.sqr();
    }
    inline gl64_t &sqr()
    {
        mul(*this);
        to();
        return *this;
    }

    // right shift
    inline gl64_t &operator>>=(unsigned r)
    {
        uint64_t tmp;
        uint32_t carry;

        while (r--)
        {
            tmp = val & 1 ? MOD : 0;
            asm("add.cc.u64 %0, %0, %2; addc.u32 %1, 0, 0;"
                : "+l"(tmp) "=r"(carry)
                : "l"(val)); // if val is even, it could be divided by 2 (rshift 1); if it is odd, add by MOD first, make it even,
                             // and then divide by 2 (rshift 1), if there is carry out in addition, it means 1 << 64 is not considerred, add 1 << 63
                             // to the result ( 1<<63 is half of 1<<64);
            val = (tmp >> 1) + ((uint64_t)carry << 63);
        }

        return *this;
    }
    friend inline gl64_t operator>>(gl64_t a, unsigned r)
    {
        return a >>= r;
    }

    // raise to a variable power, variable in respect to threadIdx,
    // but mind the ^ operator's precedence!
    inline gl64_t &operator^=(uint32_t p)
    {
        //  printf("start exp, val: %llu, b: %llu \n", val, p);
        gl64_t sqr = *this;
        *this = csel(*this, one(), p & 1); // if p is odd, return *this, else return one
                                           // printf("*this: %llu \n", val);
#pragma unroll 1
        while (p >>= 1)
        {
            sqr.mul(sqr);
            if (p & 1)
                mul(sqr);
        }
        to();

        return *this;
    }

    friend inline gl64_t operator^(gl64_t a, uint32_t p)
    {
        return a ^= p;
    }
    inline gl64_t operator()(uint32_t p)
    {
        return *this ^ p;
    }

    // raise to a constant power, e.g. x^7, to be unrolled at compile time
    inline gl64_t &operator^=(int p)
    {
        if (p < 2)
            asm("trap;");

        gl64_t sqr = *this;
        if ((p & 1) == 0)
        {
            do
            {
                sqr.mul(sqr);
                p >>= 1;
            } while ((p & 1) == 0);
            *this = sqr;
        }
        for (p >>= 1; p; p >>= 1)
        {
            sqr.mul(sqr);
            if (p & 1)
                mul(sqr);
        }
        to();

        return *this;
    }
    friend inline gl64_t operator^(gl64_t a, int p)
    {
        return a ^= p;
    }
    inline gl64_t operator()(int p)
    {
        return *this ^ p;
    }

public:
    inline gl64_t reciprocal() const
    {
        gl64_t t0, t1;

        t1 = sqr_n_mul<1>(*this, 1, *this); // 0b11
        t0 = sqr_n_mul<2>(t1, 2, t1);       // 0b1111
        t0 = sqr_n_mul<2>(t0, 2, t1);       // 0b111111
        t1 = sqr_n_mul<2>(t0, 6, t0);       // 0b111111111111
        t1 = sqr_n_mul<2>(t1, 12, t1);      // 0b111111111111111111111111
        t1 = sqr_n_mul<2>(t1, 6, t0);       // 0b111111111111111111111111111111
        t1 = sqr_n_mul<1>(t1, 1, *this);    // 0b1111111111111111111111111111111
        t1 = sqr_n_mul<2>(t1, 32, t1);      // 0b111111111111111111111111111111101111111111111111111111111111111
        t1 = sqr_n_mul<1>(t1, 1, *this);    // 0b1111111111111111111111111111111011111111111111111111111111111111; which is P-2; a^{p-2} is the inverse
        t1.to();

        return t1;
    }

    friend inline gl64_t operator/(int one, const gl64_t &a)
    {
        if (one != 1)
            asm("trap;");
        return a.reciprocal();
    }
    friend inline gl64_t operator/(const gl64_t &a, const gl64_t &b)
    {
        return a * b.reciprocal();
    }
    inline gl64_t &operator/=(const gl64_t &a)
    {
        return *this *= a.reciprocal();
    }

public:
    inline uint32_t lo() const { return (uint32_t)(val); }
    inline uint32_t hi() const { return (uint32_t)(val >> 32); }

private:
    // multiply another gl64
    inline void mul(const gl64_t &b)
    {
        // printf("start mul, val: %llu, b: %llu \n", val, b);
        uint32_t a0 = lo(), b0 = b.lo();
        uint32_t a1 = hi(), b1 = b.hi();
        uint32_t temp[4];

        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(temp[0]), "=r"(temp[1])
            : "r"(a0), "r"(b0));
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(temp[2]), "=r"(temp[3])
            : "r"(a1), "r"(b1));
        uint32_t carry;
        asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, 0, 0;"
            : "+r"(temp[1]), "+r"(temp[2]), "=r"(carry)
            : "r"(a0), "r"(b1));
        asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, %2, %5;"
            : "+r"(temp[1]), "+r"(temp[2]), "+r"(temp[3])
            : "r"(a1), "r"(b0), "r"(carry));
        reduce(temp);
        // printf("mul result: %llu \n", val);
    }

private:
    template <int unroll> // 1, 2 or 3
    static __device__ __noinline__ gl64_t sqr_n_mul(gl64_t s, uint32_t n, gl64_t m)
    {
        if (unroll & 1)
        {
            s.mul(s);
            n--;
        }
        if (unroll > 1)
        {
#pragma unroll 1
            do
            {
                s.mul(s);
                s.mul(s);
            } while (n -= 2);
        }
        s.mul(m);

        return s;
    }

    // reduce u128 to u64, put the result in val
    inline void reduce(uint32_t temp[4])
    {
        uint32_t carry;

        asm("sub.cc.u32 %0, %0, %3; subc.cc.u32 %1, %1, %4; subc.u32 %2, 0, 0;"
            : "+r"(temp[0]), "+r"(temp[1]), "=r"(carry)
            : "r"(temp[2]), "r"(temp[3]));
        asm("add.cc.u32 %0, %0, %2; addc.u32 %1, %1, %3;"
            : "+r"(temp[1]), "+r"(carry)
            : "r"(temp[2]), "r"(temp[3]));

        asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, 0, 0;"
            : "+r"(temp[0]), "+r"(temp[1]), "=r"(temp[2])
            : "r"(carry), "r"(gl64_device::W));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(temp[0]), "+r"(temp[1])
            : "r"(temp[2]), "r"(gl64_device::W));

        asm("mov.b64 %0, {%1, %2};" : "=l"(val) : "r"(temp[0]), "r"(temp[1]));
    }

    // reduce self gl64
    inline void reduce()
    {
        uint64_t tmp;
        uint32_t carry;

        asm("add.cc.u64 %0, %2, %3; addc.u32 %1, 0, 0;"
            : "=l"(tmp), "=r"(carry)
            : "l"(val), "l"(0 - MOD));
        asm("{ .reg.pred %top;");
        asm("setp.ne.u32 %top, %0, 0;" ::"r"(carry));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
        asm("}");
    }
};

#undef inline
#undef asm
#endif

#endif

#endif