#ifndef __CRYPTO_FF_GL64_T_CUH__
#define __CRYPTO_FF_GL64_T_CUH__
#ifndef __DEBUG__PRINT__
#define __DEBUG__PRINT__
#include <cstdio>
#endif
# include <cstdint>
namespace gl64_device {
    static __device__ __constant__ /*const*/ uint32_t W = 0xffffffffU;
}

#ifdef __CUDA_ARCH__
# define inline __device__ __forceinline__
# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif

class gl64_t {
private:
    uint64_t val;
public:
    using mem_t = gl64_t;
    static const uint32_t degree = 1;
    static const unsigned nbits = 64;
    static const uint64_t MOD = 0xffffffff00000001U;
    static constexpr size_t __device__ bit_length()     { return 64; }

    inline uint64_t& operator[](size_t i)               { return val; }
    inline const uint64_t& operator[](size_t i) const   { return val; }
    inline size_t len() const                           { return 1;   }

    inline gl64_t()                                     {}
    inline gl64_t(const uint64_t a)                     { val = a;  to(); }
    inline gl64_t(const uint64_t *p)                    { val = *p; to(); }

    static inline const gl64_t one()
    {   gl64_t ret; ret.val = 1; return ret;   }

    inline operator uint64_t() const
    {   auto ret = *this; ret.from(); return ret.val;   }

    inline void to()    { reduce(); }
    inline void from()  {}

    // conditionally select. return a if sel_a != 0, else b
    static inline gl64_t csel(const gl64_t& a, const gl64_t& b, int sel_a)
    {
        gl64_t ret;
        asm("{ .reg.pred %sel_a;");
        asm("setp.ne.s32 %sel_a, %0, 0;" : : "r"(sel_a));
        asm("selp.u64 %0, %1, %2, %sel_a;" : "=l"(ret.val) : "l"(a.val), "l"(b.val));
        asm("}");
        return ret;
    }


    // addition
    inline gl64_t& operator+=(const gl64_t& b)
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
        asm("setp.eq.u32 %top, %0, 0;" :: "r"(carry));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));  // if carry is zero, move tmp to val
        asm("}");

        return *this;
    }
    friend inline gl64_t operator+(gl64_t a, const gl64_t& b)
    {   
        return a += b;   
    }

    // subtraction
    inline gl64_t& operator-=(const gl64_t& b)
    {
        uint64_t tmp;
        uint32_t borrow;
        asm("{ .reg.pred %top;");
        asm("sub.cc.u64 %0, %0, %2; subc.u32 %1, 0, 0;"
            : "+l"(val), "=r"(borrow)
            : "l"(b.val));
        
        asm("add.u64 %0, %1, %2;" : "=l"(tmp) : "l"(val), "l"(MOD));
        asm("setp.ne.u32 %top, %0, 0;" :: "r"(borrow));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));   // use tmp value if borrow != 0
        asm("}");

        return *this;
    }
    friend inline gl64_t operator-(gl64_t a, const gl64_t& b)
    {   return a -= b;   }

    // multiplication
    inline gl64_t& operator*=(const gl64_t& a)
    {   mul(a);           return *this;   }
    friend inline gl64_t operator*(gl64_t a, const gl64_t& b)
    {   a.mul(b); a.to(); return a;   }

    // right shift
    inline gl64_t& operator>>=(unsigned r)
    {
        uint64_t tmp;
        uint32_t carry;

        while (r--) {
            tmp = val&1 ? MOD : 0;
            asm("add.cc.u64 %0, %0, %2; addc.u32 %1, 0, 0;"
                : "+l"(tmp) "=r"(carry)
                : "l"(val));   // if val is even, it could be divided by 2 (rshift 1); if it is odd, add by MOD first, make it even, 
                               // and then divide by 2 (rshift 1), if there is carry out in addition, it means 1 << 64 is not considerred, add 1 << 63
                               // to the result ( 1<<63 is half of 1<<64);
            val = (tmp >> 1) + ((uint64_t)carry << 63);
        }

        return *this;
    }
    friend inline gl64_t operator>>(gl64_t a, unsigned r)
    {   return a >>= r;   }

    // raise to a variable power, variable in respect to threadIdx,
    // but mind the ^ operator's precedence!
    inline gl64_t& operator^=(uint32_t p)
    {
        gl64_t sqr = *this;
        *this = csel(*this, one(), p&1);  // if p is odd, return *this, else return one

        #pragma unroll 1
        while (p >>= 1) {
            sqr.mul(sqr);
            if (p&1)
                mul(sqr);
        }
        to();

        return *this;
    }
    friend inline gl64_t operator^(gl64_t a, uint32_t p)
    {   return a ^= p;   }
    inline gl64_t operator()(uint32_t p)
    {   return *this^p;   }

public:
    inline gl64_t reciprocal() const
    {
        gl64_t t0, t1;

        t1 = sqr_n_mul<1>(*this, 1, *this); // 0b11
        t0 = sqr_n_mul<2>(t1, 2,  t1);      // 0b1111
        t0 = sqr_n_mul<2>(t0, 2,  t1);      // 0b111111
        t1 = sqr_n_mul<2>(t0, 6,  t0);      // 0b111111111111
        t1 = sqr_n_mul<2>(t1, 12, t1);      // 0b111111111111111111111111
        t1 = sqr_n_mul<2>(t1, 6,  t0);      // 0b111111111111111111111111111111
        t1 = sqr_n_mul<1>(t1, 1,  *this);   // 0b1111111111111111111111111111111
        t1 = sqr_n_mul<2>(t1, 32, t1);      // 0b111111111111111111111111111111101111111111111111111111111111111
        t1 = sqr_n_mul<1>(t1, 1,  *this);   // 0b1111111111111111111111111111111011111111111111111111111111111111; which is P-2; a^{p-2} is the inverse
        t1.to();

        return t1;
    }

    friend inline gl64_t operator/(int one, const gl64_t& a)
    {   if (one != 1) asm("trap;"); return a.reciprocal();   }
    friend inline gl64_t operator/(const gl64_t& a, const gl64_t& b)
    {   return a * b.reciprocal();   }
    inline gl64_t& operator/=(const gl64_t& a)
    {   return *this *= a.reciprocal();   }

private:
    inline uint32_t lo() const  { return (uint32_t)(val); }
    inline uint32_t hi() const  { return (uint32_t)(val>>32); }

    // multiply another gl64
    inline void mul(const gl64_t& b)
    {
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
    }
private:
    template<int unroll> // 1, 2 or 3
    static __device__ __noinline__ gl64_t sqr_n_mul(gl64_t s, uint32_t n, gl64_t m)
    {
        if (unroll&1) {
            s.mul(s);
            n--;
        }
        if (unroll > 1) {
            #pragma unroll 1
            do {
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

        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(temp[0]), "+r"(temp[1])
            : "r"(temp[2]), "r"(gl64_device::W));

        asm("mov.b64 %0, {%1, %2};" : "=l"(val) : "r"(temp[0]), "r"(temp[1]));
    }


    // reduce self gl64
    inline void reduce()
    {
        printf("invoking reduce \n");
        uint64_t tmp;
        uint32_t carry;

        asm("add.cc.u64 %0, %2, %3; addc.u32 %1, 0, 0;"
            : "=l"(tmp), "=r"(carry)
            : "l"(val), "l"(0-MOD));
        asm("{ .reg.pred %top;");
        asm("setp.ne.u32 %top, %0, 0;" :: "r"(carry));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
        asm("}");
    }
};


# undef inline
# undef asm
#endif

#endif