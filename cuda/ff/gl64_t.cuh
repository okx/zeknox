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

    inline operator uint64_t() const
    {   auto ret = *this; ret.from(); return ret.val;   }

    inline void to()    { reduce(); }
    inline void from()  {}

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


    inline gl64_t& operator*=(const gl64_t& a)
    {   mul(a);           return *this;   }
    friend inline gl64_t operator*(gl64_t a, const gl64_t& b)
    {   a.mul(b); a.to(); return a;   }

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