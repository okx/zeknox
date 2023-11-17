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
        printf("invoking asm \n");
        uint64_t tmp;
        uint32_t carry;

        asm("add.cc.u64 %0, %0, %2; addc.u32 %1, 0, 0;"
            : "+l"(val), "=r"(carry)
            : "l"(b.val));

        asm("{ .reg.pred %top;");
        asm("sub.cc.u64 %0, %2, %3; subc.u32 %1, %1, 0;"
            : "=l"(tmp), "+r"(carry)
            : "l"(val), "l"(MOD));
        asm("setp.eq.u32 %top, %0, 0;" :: "r"(carry));
        asm("@%top mov.b64 %0, %1;" : "+l"(val) : "l"(tmp));
        asm("}");

        return *this;
    }
    friend inline gl64_t operator+(gl64_t a, const gl64_t& b)
    {   
        return a += b;   
    }

private:
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