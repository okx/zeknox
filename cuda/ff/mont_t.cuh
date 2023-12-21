#if defined(__CUDA_ARCH__) && !defined(__CRYPTO_FF_MONT_T_CUH__)
# define __CRYPTO_FF_MONT_T_CUH__

# include <cstddef>
# include <cstdint>

# define inline __device__ __forceinline__
# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif

template<const size_t N, const uint32_t MOD[(N+31)/32], const uint32_t& M0,
         const uint32_t RR[(N+31)/32], const uint32_t ONE[(N+31)/32],
         const uint32_t MODx[(N+31)/32] = MOD>
class __align__(((N+63)/64)&1 ? 8 : 16) mont_t {
public:
    static const size_t nbits = N;
    static constexpr size_t __device__ bit_length() { return N; }
    static const uint32_t degree = 1;
    using mem_t = mont_t; // mem_t is an alias for mont_t
protected:
    static const size_t n = (N+31)/32; // num of u32
private:
    uint32_t even[n];

    /**
     * @brief add a into acc, with 
     * @param[in] acc, pointer to a u32
     * @param[in] a,   pointer to a const u32
     * @param[in] n, number of u32 to add (carry is carried to next if there is carry); default to the number of u32 as stored in monot_t instance
    */
    static inline void cadd_n(uint32_t* acc, const uint32_t* a, size_t n=n)
    {
        asm("add.cc.u32 %0, %0, %1;" : "+r"(acc[0]) : "r"(a[0]));
        for (size_t i = 1; i < n; i++)
            asm("addc.cc.u32 %0, %0, %1;" : "+r"(acc[i]) : "r"(a[i]));
        // return carry flag
    }

private:
    /**
     * cast the object to a uint32_t* (a pointer)
     * `const uint32_t*() const` means a pointer whose content cannot be changed; and also the value pointed is a constant
    */
    inline operator const uint32_t*() const             { return even;    }  
    inline operator uint32_t*()                         { return even;    }

public:
    inline uint32_t& operator[](size_t i)               { return even[i]; }
    inline const uint32_t& operator[](size_t i) const   { return even[i]; }
    inline size_t len() const                           { return n;       }

    inline mont_t() {}
    inline mont_t(const uint32_t *p)
    {
        for (size_t i = 0; i < n; i++)
            even[i] = p[i];
    }

    inline void store(uint32_t *p) const
    {
        for (size_t i = 0; i < n; i++)
            p[i] = even[i];
    }

    inline mont_t& operator+=(const mont_t& b)
    {
        cadd_n(&even[0], &b[0]);
        final_subc();
        return *this;
    }


    inline void final_subc()
    {
        uint32_t carry, tmp[n];

        asm("addc.u32 %0, 0, 0;" : "=r"(carry));

        asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(MOD[0]));
        for (size_t i = 1; i < n; i++)
            asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));
        asm("subc.u32 %0, %0, 0;" : "+r"(carry));

        asm("{ .reg.pred %top;");
        asm("setp.eq.u32 %top, %0, 0;" :: "r"(carry));
        for (size_t i = 0; i < n; i++)
            asm("@%top mov.b32 %0, %1;" : "+r"(even[i]) : "r"(tmp[i]));
        asm("}");
    }
}

#endif