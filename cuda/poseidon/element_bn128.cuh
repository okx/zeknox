#ifndef __ELEMENT_BN128_CUH__
#define __ELEMENT_BN128_CUH__

#include "int_types.h"
#include "cuda_utils.cuh"

#ifdef USE_CUDA
extern CONST u64 rSquareGPU[4];
#else
extern CONST u64 rSquare[4];
#endif

#ifdef USE_CUDA
#define FFE FFElementGPU
#else
#define FFE FFElement
#endif

class FFE
{
private:
    INLINE void set_vals(const u64 x[4])
    {
        z[0] = x[0];
        z[1] = x[1];
        z[2] = x[2];
        z[3] = x[3];
    }

    DEVICE void mul64(u64 a, u64 b, u64 *h, u64 *l);

    DEVICE void add64(u64 a, u64 b, u64 cin, u64 *r, u64 *cout);

    DEVICE void sub64(u64 a, u64 b, u64 bin, u64 *r, u64 *bout);

    DEVICE void madd0(u64 a, u64 b, u64 c, u64 *hi);

    DEVICE void madd1(u64 a, u64 b, u64 c, u64 *hi, u64 *lo);

    DEVICE void madd2(u64 a, u64 b, u64 c, u64 d, u64 *hi, u64 *lo);

    DEVICE void madd3(u64 a, u64 b, u64 c, u64 d, u64 e, u64 *hi, u64 *lo);

    DEVICE void _mulGeneric(u64 *z, u64 *x, u64 *y);

    DEVICE void _addGeneric(u64 *z, u64 *x, u64 *y);

    DEVICE void _fromMontGeneric(u64 *z);

public:
    u64 z[4] = {0};

    DEVICE FFE(){};

    DEVICE FFE(const u64 x[4])
    {
        set_vals(x);
    };

    DEVICE static FFE NewElement()
    {
        return FFE();
    }

    INLINE void Set(const FFE x)
    {
        set_vals(x.z);
    }

    // SetZero z = 0
    INLINE void SetZero()
    {
        z[0] = 0;
        z[1] = 0;
        z[2] = 0;
        z[3] = 0;
    }

    // SetOne z = 1 (in Montgomery form)
    INLINE void SetOne()
    {
        z[0] = 12436184717236109307u;
        z[1] = 3962172157175319849u;
        z[2] = 7381016538464732718u;
        z[3] = 1011752739694698287u;
    }

    DEVICE void SetUint64(u64 v)
    {
        z[0] = v;
        z[1] = 0;
        z[2] = 0;
        z[3] = 0;
        // z.ToMont()
        u64 r[4];
#ifdef USE_CUDA
        _mulGeneric(r, z, (u64 *)rSquareGPU);
#else
        _mulGeneric(r, z, (u64 *)rSquare);
#endif
        this->set_vals(r);
    }

    INLINE void ToMont()
    {
        u64 r[4];
#ifdef USE_CUDA
        _mulGeneric(r, z, (u64 *)rSquareGPU);
#else
        _mulGeneric(r, z, (u64 *)rSquare);
#endif
        this->set_vals(r);
    }

    INLINE void FromMont()
    {
        _fromMontGeneric(this->z);
    }

    // Add z = x + y mod q
    INLINE void Add(FFE x, FFE y)
    {
        _addGeneric(this->z, x.z, y.z);
    }

    // / Mul z = x * y mod q
    // see https://hackmd.io/@zkteam/modular_multiplication
    INLINE void Mul(FFE x, FFE y)
    {
        _mulGeneric(this->z, x.z, y.z);
    }

    DEVICE inline void Square(FFE x)
    {
        _mulGeneric(this->z, x.z, x.z);
    }

    // Exp z = x^exponent mod q
    // Note: since only Exp(x,5) is used, we implement a custom version of Exp
    DEVICE void Exp5(const FFE x)
    {
        this->Set(x);
        this->Square(z);
        this->Square(z);
        u64 r[4];
        _mulGeneric(r, z, (u64 *)x.z);
        this->set_vals(r);
    }

    DEVICE void Exp5()
    {
        u64 x[4];
        x[0] = z[0];
        x[1] = z[1];
        x[2] = z[2];
        x[3] = z[3];
        this->Square(z);
        this->Square(z);
        u64 r[4];
        _mulGeneric(r, z, x);
        this->set_vals(r);
    }
};

#endif // __ELEMENT_BN128_CUH__