#include "element_bn128.cuh"

#ifdef USE_CUDA
CONST u64 rSquareGPU[4]
#else
CONST u64 rSquare[4]
#endif
    = {
        1997599621687373223u,
        6052339484930628067u,
        10108755138030829701u,
        150537098327114917u};

INLINE void FFE::mul64(u64 a, u64 b, u64 *h, u64 *l)
{
    uint128_t c128 = (uint128_t)a * b;
#ifdef USE_CUDA
    *l = c128.lo;
    *h = c128.hi;
#else
    *l = (u64)c128;
    *h = (u64)(c128 >> 64);
#endif
}

INLINE void FFE::add64(u64 a, u64 b, u64 cin, u64 *r, u64 *cout)
{
    assert(cin == 0 || cin == 1);

    if (a > 0xFFFFFFFFFFFFFFFF - b)
    {
        *r = a + b;
        *cout = 1;
    }
    else
    {
        *r = a + b;
        *cout = 0;
    }
    if (cin == 1 && *r == 0xFFFFFFFFFFFFFFFF)
    {
        *r = 0;
        *cout = 1;
    }
    else
    {
        *r += cin;
    }
}

INLINE void FFE::sub64(u64 a, u64 b, u64 bin, u64 *r, u64 *bout)
{
    if (a < b)
    {
        *r = 0xFFFFFFFFFFFFFFFF - b + a + 1;
        *bout = 1;
    }
    else
    {
        *r = a - b;
        *bout = 0;
    }
    if (*r < bin)
    {
        *r = 0xFFFFFFFFFFFFFFFF - bin + *r + 1;
        *bout = 1;
    }
    else
    {
        *r -= bin;
    }
}

// madd0 hi = a*b + c (discards lo bits)
DEVICE void FFE::madd0(u64 a, u64 b, u64 c, u64 *hi)
{
    u64 carry, lo, tmp;
    mul64(a, b, hi, &lo);
    add64(lo, c, 0, &tmp, &carry);
    add64(*hi, 0, carry, hi, &tmp);
}

// madd1 hi, lo = a*b + c
DEVICE void FFE::madd1(u64 a, u64 b, u64 c, u64 *hi, u64 *lo)
{
    u64 carry, tmp;
    mul64(a, b, hi, lo);
    add64(*lo, c, 0, lo, &carry);
    add64(*hi, 0, carry, hi, &tmp);
}

// madd2 hi, lo = a*b + c + d
DEVICE void FFE::madd2(u64 a, u64 b, u64 c, u64 d, u64 *hi, u64 *lo)
{
    u64 carry, tmp;
    mul64(a, b, hi, lo);
    add64(c, d, 0, &c, &carry);
    add64(*hi, 0, carry, hi, &tmp);
    add64(*lo, c, 0, lo, &carry);
    add64(*hi, 0, carry, hi, &tmp);
}

DEVICE void FFE::madd3(u64 a, u64 b, u64 c, u64 d, u64 e, u64 *hi, u64 *lo)
{
    u64 carry, tmp;
    mul64(a, b, hi, lo);
    add64(c, d, 0, &c, &carry);
    add64(*hi, 0, carry, hi, &tmp);
    add64(*lo, c, 0, lo, &carry);
    add64(*hi, e, carry, hi, &tmp);
}

DEVICE void FFE::_mulGeneric(u64 *z, u64 *x, u64 *y)
{
    u64 t[4];
    u64 c[3];

    // round 0
    u64 v = x[0];
    mul64(v, y[0], &c[1], &c[0]);
    u64 m = c[0] * 14042775128853446655u;
    madd0(m, 4891460686036598785u, c[0], &c[2]);
    madd1(v, y[1], c[1], &c[1], &c[0]);
    madd2(m, 2896914383306846353u, c[2], c[0], &c[2], &t[0]);
    madd1(v, y[2], c[1], &c[1], &c[0]);
    madd2(m, 13281191951274694749u, c[2], c[0], &c[2], &t[1]);
    madd1(v, y[3], c[1], &c[1], &c[0]);
    madd3(m, 3486998266802970665u, c[0], c[2], c[1], &t[3], &t[2]);

    // round 1
    v = x[1];
    madd1(v, y[0], t[0], &c[1], &c[0]);
    m = c[0] * 14042775128853446655u;
    madd0(m, 4891460686036598785u, c[0], &c[2]);
    madd2(v, y[1], c[1], t[1], &c[1], &c[0]);
    madd2(m, 2896914383306846353u, c[2], c[0], &c[2], &t[0]);
    madd2(v, y[2], c[1], t[2], &c[1], &c[0]);
    madd2(m, 13281191951274694749u, c[2], c[0], &c[2], &t[1]);
    madd2(v, y[3], c[1], t[3], &c[1], &c[0]);
    madd3(m, 3486998266802970665u, c[0], c[2], c[1], &t[3], &t[2]);

    // round 2
    v = x[2];
    madd1(v, y[0], t[0], &c[1], &c[0]);
    m = c[0] * 14042775128853446655u;
    madd0(m, 4891460686036598785u, c[0], &c[2]);
    madd2(v, y[1], c[1], t[1], &c[1], &c[0]);
    madd2(m, 2896914383306846353u, c[2], c[0], &c[2], &t[0]);
    madd2(v, y[2], c[1], t[2], &c[1], &c[0]);
    madd2(m, 13281191951274694749u, c[2], c[0], &c[2], &t[1]);
    madd2(v, y[3], c[1], t[3], &c[1], &c[0]);
    madd3(m, 3486998266802970665u, c[0], c[2], c[1], &t[3], &t[2]);

    // round 3
    v = x[3];
    madd1(v, y[0], t[0], &c[1], &c[0]);
    m = c[0] * 14042775128853446655u;
    madd0(m, 4891460686036598785u, c[0], &c[2]);
    madd2(v, y[1], c[1], t[1], &c[1], &c[0]);
    madd2(m, 2896914383306846353u, c[2], c[0], &c[2], &z[0]);
    madd2(v, y[2], c[1], t[2], &c[1], &c[0]);
    madd2(m, 13281191951274694749u, c[2], c[0], &c[2], &z[1]);
    madd2(v, y[3], c[1], t[3], &c[1], &c[0]);
    madd3(m, 3486998266802970665u, c[0], c[2], c[1], &z[3], &z[2]);

    if (!(z[3] < 3486998266802970665u ||
          (z[3] == 3486998266802970665u &&
           (z[2] < 13281191951274694749u ||
            (z[2] == 13281191951274694749u &&
             (z[1] < 2896914383306846353u ||
              (z[1] == 2896914383306846353u &&
               (z[0] < 4891460686036598785u))))))))
    {
        u64 b, tmp;
        sub64(z[0], 4891460686036598785u, 0, &z[0], &b);
        sub64(z[1], 2896914383306846353u, b, &z[1], &b);
        sub64(z[2], 13281191951274694749u, b, &z[2], &b);
        sub64(z[3], 3486998266802970665u, b, &z[3], &tmp);
    }
}

DEVICE void FFE::_addGeneric(u64 *z, u64 *x, u64 *y)
{
    u64 carry, tmp;

    add64(x[0], y[0], 0, &z[0], &carry);
    add64(x[1], y[1], carry, &z[1], &carry);
    add64(x[2], y[2], carry, &z[2], &carry);
    add64(x[3], y[3], carry, &z[3], &tmp);

    if (!(z[3] < 3486998266802970665u ||
          (z[3] == 3486998266802970665u &&
           (z[2] < 13281191951274694749u ||
            (z[2] == 13281191951274694749u &&
             (z[1] < 2896914383306846353u ||
              (z[1] == 2896914383306846353u &&
               (z[0] < 4891460686036598785u))))))))
    {
        u64 b, tmp;
        sub64(z[0], 4891460686036598785u, 0, &z[0], &b);
        sub64(z[1], 2896914383306846353u, b, &z[1], &b);
        sub64(z[2], 13281191951274694749u, b, &z[2], &b);
        sub64(z[3], 3486998266802970665u, b, &z[3], &tmp);
    }
}

DEVICE void FFE::_fromMontGeneric(u64 *z)
{
    u64 m = z[0] * 14042775128853446655u;
    u64 c;
    madd0(m, 4891460686036598785u, z[0], &c);
    madd2(m, 2896914383306846353u, z[1], c, &c, &z[0]);
    madd2(m, 13281191951274694749u, z[2], c, &c, &z[1]);
    madd2(m, 3486998266802970665u, z[3], c, &c, &z[2]);
    z[3] = c;

    m = z[0] * 14042775128853446655u;
    madd0(m, 4891460686036598785u, z[0], &c);
    madd2(m, 2896914383306846353u, z[1], c, &c, &z[0]);
    madd2(m, 13281191951274694749u, z[2], c, &c, &z[1]);
    madd2(m, 3486998266802970665u, z[3], c, &c, &z[2]);
    z[3] = c;

    m = z[0] * 14042775128853446655u;
    madd0(m, 4891460686036598785u, z[0], &c);
    madd2(m, 2896914383306846353u, z[1], c, &c, &z[0]);
    madd2(m, 13281191951274694749u, z[2], c, &c, &z[1]);
    madd2(m, 3486998266802970665u, z[3], c, &c, &z[2]);
    z[3] = c;

    m = z[0] * 14042775128853446655u;
    madd0(m, 4891460686036598785u, z[0], &c);
    madd2(m, 2896914383306846353u, z[1], c, &c, &z[0]);
    madd2(m, 13281191951274694749u, z[2], c, &c, &z[1]);
    madd2(m, 3486998266802970665u, z[3], c, &c, &z[2]);
    z[3] = c;

    if (!(z[3] < 3486998266802970665u ||
          (z[3] == 3486998266802970665u &&
           (z[2] < 13281191951274694749u ||
            (z[2] == 13281191951274694749u &&
             (z[1] < 2896914383306846353u ||
              (z[1] == 2896914383306846353u &&
               (z[0] < 4891460686036598785u))))))))
    {
        u64 b, tmp;
        sub64(z[0], 4891460686036598785u, 0, &z[0], &b);
        sub64(z[1], 2896914383306846353u, b, &z[1], &b);
        sub64(z[2], 13281191951274694749u, b, &z[2], &b);
        sub64(z[3], 3486998266802970665u, b, &z[3], &tmp);
    }
}
