// Copyright Supranational LLC
// Copyright 2024 OKX
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CRYPTO__FF__GL64__HPP__
#define __CRYPTO__FF__GL64__HPP__

#include "types/int_types.h"
#include "ff/gl64_t.cuh"
#include "ff/gl64_params.hpp"

class cpp_gl64_t
{
private:
  uint64_t val;

public:
  using mem_t = cpp_gl64_t;
  static const uint32_t degree = 1;
  static const uint64_t MOD = 0xffffffff00000001U; // Goldilocks prime
  static const uint64_t EPSILON = 4294967295;      // ((u64)1 << 32) - 1 = 2^64 % MOD

  inline cpp_gl64_t() : val(0) {}
  inline cpp_gl64_t(uint64_t a) : val(a) {}

  // dummy constructor for CUDA compability
  inline cpp_gl64_t(uint32_t val_u128[4]) {}

  static inline const cpp_gl64_t one() { return cpp_gl64_t(1); }
  static inline const cpp_gl64_t zero() { return cpp_gl64_t((uint64_t)0); }
  inline uint64_t get_val() const { return this->val; }
  inline uint32_t lo() const { return (uint32_t)(this->val); }
  inline uint32_t hi() const { return (uint32_t)(this->val >> 32); }
  inline operator uint64_t() const { return this->val; }
  inline void make_zero() { this->val = 0; }

  static inline cpp_gl64_t omega(uint32_t logn)
  {
    return OMEGA[logn];
  }
  static inline cpp_gl64_t omega_inv(uint32_t logn)
  {
    return OMEGA_INV[logn];
  }

  static inline const cpp_gl64_t inv_log_size(uint32_t logn)
  {
    return DOMAIN_SIZE_INV[logn];
  }

  cpp_gl64_t add_canonical_u64(const u64 &rhs)
  {
    this->val = modulo_add(this->val, rhs);
    return *this;
  }

  cpp_gl64_t operator+(const cpp_gl64_t &rhs)
  {
    return cpp_gl64_t(modulo_add(this->val, rhs.get_val()));
  }

  inline cpp_gl64_t &operator+=(cpp_gl64_t &rhs)
  {
    this->val = modulo_add(this->val, rhs.get_val());
    return *this;
  }

  cpp_gl64_t operator-(const cpp_gl64_t &rhs)
  {
    return cpp_gl64_t(modulo_sub(this->val, rhs.get_val()));
  }

  inline cpp_gl64_t &operator-=(cpp_gl64_t &rhs)
  {
    this->val = modulo_sub(this->val, rhs.get_val());
    return *this;
  }

  cpp_gl64_t operator*(const cpp_gl64_t &rhs) const
  {
    u64 v1 = reduce128((u128)this->val * (u128)rhs.get_val());
    return cpp_gl64_t(v1);
  }

  inline cpp_gl64_t &operator*=(cpp_gl64_t b)
  {
    this->val = reduce128((u128)this->val * (u128)b.val);
    return *this;
  }

  inline cpp_gl64_t &sqr() { return *this; }

  static cpp_gl64_t from_canonical_u64(u64 x)
  {
    assert(x < MOD);
    return cpp_gl64_t(x);
  }

  static inline cpp_gl64_t from_noncanonical_u96(u128 x)
  {
    u64 x_hi = (u64)(x >> 64);
    u64 x_lo = (u64)x;
    return from_noncanonical_u96(x_lo, x_hi);
  }

  static inline cpp_gl64_t from_noncanonical_u96(u64 x_lo, u64 x_hi)
  {
    u64 t1 = ((u64)x_hi) * (u64)EPSILON;
    return cpp_gl64_t(modulo_add(x_lo, t1));
  }

  static inline cpp_gl64_t from_noncanonical_u128(u128 n)
  {
    return cpp_gl64_t(reduce128(n));
  }

  u64 to_noncanonical_u64()
  {
    return this->val;
  }

  inline cpp_gl64_t multiply_accumulate(cpp_gl64_t x, cpp_gl64_t y)
  {
    return *this + x * y;
  }

  // dummies
  inline cpp_gl64_t &operator^=(int p) { return *this; }
  inline cpp_gl64_t reciprocal() const { return 1 / (*this); }

private:
  /*
   * if x + y >= ORDER => x + y - ORDER
   * we avoid doing x + y because it may overflow 64 bits!
   * - the condition x + y < ORDER <=> x < ORDER - y
   * - x + y - ORDER <=> y - (ORDER - x)
   */
  static inline u64 modulo_add(u64 x, u64 y)
  {
    return (x < MOD - y) ? x + y : y - (MOD - x);
  }

  /*
   * we assume x, y < ORDER
   * if x < y, we need ORDER - y + x
   */
  static inline u64 modulo_sub(u64 x, u64 y)
  {
    return (x > y) ? x - y : x + (MOD - y);
  }

  /*
   * this does modulo multuply only for x, y < 2^32 such that it does not overflow 64 bits!
   */
  static inline u64 modulo_mul(u64 x, u64 y)
  {
    assert(x < (u64)1 << 32);
    assert(y < (u64)1 << 32);
    x = x * y;
    return (x > MOD) ? x - MOD : x;
  }

  static inline u64 reduce128(u128 x)
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
};

// typedefs to make the code compile
#ifndef USE_CUDA
typedef cpp_gl64_t fr_t;
#else // USE_CUDA
#ifndef __CUDA_ARCH__
typedef cpp_gl64_t gl64_t;
typedef cpp_gl64_t fr_t;
#else
typedef gl64_t fr_t;
#endif // __CUDA_ARCH__
#endif // USE_CUDA

#endif // __CRYPTO__FF__GL64__HPP__