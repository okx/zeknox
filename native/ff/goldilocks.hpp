// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CRYPTO__FF__GL64__HPP__
#define __CRYPTO__FF__GL64__HPP__

#include "types/int_types.h"

#include "ff/gl64_t.cuh" // device-side field types

#ifndef __CUDA_ARCH__ // host-side stand-in to make CUDA code compile
#include <cstdint>    //it also produces correct results for the host-side code

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#include "ff/gl64_params.hpp"

class gl64_t
{
private:
  uint64_t val;

public:
  using mem_t = gl64_t;
  static const uint32_t degree = 1;
  static const uint64_t MOD = 0xffffffff00000001U;
  static const uint64_t EPSILON = 4294967295;              // ((u64)1 << 32) - 1 = 2^64 % MOD
  static const uint64_t EPSILON2 = 18446744069414584320ul; // 2^96 % MOD

  inline gl64_t() {}
  inline gl64_t(uint64_t a) : val(a) {}

  // dummy constructor for CUDA compability
  inline gl64_t(uint32_t val_u128[4]) {}

  static inline const gl64_t one() { return 1; }
  static inline const gl64_t zero() { return (uint64_t)0; }
  inline uint64_t get_val() const { return val; }
  inline uint32_t lo() const { return (uint32_t)(val); }
  inline uint32_t hi() const { return (uint32_t)(val >> 32); }
  inline operator uint64_t() const { return val; }

  static inline gl64_t omega(uint32_t logn)
  {
    return omegas[logn];
  }
  static inline gl64_t omega_inv(uint32_t logn)
  {
    return omegas_inv[logn];
  }

  static inline const gl64_t inv_log_size(uint32_t logn)
  {
    return domain_size_inv[logn];
  }

  gl64_t add_canonical_u64(const u64 &rhs)
  {
    this->val = modulo_add(this->val, rhs);
    return *this;
  }

  gl64_t operator+(const gl64_t &rhs)
  {
    return gl64_t(modulo_add(this->val, rhs.get_val()));
  }

  inline gl64_t &operator+=(gl64_t b)
  {
    this->val = modulo_add(this->val, b.val);
    return *this;
  }

  gl64_t operator-(const gl64_t &rhs)
  {
    return gl64_t(modulo_sub(this->val, rhs.get_val()));
  }

  inline gl64_t &operator-=(gl64_t b)
  {
    this->val = modulo_sub(this->val, b.val);
    return *this;
  }

  gl64_t operator*(const gl64_t &rhs) const
  {

    u64 v1 = reduce128((u128)this->val * (u128)rhs.get_val());

    return gl64_t(v1);
  }

  inline gl64_t &operator*=(gl64_t b) { return *this; }

  inline gl64_t &sqr() { return *this; }

  static gl64_t from_canonical_u64(u64 x)
  {
    assert(x < MOD);
    return gl64_t(x);
  }

  static inline gl64_t from_noncanonical_u96(u128 x)
  {
    u64 x_hi = (u64)(x >> 64);
    u64 x_lo = (u64)x;
    return from_noncanonical_u96(x_lo, x_hi);
  }

  static inline gl64_t from_noncanonical_u96(u64 x_lo, u64 x_hi)
  {
    // v1
    // u64 t1 = ((u64)n_hi) * (u64)EPSILON;
    // return gl64_t(modulo_add(n_lo, t1));

    // v2
    x_lo -= x_hi;
    x_lo += (x_hi << 32);
    if (x_lo >= MOD)
    {
        x_lo -= MOD;
    }
    return x_lo;
  }

  static inline gl64_t from_noncanonical_u128(u128 n)
  {
    return gl64_t(reduce128(n));
  }

  u64 to_noncanonical_u64()
  {
    return this->val;
  }

  inline gl64_t multiply_accumulate(gl64_t x, gl64_t y)
  {
    return *this + x * y;
  }

  // dummies
  inline gl64_t &operator^=(int p) { return *this; }
  inline gl64_t reciprocal() const { return 1 / (*this); }

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
    // v1
    /*
    u64 x_lo = (u64)(x & 0xFFFFFFFFFFFFFFFF);
    u64 x_hi = (u64)(x >> 64);

    u64 x_hi_hi = x_hi >> 32;
    u64 x_hi_lo = x_hi & EPSILON;

    u64 t0 = modulo_sub(x_lo, x_hi_hi);
    u64 t1 = x_hi_lo * EPSILON;
    u64 t2 = modulo_add(t0, t1);

    return t2;
    */

   // v2
    volatile u64 x_lo = (u64)x;
    volatile u64 x_hi = (u64)(x >> 64);
    volatile u64 x_hi_hi = x_hi >> 32;
    volatile u64 x_hi_lo = x_hi & 0xFFFFFFFF;
    volatile u64 t1 = x_hi_lo * 0xFFFFFFFF;
    __asm__(
        "sub %2, %1\n\t"
        "sbb %%ebx, %%ebx\n\t"
        "sub %%rbx, %1\n\t"
        "add %1, %0\n\t"
        "sbb %%ebx, %%ebx\n\t"
        "add %%rbx, %0"
        : "+r"(t1)
        : "r"(x_lo), "r"(x_hi_hi)
        : "rbx");
    return t1;
  }
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif // __CUDA_ARCH__

typedef gl64_t fr_t;

/*
template <>
struct std::hash<gl64_t>
{
  std::size_t operator()(const gl64_t &key) const
  {
    std::size_t hash = 0;
    // boost hashing, see
    // https://stackoverflow.com/questions/35985960/c-why-is-boosthash-combine-the-best-way-to-combine-hash-values/35991300#35991300
    // for (int i = 0; i < CONFIG::limbs_count; i++)
    hash ^= std::hash<uint32_t>()(key.lo()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<uint32_t>()(key.hi()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};
*/

// #endif
#endif