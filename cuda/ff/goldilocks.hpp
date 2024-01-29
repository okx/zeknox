// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CRYPTO__FF__GL64__HPP__
#define __CRYPTO__FF__GL64__HPP__
// #include <iomanip>
// #include <iostream>
// #include <random>
// #include <sstream>
// #include <string>
// #ifdef __NVCC__
#include "gl64_t.cuh" // device-side field types

#ifndef __CUDA_ARCH__ // host-side stand-in to make CUDA code compile,
#include <cstdint>    // not to produce correct result...

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
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
    0x185629dcda58878c};

static constexpr uint64_t omegas_inv[32] = {0xffffffff00000000, 0xfffeffff00000001, 0x000000ffffffff00, 0x0000001000000000, 0xfffffffefffc0001, 0xfdffffff00000001, 0xffefffff00000011, 0x1d62e30fa4a4eeb0, 0x3de19c67cf496a74, 0x3b9ae9d1d8d87589, 0x76a40e0866a8e50d, 0x9af01e431fbd6ea0, 0x3712791d9eb0314a, 0x409730a1895adfb6, 0x158ee068c8241329, 0x6d341b1c9a04ed19, 0xcc9e5a57b8343b3f, 0x22e1fbf03f8b95d6, 0x46a23c48234c7df9, 0xef8856969fe6ed7b, 0xa52008ac564a2368, 0xd46e5a4c36458c11, 0x4bb9aee372cf655e, 0x10eb845263814db7, 0xc01f93fc71bb0b9b, 0xea52f593bb20759a, 0x91f3853f38e675d9, 0x3ea7eab8d8857184, 0xe4d14a114454645d, 0xe2434909eec4f00b, 0x95c0ec9a7ab50701, 0x76b6b635b6fc8719};

class gl64_t
{
public:
  uint64_t val;

public:
  using mem_t = gl64_t;
  static const uint32_t degree = 1;
  static const uint64_t MOD = 0xffffffff00000001U;
  static inline gl64_t omega(uint32_t logn)
  {

    if (logn == 0)
    {
      return one();
    }

    return omegas[logn - 1];
  }
  static inline gl64_t omega_inv(uint32_t logn)
  {
    if (logn == 0)
    {
      return one();
    }
    return omegas_inv[logn - 1];
  }

  static inline const gl64_t inv_log_size(uint32_t logn)
  {
    if (logn == 0)
    {
      return one();
    }
    return inv_logs[logn - 1];
  }

  static inline const gl64_t one() { return 1; }
  static inline const gl64_t zero() { return 0; }

  inline gl64_t() {}
  inline gl64_t(uint64_t a) : val(a) {}
  inline operator uint64_t() const { return val; }

  inline gl64_t &operator+=(gl64_t b) { return *this; }
  inline gl64_t &operator-=(gl64_t b) { return *this; }
  inline gl64_t &operator*=(gl64_t b) { return *this; }
  inline gl64_t &operator^=(int p) { return *this; }
  inline gl64_t &sqr() { return *this; }
  inline gl64_t reciprocal() const {}
  // inline void zero()                  { val = 0;      }
  inline uint32_t lo() const { return (uint32_t)(val); }
  inline uint32_t hi() const { return (uint32_t)(val >> 32); }
};
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif
typedef gl64_t fr_t;

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

// #endif
#endif