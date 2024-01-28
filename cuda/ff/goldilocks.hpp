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
# include "gl64_t.cuh"  // device-side field types

# ifndef __CUDA_ARCH__  // host-side stand-in to make CUDA code compile,
#  include <cstdint>    // not to produce correct result...

#  if defined(__GNUC__) || defined(__clang__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#  endif
class gl64_t {
public:
    uint64_t val;
public:
    using mem_t = gl64_t;
    static const uint32_t degree = 1;
    static const uint64_t MOD = 0xffffffff00000001U;
    static inline const gl64_t inv_log_size(uint32_t logn) {}
    static inline gl64_t omega(uint32_t logn) {}
    static inline gl64_t omega_inv(uint32_t logn) {}

    inline gl64_t()                     {}
    inline gl64_t(uint64_t a) : val(a)  {}
    inline operator uint64_t() const    { return val;   }
    static inline const gl64_t one()    { return 1;     }
    inline gl64_t& operator+=(gl64_t b) { return *this; }
    inline gl64_t& operator-=(gl64_t b) { return *this; }
    inline gl64_t& operator*=(gl64_t b) { return *this; }
    inline gl64_t& operator^=(int p)    { return *this; }
    inline gl64_t& sqr()                { return *this; }
    inline gl64_t reciprocal() const {}
    inline void zero()                  { val = 0;      }
     inline uint32_t lo() const { return (uint32_t)(val); }
    inline uint32_t hi() const { return (uint32_t)(val >> 32); }
};
#  if defined(__GNUC__) || defined(__clang__)
#   pragma GCC diagnostic pop
#  endif
# endif
typedef gl64_t fr_t;

    template<>
struct std::hash<gl64_t> {
  std::size_t operator()(const gl64_t& key) const
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