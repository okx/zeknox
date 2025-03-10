// Copyright (c) 2023 Ingonyama
// under MIT License

#ifndef __CRYPTO_PRIMITIVES_AFFINE_CUH__
#define __CRYPTO_PRIMITIVES_AFFINE_CUH__
#include "field.cuh"

template <class FF>
class Affine
{
public:
  FF x;
  FF y;

  static HOST_DEVICE_INLINE Affine neg(const Affine& point) { return {point.x, FF::neg(point.y)}; }

  static HOST_DEVICE_INLINE Affine to_montgomery(const Affine& point)
  {
    return {FF::to_montgomery(point.x), FF::to_montgomery(point.y)};
  }

  static HOST_DEVICE_INLINE Affine from_montgomery(const Affine& point)
  {
    return {FF::from_montgomery(point.x), FF::from_montgomery(point.y)};
  }

  friend HOST_DEVICE_INLINE bool operator==(const Affine& xs, const Affine& ys)
  {
    return (xs.x == ys.x) && (xs.y == ys.y);
  }

  friend HOST_INLINE std::ostream& operator<<(std::ostream& os, const Affine& point)
  {
    os << "x: " << point.x << "; y: " << point.y;
    return os;
  }
};
#endif