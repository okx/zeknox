#pragma once
#include "../../utils/storage.cuh"

namespace PARAMS_GL64 {
  struct fp_config {
    static constexpr unsigned limbs_count = 2;
    static constexpr unsigned omegas_count = 32;
    static constexpr unsigned modulus_bit_count = 64;

    static constexpr storage<limbs_count> modulus = {0x00000001, 0xffffffff};
    static constexpr storage<limbs_count> modulus_2 = {0x00000002, 0xfffffffe, 0x00000001};
    static constexpr storage<limbs_count> modulus_4 = {0x00000004, 0xfffffffc, 0x00000003};
    static constexpr storage<2*limbs_count> modulus_wide = {0x00000001, 0xffffffff, 0x00000000, 0x00000000};
    static constexpr storage<2*limbs_count> modulus_squared = {0x00000001, 0xfffffffe, 0x00000002, 0xfffffffe};
    static constexpr storage<2*limbs_count> modulus_squared_2 = {0x00000002, 0xfffffffc, 0x00000005, 0xfffffffc, 0x00000001};
    static constexpr storage<2*limbs_count> modulus_squared_4 = {0x00000004, 0xfffffff8, 0x0000000b, 0xfffffff8, 0x00000003};

    static constexpr storage<limbs_count> m = {0xffffffff, 0x00000000, 0x00000001};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0xffffffff, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x00000001, 0xfffffffe};

    static constexpr storage_array<omegas_count, limbs_count> omega = { {
              {0x00000000, 0xffffffff},
              {0x00000000, 0x00010000},
              {0xff000001, 0xfffffffe},
              {0x00000001, 0xefffffff},
              {0xffffc000, 0x00003fff},
              {0x00000000, 0x00000080},
              {0x08000001, 0xf80007ff},
              {0xe60ca966, 0xbf79143c},
              {0x5c411f4e, 0x1905d02a},
              {0x8bfed972, 0x9d8f2ad7},
              {0x1da1c8cf, 0x0653b480},
              {0x959dfcb6, 0xf2c35199},
              {0x35d17997, 0x1544ef23},
              {0x10bba1e2, 0xe0ee0993},
              {0x2306baac, 0xf6b2cffe},
              {0xbf79450e, 0x54df9630},
              {0xaa3d8a0e, 0xabd0a6e8},
              {0x05f9beac, 0x81281a7b},
              {0x8caa3302, 0xfbd41c6b},
              {0x5e93e76d, 0x30ba2ecd},
              {0x32322654, 0xf502aef5},
              {0xe67246b5, 0x4b2a18ad},
              {0x36fbc98b, 0xea9d5a13},
              {0xc307e171, 0x86cdcc31},
              {0x6ecfefd8, 0x4bbaf597},
              {0x78d6e286, 0xed41d05b},
              {0x915a171d, 0x10d78dd8},
              {0x004a4485, 0x59049500},
              {0xa46d2666, 0xdfa8c93b},
              {0xb86a0845, 0x7e9bd009},
              {0x5588e659, 0x400a7f75},
              {0xda58878c, 0x185629dc}
    } };


    static constexpr storage_array<omegas_count, limbs_count> omega_inv = { {
              {0x00000000, 0xffffffff},
              {0x00000001, 0xfffeffff},
              {0xffffff00, 0x000000ff},
              {0x00000000, 0x00000010},
              {0xfffc0001, 0xfffffffe},
              {0x00000001, 0xfdffffff},
              {0x00000011, 0xffefffff},
              {0xa4a4eeb0, 0x1d62e30f},
              {0xcf496a74, 0x3de19c67},
              {0xd8d87589, 0x3b9ae9d1},
              {0x66a8e50d, 0x76a40e08},
              {0x1fbd6ea0, 0x9af01e43},
              {0x9eb0314a, 0x3712791d},
              {0x895adfb6, 0x409730a1},
              {0xc8241329, 0x158ee068},
              {0x9a04ed19, 0x6d341b1c},
              {0xb8343b3f, 0xcc9e5a57},
              {0x3f8b95d6, 0x22e1fbf0},
              {0x234c7df9, 0x46a23c48},
              {0x9fe6ed7b, 0xef885696},
              {0x564a2368, 0xa52008ac},
              {0x36458c11, 0xd46e5a4c},
              {0x72cf655e, 0x4bb9aee3},
              {0x63814db7, 0x10eb8452},
              {0x71bb0b9b, 0xc01f93fc},
              {0xbb20759a, 0xea52f593},
              {0x38e675d9, 0x91f3853f},
              {0xd8857184, 0x3ea7eab8},
              {0x4454645d, 0xe4d14a11},
              {0xeec4f00b, 0xe2434909},
              {0x7ab50701, 0x95c0ec9a},
              {0xb6fc8719, 0x76b6b635}
    } };
    

    static constexpr storage_array<omegas_count, limbs_count> inv = { {
              {0x80000001, 0x7fffffff},
              {0x40000001, 0xbfffffff},
              {0x20000001, 0xdfffffff},
              {0x10000001, 0xefffffff},
              {0x08000001, 0xf7ffffff},
              {0x04000001, 0xfbffffff},
              {0x02000001, 0xfdffffff},
              {0x01000001, 0xfeffffff},
              {0x00800001, 0xff7fffff},
              {0x00400001, 0xffbfffff},
              {0x00200001, 0xffdfffff},
              {0x00100001, 0xffefffff},
              {0x00080001, 0xfff7ffff},
              {0x00040001, 0xfffbffff},
              {0x00020001, 0xfffdffff},
              {0x00010001, 0xfffeffff},
              {0x00008001, 0xffff7fff},
              {0x00004001, 0xffffbfff},
              {0x00002001, 0xffffdfff},
              {0x00001001, 0xffffefff},
              {0x00000801, 0xfffff7ff},
              {0x00000401, 0xfffffbff},
              {0x00000201, 0xfffffdff},
              {0x00000101, 0xfffffeff},
              {0x00000081, 0xffffff7f},
              {0x00000041, 0xffffffbf},
              {0x00000021, 0xffffffdf},
              {0x00000011, 0xffffffef},
              {0x00000009, 0xfffffff7},
              {0x00000005, 0xfffffffb},
              {0x00000003, 0xfffffffd},
              {0x00000002, 0xfffffffe}
    } }; 
  };

  struct fq_config {
    static constexpr unsigned limbs_count = None;
    static constexpr unsigned modulus_bit_count = None;
    static constexpr storage<limbs_count> modulus = {${fq_modulus}};
    static constexpr storage<limbs_count> modulus_2 = {${fq_modulus_2}};
    static constexpr storage<limbs_count> modulus_4 = {${fq_modulus_4}};
    static constexpr storage<2*limbs_count> modulus_wide = {${fq_modulus_wide}};
    static constexpr storage<2*limbs_count> modulus_squared = {${fq_modulus_squared}};
    static constexpr storage<2*limbs_count> modulus_squared_2 = {${fq_modulus_squared_2}};
    static constexpr storage<2*limbs_count> modulus_squared_4 = {${fq_modulus_squared_4}};
    static constexpr storage<limbs_count> m = {${fq_m}};
    static constexpr storage<limbs_count> one = {${fq_one}};
    static constexpr storage<limbs_count> zero = {${fq_zero}};
    static constexpr storage<limbs_count> montgomery_r = {${fq_montgomery_r}};
    static constexpr storage<limbs_count> montgomery_r_inv = {${fq_montgomery_r_inv}};
    // i^2, the square of the imaginary unit for the extension field
    static constexpr uint32_t i_squared = ${nonresidue};
    // true if i^2 is negative
    static constexpr bool i_squared_is_negative = ${nonresidue_is_negative};
  };

  // G1 and G2 generators
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {${fq_gen_x}};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {${fq_gen_y}};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_re = {${fq_gen_x_re}};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_im = {${fq_gen_x_im}};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_re = {${fq_gen_y_re}};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_im = {${fq_gen_y_im}};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {${weier_b}};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_re = {${weier_b_g2_re}};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_im = {${weier_b_g2_im}};
}