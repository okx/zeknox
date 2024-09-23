// Copyright 2024 OKX
#include "ff/goldilocks.hpp"

const fr_t group_gen = fr_t(0x0000000000000007);
const fr_t group_gen_inverse = fr_t(0x249249246db6db6e);

const int S = 32;



const fr_t forward_roots_of_unity[S + 1] = {
    fr_t(0x0000000000000001),
    fr_t(0xffffffff00000000),
    fr_t(0x0001000000000000),
    fr_t(0xfffffffeff000001),
    fr_t(0xefffffff00000001),
    fr_t(0x00003fffffffc000),
    fr_t(0x0000008000000000),
    fr_t(0xf80007ff08000001),
    fr_t(0xbf79143ce60ca966),
    fr_t(0x1905d02a5c411f4e),
    fr_t(0x9d8f2ad78bfed972),
    fr_t(0x0653b4801da1c8cf),
    fr_t(0xf2c35199959dfcb6),
    fr_t(0x1544ef2335d17997),
    fr_t(0xe0ee099310bba1e2),
    fr_t(0xf6b2cffe2306baac),
    fr_t(0x54df9630bf79450e),
    fr_t(0xabd0a6e8aa3d8a0e),
    fr_t(0x81281a7b05f9beac),
    fr_t(0xfbd41c6b8caa3302),
    fr_t(0x30ba2ecd5e93e76d),
    fr_t(0xf502aef532322654),
    fr_t(0x4b2a18ade67246b5),
    fr_t(0xea9d5a1336fbc98b),
    fr_t(0x86cdcc31c307e171),
    fr_t(0x4bbaf5976ecfefd8),
    fr_t(0xed41d05b78d6e286),
    fr_t(0x10d78dd8915a171d),
    fr_t(0x59049500004a4485),
    fr_t(0xdfa8c93ba46d2666),
    fr_t(0x7e9bd009b86a0845),
    fr_t(0x400a7f755588e659),  // rou_31, where  rou_32^{1<<31} = 1
    fr_t(0x185629dcda58878c),  // rou_32, where  rou_32^{1<<32} = 1
};

const fr_t inverse_roots_of_unity[S + 1] = {
    fr_t(0x0000000000000001),
    fr_t(0xffffffff00000000),
    fr_t(0xfffeffff00000001),
    fr_t(0x000000ffffffff00),
    fr_t(0x0000001000000000),
    fr_t(0xfffffffefffc0001),
    fr_t(0xfdffffff00000001),
    fr_t(0xffefffff00000011),
    fr_t(0x1d62e30fa4a4eeb0),
    fr_t(0x3de19c67cf496a74),
    fr_t(0x3b9ae9d1d8d87589),
    fr_t(0x76a40e0866a8e50d),
    fr_t(0x9af01e431fbd6ea0),
    fr_t(0x3712791d9eb0314a),
    fr_t(0x409730a1895adfb6),
    fr_t(0x158ee068c8241329),
    fr_t(0x6d341b1c9a04ed19),
    fr_t(0xcc9e5a57b8343b3f),
    fr_t(0x22e1fbf03f8b95d6),
    fr_t(0x46a23c48234c7df9),
    fr_t(0xef8856969fe6ed7b),
    fr_t(0xa52008ac564a2368),
    fr_t(0xd46e5a4c36458c11),
    fr_t(0x4bb9aee372cf655e),
    fr_t(0x10eb845263814db7),
    fr_t(0xc01f93fc71bb0b9b),
    fr_t(0xea52f593bb20759a),
    fr_t(0x91f3853f38e675d9),
    fr_t(0x3ea7eab8d8857184),
    fr_t(0xe4d14a114454645d),
    fr_t(0xe2434909eec4f00b),
    fr_t(0x95c0ec9a7ab50701),
    fr_t(0x76b6b635b6fc8719),  // rou_inv_32, where rou_inv_32 = rou_32^{-1}
};

const fr_t domain_size_inverse[S + 1] = {
    fr_t(0x0000000000000001),  // 1^{-1}
    fr_t(0x7fffffff80000001),  // 2^{-1}
    fr_t(0xbfffffff40000001),  // (1 << 2)^{-1}
    fr_t(0xdfffffff20000001),  // (1 << 3)^{-1}
    fr_t(0xefffffff10000001),
    fr_t(0xf7ffffff08000001),
    fr_t(0xfbffffff04000001),
    fr_t(0xfdffffff02000001),
    fr_t(0xfeffffff01000001),
    fr_t(0xff7fffff00800001),
    fr_t(0xffbfffff00400001),
    fr_t(0xffdfffff00200001),
    fr_t(0xffefffff00100001),
    fr_t(0xfff7ffff00080001),
    fr_t(0xfffbffff00040001),
    fr_t(0xfffdffff00020001),
    fr_t(0xfffeffff00010001),
    fr_t(0xffff7fff00008001),
    fr_t(0xffffbfff00004001),
    fr_t(0xffffdfff00002001),
    fr_t(0xffffefff00001001),
    fr_t(0xfffff7ff00000801),
    fr_t(0xfffffbff00000401),
    fr_t(0xfffffdff00000201),
    fr_t(0xfffffeff00000101),
    fr_t(0xffffff7f00000081),
    fr_t(0xffffffbf00000041),
    fr_t(0xffffffdf00000021),
    fr_t(0xffffffef00000011),
    fr_t(0xfffffff700000009),
    fr_t(0xfffffffb00000005),
    fr_t(0xfffffffd00000003),
    fr_t(0xfffffffe00000002),  // (1 << 32)^{-1}
};
