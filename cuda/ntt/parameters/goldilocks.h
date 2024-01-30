#include "ff/goldilocks.hpp"

const fr_t group_gen = fr_t(0x0000000000000007);
const fr_t group_gen_inverse = fr_t(0x249249246db6db6e);

const int S = 32;

#ifdef __X1_PROVER__
// copy from https://github.com/okx/goldilocks/blob/f89eb016830f8c4301482d83691aed22e5a92742/src/goldilocks_base_field.cpp#L11
const fr_t forward_roots_of_unity[S + 1] = {
    fr_t(0x1),
    fr_t(18446744069414584320ULL),
    fr_t(281474976710656ULL),
    fr_t(16777216ULL),
    fr_t(4096ULL),
    fr_t(64ULL),
    fr_t(8ULL),
    fr_t(2198989700608ULL),
    fr_t(4404853092538523347ULL),
    fr_t(6434636298004421797ULL),
    fr_t(4255134452441852017ULL),
    fr_t(9113133275150391358ULL),
    fr_t(4355325209153869931ULL),
    fr_t(4308460244895131701ULL),
    fr_t(7126024226993609386ULL),
    fr_t(1873558160482552414ULL),
    fr_t(8167150655112846419ULL),
    fr_t(5718075921287398682ULL),
    fr_t(3411401055030829696ULL),
    fr_t(8982441859486529725ULL),
    fr_t(1971462654193939361ULL),
    fr_t(6553637399136210105ULL),
    fr_t(8124823329697072476ULL),
    fr_t(5936499541590631774ULL),
    fr_t(2709866199236980323ULL),
    fr_t(8877499657461974390ULL),
    fr_t(3757607247483852735ULL),
    fr_t(4969973714567017225ULL),
    fr_t(2147253751702802259ULL),
    fr_t(2530564950562219707ULL),
    fr_t(1905180297017055339ULL),
    fr_t(3524815499551269279ULL),
    fr_t(7277203076849721926ULL),
};
// modular inverse of `forward_roots_of_unity`
// inverse_roots_of_unity[i] = forward_roots_of_unity[i] ^ (p-2) % p
// p = 2^64 - 2^32 + 1
// calculated by sageMath
const fr_t inverse_roots_of_unity[S + 1] = {
    fr_t(0x0000000000000001),
    fr_t(0xffffffff00000000),
    fr_t(0xfffeffff00000001),
    fr_t(0xfffffeff00000101),
    fr_t(0xffefffff00100001),
    fr_t(0xfbffffff04000001),
    fr_t(0xdfffffff20000001),
    fr_t(0x3fffbfffc0),
    fr_t(0x7f4949dce07bf05d),
    fr_t(0x4bd6bb172e15d48c),
    fr_t(0x38bc97652b54c741),
    fr_t(0x553a9b711648c890),
    fr_t(0x55da9bb68958caa),
    fr_t(0xa0a62f8f0bb8e2b6),
    fr_t(0x276fd7ae450aee4b),
    fr_t(0x7b687b64f5de658f),
    fr_t(0x7de5776cbda187e9),
    fr_t(0xd2199b156a6f3b06),
    fr_t(0xd01c8acd8ea0e8c0),
    fr_t(0x4f38b2439950a4cf),
    fr_t(0x5987c395dd5dfdcf),
    fr_t(0x46cf3d56125452b1),
    fr_t(0x909c4b1a44a69ccb),
    fr_t(0xc188678a32a54199),
    fr_t(0xf3650f9ddfcaffa8),
    fr_t(0xe8ef0e3e40a92655),
    fr_t(0x7c8abec072bb46a6),
    fr_t(0xe0bfc17d5c5a7a04),
    fr_t(0x4c6b8a5a0b79f23a),
    fr_t(0x6b4d20533ce584fe),
    fr_t(0xe5cceae468a70ec2),
    fr_t(0x8958579f296dac7a),
    fr_t(0x16d265893b5b7e85),
};
#else
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
#endif

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
