// Copyright Supranational LLC
// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __CRYPTO__PRIMITIVES__FIELD__CUH__
#define __CRYPTO__PRIMITIVES__FIELD__CUH__
#include <utils/storage.cuh>

namespace PARAMS_BN254 {
  struct fp_config {
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned omegas_count = 28;
    static constexpr unsigned modulus_bit_count = 254;
    static constexpr unsigned num_of_reductions = 1;

    static constexpr storage<limbs_count> modulus = {0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848,
                                                     0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> modulus_2 = {0xe0000002, 0x87c3eb27, 0xf372e122, 0x5067d090,
                                                       0x0302b0ba, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
    static constexpr storage<limbs_count> modulus_4 = {0xc0000004, 0x0f87d64f, 0xe6e5c245, 0xa0cfa121,
                                                       0x06056174, 0xe14116da, 0x84c680a6, 0xc19139cb};
    static constexpr storage<limbs_count> neg_modulus = {0x0fffffff, 0xbc1e0a6c, 0x86468f6e, 0xd7cc17b7,
                                                         0x7e7ea7a2, 0x47afba49, 0x1ece5fd6, 0xcf9bb18d};
    static constexpr storage<2 * limbs_count> modulus_wide = {
      0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2 * limbs_count> modulus_squared = {
      0xe0000001, 0x08c3eb27, 0xdcb34000, 0xc7f26223, 0x68c9bb7f, 0xffe9a62c, 0xe821ddb0, 0xa6ce1975,
      0x47b62fe7, 0x2c77527b, 0xd379d3df, 0x85f73bb0, 0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
    static constexpr storage<2 * limbs_count> modulus_squared_2 = {
      0xc0000002, 0x1187d64f, 0xb9668000, 0x8fe4c447, 0xd19376ff, 0xffd34c58, 0xd043bb61, 0x4d9c32eb,
      0x8f6c5fcf, 0x58eea4f6, 0xa6f3a7be, 0x0bee7761, 0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
    static constexpr storage<2 * limbs_count> modulus_squared_4 = {
      0x80000004, 0x230fac9f, 0x72cd0000, 0x1fc9888f, 0xa326edff, 0xffa698b1, 0xa08776c3, 0x9b3865d7,
      0x1ed8bf9e, 0xb1dd49ed, 0x4de74f7c, 0x17dceec3, 0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};

    static constexpr storage<limbs_count> m = {0xbe1de925, 0x620703a6, 0x09e880ae, 0x71448520,
                                               0x68073014, 0xab074a58, 0x623a04a7, 0x54a47462};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                  0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695,
                                                          0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x6db1194e, 0xdc5ba005, 0xe111ec87, 0x090ef5a9,
                                                              0xaeb85d5d, 0xc8260de4, 0x82c5551c, 0x15ebf951};

    static constexpr storage_array<omegas_count, limbs_count> omega = {
      {{0xf0000000, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72},
       {0x8f703636, 0x23120470, 0xfd736bec, 0x5cea24f6, 0x3fd84104, 0x048b6e19, 0xe131a029, 0x30644e72},
       {0xc1bd5e80, 0x948dad4a, 0xf8170a0a, 0x52627366, 0x96afef36, 0xec9b9e2f, 0xc8c14f22, 0x2b337de1},
       {0xe306460b, 0xb11509c6, 0x174efb98, 0x996dfbe1, 0x94dd508c, 0x1c6e4f45, 0x16cbbf4e, 0x21082ca2},
       {0x3bb512d0, 0x3eed4c53, 0x838eeb1d, 0x9c18d51b, 0x47c0b2a9, 0x9678200d, 0x306b93d2, 0x09c532c6},
       {0x118f023a, 0xdb94fb05, 0x26e324be, 0x46a6cb24, 0x49bdadf2, 0xc24cdb76, 0x5b080fca, 0x1418144d},
       {0xba9d1811, 0x9d0e470c, 0xb6f24c79, 0x1dcb5564, 0xe85943e0, 0xdf5ce19c, 0xad310991, 0x16e73dfd},
       {0x74a57a76, 0xc8936191, 0x6750f230, 0x61794254, 0x9f36ffb0, 0xf086204a, 0xa6148404, 0x07b0c561},
       {0x470157ce, 0x893a7fa1, 0xfc782d75, 0xe8302a41, 0xdd9b0675, 0xffc02c0e, 0xf6e72f5b, 0x0f1ded1e},
       {0xbc2e5912, 0x11f995e1, 0xa8d2d7ab, 0x39ba79c0, 0xb08771e3, 0xebbebc2b, 0x7017a420, 0x06fd19c1},
       {0x769a2ee2, 0xd00a58f9, 0x7494f0ca, 0xb8c12c17, 0xa5355d71, 0xb4027fd7, 0x99c5042b, 0x027a3584},
       {0x0042d43a, 0x1c477572, 0x6f039bb9, 0x76f169c7, 0xfd5a90a9, 0x01ddd073, 0xde2fd10f, 0x0931d596},
       {0x9bbdd310, 0x4aa49b8d, 0x8e3a2d76, 0xd31bf3e2, 0x78b2667b, 0x001deac8, 0xb869ae62, 0x006fab49},
       {0x617c6e85, 0xadaa01c2, 0x7420aae6, 0xb4a93ee1, 0x0ddca8a8, 0x1f4e51b8, 0xcdd9e481, 0x2d965651},
       {0x4e26ecfb, 0xa93458fd, 0x4115a009, 0x022a2a2d, 0x69ec2bd0, 0x017171fa, 0x5941dc91, 0x2d1ba66f},
       {0xdaac43b7, 0xd1628ba2, 0xe4347e7d, 0x16c8601d, 0xe081dcff, 0x649abebd, 0x5981ed45, 0x00eeb2cb},
       {0xce8f58e5, 0x276e5858, 0x5655210e, 0x0512eca9, 0xe70e61f3, 0xc3708cc6, 0xa7d74902, 0x1bf82deb},
       {0x7dcdc0e0, 0x84c6bfa5, 0x13f4d1bd, 0xc57088ff, 0xb5b95e4d, 0x5c0176fb, 0x3a8d46c1, 0x19ddbcaf},
       {0x613f6cbd, 0x5c1d597f, 0x8357473a, 0x30525841, 0x968e4915, 0x51829353, 0x844bca52, 0x2260e724},
       {0x53337857, 0x53422da9, 0xdbed349f, 0xac616632, 0x06d1e303, 0x27508aba, 0x0a0ed063, 0x26125da1},
       {0xfcd0b523, 0xb2c87885, 0xca5a5ce3, 0x58f50577, 0x8598fc8c, 0x4222150e, 0xae2bdd1a, 0x1ded8980},
       {0xa219447e, 0xa76dde56, 0x359eebbb, 0xec1a1f05, 0x8be08215, 0xcda0ceb6, 0xb1f8d9a7, 0x1ad92f46},
       {0xab80c59d, 0xb54d4506, 0x22dd991f, 0x5680c640, 0xbc23a139, 0x6b7bcf70, 0x5ab4c74d, 0x0210fe63},
       {0xe32b045b, 0x1c25f1e3, 0x2e832696, 0x145e0db8, 0x71c6441f, 0x852e2a03, 0x845d50d2, 0x0c9fabc7},
       {0xb878331a, 0xeccd4f3e, 0x8dc6d26e, 0x7b26b748, 0xd9130cd4, 0xa19b0361, 0x326341ef, 0x2a734ebb},
       {0x2f4e9212, 0x1c79bd57, 0x3d68f9ae, 0x605b52b6, 0xb8d89d4a, 0x0113eff9, 0xf1ff73b2, 0x1067569a},
       {0x80928c44, 0x034afc45, 0xf6437da2, 0xb4823532, 0x6dc6e364, 0x5f256a9f, 0xb363ebe8, 0x049ae702},
       {0x725b19f0, 0x9bd61b6e, 0x41112ed4, 0x402d111e, 0x8ef62abc, 0x00e0a7eb, 0xa58a7e85, 0x2a3c09f0}}};

    static constexpr storage_array<omegas_count, limbs_count> omega_inv = {
      {{0xf0000000, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72},
       {0x608fc9cb, 0x20cff123, 0x7c4604a5, 0xcb49c351, 0x41a91758, 0xb3c4d79d, 0x00000000, 0x00000000},
       {0x07b95a9b, 0x8b11d9ab, 0x41671f56, 0x20710ead, 0x30f81dee, 0xfb3acaee, 0x9778465c, 0x130b1711},
       {0x373428de, 0xb85a71e6, 0xaeb0337e, 0x74954d30, 0x303402b7, 0x2bfc85eb, 0x409556c0, 0x02e40daf},
       {0xf210979d, 0x8c99980c, 0x34905b4d, 0xef8f3113, 0xdf25d8e7, 0x0aeaf3e7, 0x03bfbd79, 0x27247136},
       {0x763d698f, 0x78ce6a0b, 0x1d3213ee, 0xd80396ec, 0x67a8a676, 0x035cdc75, 0xb2a13d3a, 0x26177cf2},
       {0xc64427d7, 0xdddf985f, 0xa49e95bd, 0xaa4f964a, 0x5def8b04, 0x427c045f, 0x7969b732, 0x1641c053},
       {0x0329f5d6, 0x692c553d, 0x8712848a, 0xa54cf8c6, 0x38e2b5e6, 0x64751ad9, 0x7422fad3, 0x204bd327},
       {0xaf6b3e4e, 0x52f26c0f, 0xf0bcc0c8, 0x4c277a07, 0xe4fcfcab, 0x546875d5, 0xaa9995b3, 0x09d8f821},
       {0xb2e5cc71, 0xcaa2e1e9, 0x6e43404e, 0xed42b68e, 0x7a2c7f0a, 0x6ed80915, 0xde3c86d6, 0x1c4042c7},
       {0x579d71ae, 0x20a3a65d, 0x0adc4420, 0xfd7efed8, 0xfddabf54, 0x3bb6dcd7, 0xbc73d07b, 0x0fa9bb21},
       {0xc79e0e57, 0xb6f70f8d, 0xa04e05ac, 0x269d3fde, 0x2ba088d9, 0xcf2e371c, 0x11b88d9c, 0x1af864d2},
       {0xabd95dc9, 0x3b0b205a, 0x978188ca, 0xc8df74fa, 0x6a1cb6c8, 0x08e124db, 0xbfac6104, 0x1670ed58},
       {0x641c8410, 0xf8eee934, 0x677771c0, 0xf40976b0, 0x558e6e8c, 0x11680d42, 0x06e7e9e9, 0x281c036f},
       {0xb2dbc0b4, 0xc92a742f, 0x4d384e68, 0xc3f02842, 0x2fa43d0d, 0x22701b6f, 0xe4590b37, 0x05d33766},
       {0x02d842d4, 0x922d5ac8, 0xc830e4c6, 0x91126414, 0x082f37e0, 0xe92338c0, 0x7fe704e8, 0x0b5d56b7},
       {0xd96f0d22, 0x20e75251, 0x6bd4e8c9, 0xc01c7f08, 0xf9dd50c4, 0x37d8b00b, 0xc43ca872, 0x244cf010},
       {0x66c5174c, 0x7a823174, 0x22d5ad70, 0x7dbe118c, 0x111119c5, 0xf8d7c71d, 0x83780e87, 0x036853f0},
       {0xca535321, 0xd98f9924, 0xe66e6c81, 0x22dbc0ef, 0x664ae1b7, 0xa15cf806, 0xa314fb67, 0x06e402c0},
       {0xe26c91f3, 0x0852a8fd, 0x3baca626, 0x521f45cb, 0x2c51bfca, 0xab6473bc, 0x2100895f, 0x100c332d},
       {0xa376d0f0, 0xf5fac783, 0x940797d3, 0x50fd246e, 0x145f5278, 0xab14ecc1, 0x41091b14, 0x19c6dfb8},
       {0x7faa1396, 0x43dc52e2, 0x4beced23, 0xd437be9d, 0x6d3c38c3, 0xecc11e9c, 0x0c74a876, 0x2eb58439},
       {0xd69ca83b, 0x811b03e7, 0xa1a6eadf, 0x126a786b, 0x4e2b8e61, 0x1dd75c9f, 0xbda6792b, 0x2165a1a5},
       {0x110b737b, 0x02e1d4d1, 0xb323a164, 0x7be1488d, 0x9cd06163, 0xa334d317, 0xdb50e9cd, 0x2710c370},
       {0x9550fe47, 0x45d2f3cb, 0xf6a8efc4, 0x5f43327b, 0xe993ee18, 0x5bcd0d50, 0xb21de952, 0x27f035bd},
       {0x232e3983, 0x1d63cbae, 0xaa1b58e2, 0xac815161, 0x6aeb019e, 0x531f42a5, 0x03ca2ef5, 0x2dcd51d9},
       {0x980db869, 0xa8b64ba8, 0xc9718f6c, 0x4c787f72, 0x15d27ced, 0x7746a25a, 0x435a46e9, 0x110bf78f},
       {0x9d18157e, 0x72394277, 0xfd399d5d, 0xec9d51f8, 0x49d5387f, 0x6117635d, 0x9c229cd5, 0x01b77519}}};

    static constexpr storage_array<omegas_count, limbs_count> inv = {
      {{0xf8000001, 0xa1f0fac9, 0x3cdcb848, 0x9419f424, 0x40c0ac2e, 0xdc2822db, 0x7098d014, 0x18322739},
       {0xf4000001, 0xf2e9782e, 0x5b4b146c, 0xde26ee36, 0xe1210245, 0x4a3c3448, 0x28e5381f, 0x244b3ad6},
       {0x72000001, 0x1b65b6e1, 0x6a82427f, 0x832d6b3f, 0xb1512d51, 0x81463cff, 0x850b6c24, 0x2a57c4a4},
       {0xb1000001, 0x2fa3d63a, 0xf21dd988, 0x55b0a9c3, 0x196942d7, 0x1ccb415b, 0xb31e8627, 0x2d5e098b},
       {0x50800001, 0xb9c2e5e7, 0x35eba50c, 0x3ef24906, 0xcd754d9a, 0x6a8dc388, 0x4a281328, 0x2ee12bff},
       {0xa0400001, 0xfed26dbd, 0x57d28ace, 0xb39318a7, 0xa77b52fb, 0x116f049f, 0x15acd9a9, 0x2fa2bd39},
       {0xc8200001, 0x215a31a8, 0xe8c5fdb0, 0x6de38077, 0x147e55ac, 0x64dfa52b, 0xfb6f3ce9, 0x300385d5},
       {0x5c100001, 0xb29e139e, 0x313fb720, 0xcb0bb460, 0xcaffd704, 0x8e97f570, 0x6e506e89, 0x3033ea24},
       {0x26080001, 0xfb400499, 0x557c93d8, 0xf99fce54, 0xa64097b0, 0xa3741d93, 0xa7c10759, 0x304c1c4b},
       {0x8b040001, 0x1f90fd16, 0x679b0235, 0x10e9db4e, 0x13e0f807, 0xade231a5, 0x447953c1, 0x3058355f},
       {0x3d820001, 0x31b97955, 0x70aa3963, 0x1c8ee1cb, 0xcab12832, 0xb3193bad, 0x12d579f5, 0x305e41e9},
       {0x96c10001, 0x3acdb774, 0xf531d4fa, 0xa2616509, 0x26194047, 0xb5b4c0b2, 0xfa038d0f, 0x3061482d},
       {0x43608001, 0xbf57d684, 0x3775a2c5, 0x654aa6a9, 0x53cd4c52, 0xb7028334, 0x6d9a969c, 0x3062cb50},
       {0x19b04001, 0x819ce60c, 0xd89789ab, 0xc6bf4778, 0x6aa75257, 0x37a96475, 0xa7661b63, 0x30638ce1},
       {0x04d82001, 0x62bf6dd0, 0xa9287d1e, 0x777997e0, 0xf614555a, 0x77fcd515, 0x444bddc6, 0x3063edaa},
       {0xfa6c1001, 0xd350b1b1, 0x9170f6d7, 0xcfd6c014, 0x3bcad6db, 0x18268d66, 0x92bebef8, 0x30641e0e},
       {0xf5360801, 0x8b9953a2, 0x859533b4, 0x7c05542e, 0x5ea6179c, 0xe83b698e, 0xb9f82f90, 0x30643640},
       {0x729b0401, 0xe7bda49b, 0x7fa75222, 0xd21c9e3b, 0x7013b7fc, 0x5045d7a2, 0xcd94e7dd, 0x30644259},
       {0xb14d8201, 0x15cfcd17, 0xfcb0615a, 0xfd284341, 0x78ca882c, 0x844b0eac, 0x57634403, 0x30644866},
       {0xd0a6c101, 0xacd8e155, 0x3b34e8f5, 0x12ae15c5, 0x7d25f045, 0x9e4daa31, 0x9c4a7216, 0x30644b6c},
       {0xe0536081, 0x785d6b74, 0xda772cc3, 0x1d70ff06, 0xff53a451, 0x2b4ef7f3, 0xbebe0920, 0x30644cef},
       {0x6829b041, 0x5e1fb084, 0xaa184eaa, 0x22d273a7, 0x406a7e57, 0xf1cf9ed5, 0x4ff7d4a4, 0x30644db1},
       {0x2c14d821, 0xd100d30c, 0x11e8df9d, 0x25832df8, 0xe0f5eb5a, 0x550ff245, 0x1894ba67, 0x30644e12},
       {0x0e0a6c11, 0x8a716450, 0x45d12817, 0xa6db8b20, 0x313ba1db, 0x86b01bfe, 0x7ce32d48, 0x30644e42},
       {0xff053609, 0x6729acf1, 0x5fc54c54, 0x6787b9b4, 0x595e7d1c, 0x1f8030da, 0xaf0a66b9, 0x30644e5a},
       {0xf7829b05, 0xd585d142, 0x6cbf5e72, 0xc7ddd0fe, 0x6d6feabc, 0x6be83b48, 0xc81e0371, 0x30644e66},
       {0x73c14d83, 0x0cb3e36b, 0x733c6782, 0xf808dca3, 0x7778a18c, 0x921c407f, 0xd4a7d1cd, 0x30644e6c},
       {0xb1e0a6c2, 0xa84aec7f, 0xf67aec09, 0x101e6275, 0xfc7cfcf5, 0xa536431a, 0xdaecb8fb, 0x30644e6f}}};
  };

  struct fq_config {
    static constexpr unsigned limbs_count = 8;
    static constexpr unsigned modulus_bit_count = 254;
    static constexpr unsigned num_of_reductions = 1;
    static constexpr storage<limbs_count> modulus = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91,
                                                     0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
    static constexpr storage<limbs_count> modulus_2 = {0xb0f9fa8e, 0x7841182d, 0xd0e3951a, 0x2f02d522,
                                                       0x0302b0bb, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
    static constexpr storage<limbs_count> modulus_4 = {0x61f3f51c, 0xf082305b, 0xa1c72a34, 0x5e05aa45,
                                                       0x06056176, 0xe14116da, 0x84c680a6, 0xc19139cb};
    static constexpr storage<limbs_count> neg_modulus = {0x278302b9, 0xc3df73e9, 0x978e3572, 0x687e956e,
                                                         0x7e7ea7a2, 0x47afba49, 0x1ece5fd6, 0xcf9bb18d};
    static constexpr storage<2 * limbs_count> modulus_wide = {
      0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72,
      0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<2 * limbs_count> modulus_squared = {
      0x275d69b1, 0x3b5458a2, 0x09eac101, 0xa602072d, 0x6d96cadc, 0x4a50189c, 0x7a1242c8, 0x04689e95,
      0x34c6b38d, 0x26edfa5c, 0x16375606, 0xb00b8551, 0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
    static constexpr storage<2 * limbs_count> modulus_squared_2 = {
      0x4ebad362, 0x76a8b144, 0x13d58202, 0x4c040e5a, 0xdb2d95b9, 0x94a03138, 0xf4248590, 0x08d13d2a,
      0x698d671a, 0x4ddbf4b8, 0x2c6eac0c, 0x60170aa2, 0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
    static constexpr storage<2 * limbs_count> modulus_squared_4 = {
      0x9d75a6c4, 0xed516288, 0x27ab0404, 0x98081cb4, 0xb65b2b72, 0x29406271, 0xe8490b21, 0x11a27a55,
      0xd31ace34, 0x9bb7e970, 0x58dd5818, 0xc02e1544, 0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};
    static constexpr storage<limbs_count> m = {0x19bf90e5, 0x6f3aed8a, 0x67cd4c08, 0xae965e17,
                                               0x68073013, 0xab074a58, 0x623a04a7, 0x54a47462};
    static constexpr storage<limbs_count> one = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> zero = {0x00000000, 0x00000000, 0x00000000, 0x00000000,
                                                  0x00000000, 0x00000000, 0x00000000, 0x00000000};
    static constexpr storage<limbs_count> montgomery_r = {0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28,
                                                          0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1};
    static constexpr storage<limbs_count> montgomery_r_inv = {0x014afa37, 0xed84884a, 0x0278edf8, 0xeb202285,
                                                              0xb74492d9, 0xcf63e9cf, 0x59e5c639, 0x2e671571};
    // i^2, the square of the imaginary unit for the extension field
    static constexpr uint32_t i_squared = 1;
    // true if i^2 is negative
    static constexpr bool i_squared_is_negative = true;
  };

  // G1 and G2 generators
  static constexpr storage<fq_config::limbs_count> g1_gen_x = {0x00000001, 0x00000000, 0x00000000, 0x00000000,
                                                               0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> g1_gen_y = {0x00000002, 0x00000000, 0x00000000, 0x00000000,
                                                               0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_re = {0xd992f6ed, 0x46debd5c, 0xf75edadd, 0x674322d4,
                                                                  0x5e5c4479, 0x426a0066, 0x121f1e76, 0x1800deef};
  static constexpr storage<fq_config::limbs_count> g2_gen_x_im = {0xaef312c2, 0x97e485b7, 0x35a9e712, 0xf1aa4933,
                                                                  0x31fb5d25, 0x7260bfb7, 0x920d483a, 0x198e9393};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_re = {0x66fa7daa, 0x4ce6cc01, 0x0c43d37b, 0xe3d1e769,
                                                                  0x8dcb408f, 0x4aab7180, 0xdb8c6deb, 0x12c85ea5};
  static constexpr storage<fq_config::limbs_count> g2_gen_y_im = {0xd122975b, 0x55acdadc, 0x70b38ef3, 0xbc4b3133,
                                                                  0x690c3395, 0xec9e99ad, 0x585ff075, 0x090689d0};

  static constexpr storage<fq_config::limbs_count> weierstrass_b = {0x00000003, 0x00000000, 0x00000000, 0x00000000,
                                                                    0x00000000, 0x00000000, 0x00000000, 0x00000000};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_re = {
    0x24a138e5, 0x3267e6dc, 0x59dbefa3, 0xb5b4c5e5, 0x1be06ac3, 0x81be1899, 0xceb8aaae, 0x2b149d40};
  static constexpr storage<fq_config::limbs_count> weierstrass_b_g2_im = {
    0x85c315d2, 0xe4a2bd06, 0xe52d1852, 0xa74fa084, 0xeed8fdf4, 0xcd2cafad, 0x3af0fed4, 0x009713b0};
} // namespace PARAMS_BN254
#endif