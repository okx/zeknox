// Copyright 2024 OKX Group
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "poseidon/poseidon_permutation.cuh"

__device__ u32 GPU_MDS_MATRIX_CIRC[12] = {17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20};

__device__ u32 GPU_MDS_MATRIX_DIAG[12] = {8, (u64)0u, (u64)0u, (u64)0u, (u64)0u, (u64)0u, (u64)0u, (u64)0u, (u64)0u, (u64)0u, (u64)0u, (u64)0u};

__device__ u64 GPU_ALL_ROUND_CONSTANTS[MAX_WIDTH * N_ROUNDS] = {
    0xb585f766f2144405,
    0x7746a55f43921ad7,
    0xb2fb0d31cee799b4,
    0x0f6760a4803427d7,
    0xe10d666650f4e012,
    0x8cae14cb07d09bf1,
    0xd438539c95f63e9f,
    0xef781c7ce35b4c3d,
    0xcdc4a239b0c44426,
    0x277fa208bf337bff,
    0xe17653a29da578a1,
    0xc54302f225db2c76,
    0x86287821f722c881,
    0x59cd1a8a41c18e55,
    0xc3b919ad495dc574,
    0xa484c4c5ef6a0781,
    0x308bbd23dc5416cc,
    0x6e4a40c18f30c09c,
    0x9a2eedb70d8f8cfa,
    0xe360c6e0ae486f38,
    0xd5c7718fbfc647fb,
    0xc35eae071903ff0b,
    0x849c2656969c4be7,
    0xc0572c8c08cbbbad,
    0xe9fa634a21de0082,
    0xf56f6d48959a600d,
    0xf7d713e806391165,
    0x8297132b32825daf,
    0xad6805e0e30b2c8a,
    0xac51d9f5fcf8535e,
    0x502ad7dc18c2ad87,
    0x57a1550c110b3041,
    0x66bbd30e6ce0e583,
    0x0da2abef589d644e,
    0xf061274fdb150d61,
    0x28b8ec3ae9c29633,
    0x92a756e67e2b9413,
    0x70e741ebfee96586,
    0x019d5ee2af82ec1c,
    0x6f6f2ed772466352,
    0x7cf416cfe7e14ca1,
    0x61df517b86a46439,
    0x85dc499b11d77b75,
    0x4b959b48b9c10733,
    0xe8be3e5da8043e57,
    0xf5c0bc1de6da8699,
    0x40b12cbf09ef74bf,
    0xa637093ecb2ad631,
    0x3cc3f892184df408,
    0x2e479dc157bf31bb,
    0x6f49de07a6234346,
    0x213ce7bede378d7b,
    0x5b0431345d4dea83,
    0xa2de45780344d6a1,
    0x7103aaf94a7bf308,
    0x5326fc0d97279301,
    0xa9ceb74fec024747,
    0x27f8ec88bb21b1a3,
    0xfceb4fda1ded0893,
    0xfac6ff1346a41675,
    0x7131aa45268d7d8c,
    0x9351036095630f9f,
    0xad535b24afc26bfb,
    0x4627f5c6993e44be,
    0x645cf794b8f1cc58,
    0x241c70ed0af61617,
    0xacb8e076647905f1,
    0x3737e9db4c4f474d,
    0xe7ea5e33e75fffb6,
    0x90dee49fc9bfc23a,
    0xd1b1edf76bc09c92,
    0x0b65481ba645c602,
    0x99ad1aab0814283b,
    0x438a7c91d416ca4d,
    0xb60de3bcc5ea751c,
    0xc99cab6aef6f58bc,
    0x69a5ed92a72ee4ff,
    0x5e7b329c1ed4ad71,
    0x5fc0ac0800144885,
    0x32db829239774eca,
    0x0ade699c5830f310,
    0x7cc5583b10415f21,
    0x85df9ed2e166d64f,
    0x6604df4fee32bcb1,
    0xeb84f608da56ef48,
    0xda608834c40e603d,
    0x8f97fe408061f183,
    0xa93f485c96f37b89,
    0x6704e8ee8f18d563,
    0xcee3e9ac1e072119,
    0x510d0e65e2b470c1,
    0xf6323f486b9038f0,
    0x0b508cdeffa5ceef,
    0xf2417089e4fb3cbd,
    0x60e75c2890d15730,
    0xa6217d8bf660f29c,
    0x7159cd30c3ac118e,
    0x839b4e8fafead540,
    0x0d3f3e5e82920adc,
    0x8f7d83bddee7bba8,
    0x780f2243ea071d06,
    0xeb915845f3de1634,
    0xd19e120d26b6f386,
    0x016ee53a7e5fecc6,
    0xcb5fd54e7933e477,
    0xacb8417879fd449f,
    0x9c22190be7f74732,
    0x5d693c1ba3ba3621,
    0xdcef0797c2b69ec7,
    0x3d639263da827b13,
    0xe273fd971bc8d0e7,
    0x418f02702d227ed5,
    0x8c25fda3b503038c,
    0x2cbaed4daec8c07c,
    0x5f58e6afcdd6ddc2,
    0x284650ac5e1b0eba,
    0x635b337ee819dab5,
    0x9f9a036ed4f2d49f,
    0xb93e260cae5c170e,
    0xb0a7eae879ddb76d,
    0xd0762cbc8ca6570c,
    0x34c6efb812b04bf5,
    0x40bf0ab5fa14c112,
    0xb6b570fc7c5740d3,
    0x5a27b9002de33454,
    0xb1a5b165b6d2b2d2,
    0x8722e0ace9d1be22,
    0x788ee3b37e5680fb,
    0x14a726661551e284,
    0x98b7672f9ef3b419,
    0xbb93ae776bb30e3a,
    0x28fd3b046380f850,
    0x30a4680593258387,
    0x337dc00c61bd9ce1,
    0xd5eca244c7a4ff1d,
    0x7762638264d279bd,
    0xc1e434bedeefd767,
    0x0299351a53b8ec22,
    0xb2d456e4ad251b80,
    0x3e9ed1fda49cea0b,
    0x2972a92ba450bed8,
    0x20216dd77be493de,
    0xadffe8cf28449ec6,
    0x1c4dbb1c4c27d243,
    0x15a16a8a8322d458,
    0x388a128b7fd9a609,
    0x2300e5d6baedf0fb,
    0x2f63aa8647e15104,
    0xf1c36ce86ecec269,
    0x27181125183970c9,
    0xe584029370dca96d,
    0x4d9bbc3e02f1cfb2,
    0xea35bc29692af6f8,
    0x18e21b4beabb4137,
    0x1e3b9fc625b554f4,
    0x25d64362697828fd,
    0x5a3f1bb1c53a9645,
    0xdb7f023869fb8d38,
    0xb462065911d4e1fc,
    0x49c24ae4437d8030,
    0xd793862c112b0566,
    0xaadd1106730d8feb,
    0xc43b6e0e97b0d568,
    0xe29024c18ee6fca2,
    0x5e50c27535b88c66,
    0x10383f20a4ff9a87,
    0x38e8ee9d71a45af8,
    0xdd5118375bf1a9b9,
    0x775005982d74d7f7,
    0x86ab99b4dde6c8b0,
    0xb1204f603f51c080,
    0xef61ac8470250ecf,
    0x1bbcd90f132c603f,
    0x0cd1dabd964db557,
    0x11a3ae5beb9d1ec9,
    0xf755bfeea585d11d,
    0xa3b83250268ea4d7,
    0x516306f4927c93af,
    0xddb4ac49c9efa1da,
    0x64bb6dec369d4418,
    0xf9cc95c22b4c1fcc,
    0x08d37f755f4ae9f6,
    0xeec49b613478675b,
    0xf143933aed25e0b0,
    0xe4c5dd8255dfc622,
    0xe7ad7756f193198e,
    0x92c2318b87fff9cb,
    0x739c25f8fd73596d,
    0x5636cac9f16dfed0,
    0xdd8f909a938e0172,
    0xc6401fe115063f5b,
    0x8ad97b33f1ac1455,
    0x0c49366bb25e8513,
    0x0784d3d2f1698309,
    0x530fb67ea1809a81,
    0x410492299bb01f49,
    0x139542347424b9ac,
    0x9cb0bd5ea1a1115e,
    0x02e3f615c38f49a1,
    0x985d4f4a9c5291ef,
    0x775b9feafdcd26e7,
    0x304265a6384f0f2d,
    0x593664c39773012c,
    0x4f0a2e5fb028f2ce,
    0xdd611f1000c17442,
    0xd8185f9adfea4fd0,
    0xef87139ca9a3ab1e,
    0x3ba71336c34ee133,
    0x7d3a455d56b70238,
    0x660d32e130182684,
    0x297a863f48cd1f43,
    0x90e0a736a751ebb7,
    0x549f80ce550c4fd3,
    0x0f73b2922f38bd64,
    0x16bf1f73fb7a9c3f,
    0x6d1f5a59005bec17,
    0x02ff876fa5ef97c4,
    0xc5cb72a2a51159b0,
    0x8470f39d2d5c900e,
    0x25abb3f1d39fcb76,
    0x23eb8cc9b372442f,
    0xd687ba55c64f6364,
    0xda8d9e90fd8ff158,
    0xe3cbdc7d2fe45ea7,
    0xb9a8c9b3aee52297,
    0xc0d28a5c10960bd3,
    0x45d7ac9b68f71a34,
    0xeeb76e397069e804,
    0x3d06c8bd1514e2d9,
    0x9c9c98207cb10767,
    0x65700b51aedfb5ef,
    0x911f451539869408,
    0x7ae6849fbc3a0ec6,
    0x3bb340eba06afe7e,
    0xb46e9d8b682ea65e,
    0x8dcf22f9a3b34356,
    0x77bdaeda586257a7,
    0xf19e400a5104d20d,
    0xc368a348e46d950f,
    0x9ef1cd60e679f284,
    0xe89cd854d5d01d33,
    0x5cd377dc8bb882a2,
    0xa7b0fb7883eee860,
    0x7684403ec392950d,
    0x5fa3f06f4fed3b52,
    0x8df57ac11bc04831,
    0x2db01efa1e1e1897,
    0x54846de4aadb9ca2,
    0xba6745385893c784,
    0x541d496344d2c75b,
    0xe909678474e687fe,
    0xdfe89923f6c9c2ff,
    0xece5a71e0cfedc75,
    0x5ff98fd5d51fe610,
    0x83e8941918964615,
    0x5922040b47f150c1,
    0xf97d750e3dd94521,
    0x5080d4c2b86f56d7,
    0xa7de115b56c78d70,
    0x6a9242ac87538194,
    0xf7856ef7f9173e44,
    0x2265fc92feb0dc09,
    0x17dfc8e4f7ba8a57,
    0x9001a64209f21db8,
    0x90004c1371b893c5,
    0xb932b7cf752e5545,
    0xa0b1df81b6fe59fc,
    0x8ef1dd26770af2c2,
    0x0541a4f9cfbeed35,
    0x9e61106178bfc530,
    0xb3767e80935d8af2,
    0x0098d5782065af06,
    0x31d191cd5c1466c7,
    0x410fefafa319ac9d,
    0xbdf8f242e316c4ab,
    0x9e8cd55b57637ed0,
    0xde122bebe9a39368,
    0x4d001fd58f002526,
    0xca6637000eb4a9f8,
    0x2f2339d624f91f78,
    0x6d1a7918c80df518,
    0xdf9a4939342308e9,
    0xebc2151ee6c8398c,
    0x03cc2ba8a1116515,
    0xd341d037e840cf83,
    0x387cb5d25af4afcc,
    0xbba2515f22909e87,
    0x7248fe7705f38e47,
    0x4d61e56a525d225a,
    0x262e963c8da05d3d,
    0x59e89b094d220ec2,
    0x055d5b52b78b9c5e,
    0x82b27eb33514ef99,
    0xd30094ca96b7ce7b,
    0xcf5cb381cd0a1535,
    0xfeed4db6919e5a7c,
    0x41703f53753be59f,
    0x5eeea940fcde8b6f,
    0x4cd1f1b175100206,
    0x4a20358574454ec0,
    0x1478d361dbbf9fac,
    0x6f02dc07d141875c,
    0x296a202ed8e556a2,
    0x2afd67999bf32ee5,
    0x7acfd96efa95491d,
    0x6798ba0c0abb2c6d,
    0x34c6f57b26c92122,
    0x5736e1bad206b5de,
    0x20057d2a0056521b,
    0x3dea5bd5d0578bd7,
    0x16e50d897d4634ac,
    0x29bff3ecb9b7a6e3,
    0x475cd3205a3bdcde,
    0x18a42105c31b7e88,
    0x023e7414af663068,
    0x15147108121967d7,
    0xe4a3dff1d7d6fef9,
    0x01a8d1a588085737,
    0x11b4c74eda62beef,
    0xe587cc0d69a73346,
    0x1ff7327017aa2a6e,
    0x594e29c42473d06b,
    0xf6f31db1899b12d5,
    0xc02ac5e47312d3ca,
    0xe70201e960cb78b8,
    0x6f90ff3b6a65f108,
    0x42747a7245e7fa84,
    0xd1f507e43ab749b2,
    0x1c86d265f15750cd,
    0x3996ce73dd832c1c,
    0x8e7fba02983224bd,
    0xba0dec7103255dd4,
    0x9e9cbd781628fc5b,
    0xdae8645996edd6a5,
    0xdebe0853b1a1d378,
    0xa49229d24d014343,
    0x7be5b9ffda905e1c,
    0xa3c95eaec244aa30,
    0x0230bca8f4df0544,
    0x4135c2bebfe148c6,
    0x166fc0cc438a3c72,
    0x3762b59a8ae83efa,
    0xe8928a4c89114750,
    0x2a440b51a4945ee5,
    0x80cefd2b7d99ff83,
    0xbb9879c6e61fd62a,
    0x6e7c8f1a84265034,
    0x164bb2de1bbeddc8,
    0xf3c12fe54d5c653b,
    0x40b9e922ed9771e2,
    0x551f5b0fbe7b1840,
    0x25032aa7c4cb1811,
    0xaaed34074b164346,
    0x8ffd96bbf9c9c81d,
    0x70fc91eb5937085c,
    0x7f795e2a5f915440,
    0x4543d9df5476d3cb,
    0xf172d73e004fc90d,
    0xdfd1c4febcc81238,
    0xbc8dfb627fe558fc,
};

__device__ u64 GPU_FAST_PARTIAL_ROUND_CONSTANTS[N_PARTIAL_ROUNDS] = {
    0x74cb2e819ae421ab, 0xd2559d2370e7f663, 0x62bf78acf843d17c, 0xd5ab7b67e14d1fb4,
    0xb9fe2ae6e0969bdc, 0xe33fdf79f92a10e8, 0x0ea2bb4c2b25989b, 0xca9121fbf9d38f06,
    0xbdd9b0aa81f58fa4, 0x83079fa4ecf20d7e, 0x650b838edfcc4ad3, 0x77180c88583c76ac,
    0xaf8c20753143a180, 0xb8ccfe9989a39175, 0x954a1729f60cc9c5, 0xdeb5b550c4dca53b,
    0xf01bb0b00f77011e, 0xa1ebb404b676afd9, 0x860b6e1597a0173e, 0x308bb65a036acbce,
    0x1aca78f31c97c876, 0x0};

__device__ u64 GPU_FAST_PARTIAL_FIRST_ROUND_CONSTANT[12] = {
    0x3cc3f892184df408, 0xe993fd841e7e97f1, 0xf2831d3575f0f3af, 0xd2500e0a350994ca,
    0xc5571f35d7288633, 0x91d89c5184109a02, 0xf37f925d04e5667b, 0x2d6e448371955a69,
    0x740ef19ce01398a1, 0x694d24c0752fdf45, 0x60936af96ee2f148, 0xc33448feadc78f0c};

__device__ u64 GPU_FAST_PARTIAL_ROUND_VS[N_PARTIAL_ROUNDS][11] = {
    {0x94877900674181c3, 0xc6c67cc37a2a2bbd, 0xd667c2055387940f, 0x0ba63a63e94b5ff0,
     0x99460cc41b8f079f, 0x7ff02375ed524bb3, 0xea0870b47a8caf0e, 0xabcad82633b7bc9d,
     0x3b8d135261052241, 0xfb4515f5e5b0d539, 0x3ee8011c2b37f77c},
    {0x0adef3740e71c726, 0xa37bf67c6f986559, 0xc6b16f7ed4fa1b00, 0x6a065da88d8bfc3c,
     0x4cabc0916844b46f, 0x407faac0f02e78d1, 0x07a786d9cf0852cf, 0x42433fb6949a629a,
     0x891682a147ce43b0, 0x26cfd58e7b003b55, 0x2bbf0ed7b657acb3},
    {0x481ac7746b159c67, 0xe367de32f108e278, 0x73f260087ad28bec, 0x5cfc82216bc1bdca,
     0xcaccc870a2663a0e, 0xdb69cd7b4298c45d, 0x7bc9e0c57243e62d, 0x3cc51c5d368693ae,
     0x366b4e8cc068895b, 0x2bd18715cdabbca4, 0xa752061c4f33b8cf},
    {0xb22d2432b72d5098, 0x9e18a487f44d2fe4, 0x4b39e14ce22abd3c, 0x9e77fde2eb315e0d,
     0xca5e0385fe67014d, 0x0c2cb99bf1b6bddb, 0x99ec1cd2a4460bfe, 0x8577a815a2ff843f,
     0x7d80a6b4fd6518a5, 0xeb6c67123eab62cb, 0x8f7851650eca21a5},
    {0x11ba9a1b81718c2a, 0x9f7d798a3323410c, 0xa821855c8c1cf5e5, 0x535e8d6fac0031b2,
     0x404e7c751b634320, 0xa729353f6e55d354, 0x4db97d92e58bb831, 0xb53926c27897bf7d,
     0x965040d52fe115c5, 0x9565fa41ebd31fd7, 0xaae4438c877ea8f4},
    {0x37f4e36af6073c6e, 0x4edc0918210800e9, 0xc44998e99eae4188, 0x9f4310d05d068338,
     0x9ec7fe4350680f29, 0xc5b2c1fdc0b50874, 0xa01920c5ef8b2ebe, 0x59fa6f8bd91d58ba,
     0x8bfc9eb89b515a82, 0xbe86a7a2555ae775, 0xcbb8bbaa3810babf},
    {0x577f9a9e7ee3f9c2, 0x88c522b949ace7b1, 0x82f07007c8b72106, 0x8283d37c6675b50e,
     0x98b074d9bbac1123, 0x75c56fb7758317c1, 0xfed24e206052bc72, 0x26d7c3d1bc07dae5,
     0xf88c5e441e28dbb4, 0x4fe27f9f96615270, 0x514d4ba49c2b14fe},
    {0xf02a3ac068ee110b, 0x0a3630dafb8ae2d7, 0xce0dc874eaf9b55c, 0x9a95f6cff5b55c7e,
     0x626d76abfed00c7b, 0xa0c1cf1251c204ad, 0xdaebd3006321052c, 0x3d4bd48b625a8065,
     0x7f1e584e071f6ed2, 0x720574f0501caed3, 0xe3260ba93d23540a},
    {0xab1cbd41d8c1e335, 0x9322ed4c0bc2df01, 0x51c3c0983d4284e5, 0x94178e291145c231,
     0xfd0f1a973d6b2085, 0xd427ad96e2b39719, 0x8a52437fecaac06b, 0xdc20ee4b8c4c9a80,
     0xa2c98e9549da2100, 0x1603fe12613db5b6, 0x0e174929433c5505},
    {0x3d4eab2b8ef5f796, 0xcfff421583896e22, 0x4143cb32d39ac3d9, 0x22365051b78a5b65,
     0x6f7fd010d027c9b6, 0xd9dd36fba77522ab, 0xa44cf1cb33e37165, 0x3fc83d3038c86417,
     0xc4588d418e88d270, 0xce1320f10ab80fe2, 0xdb5eadbbec18de5d},
    {0x1183dfce7c454afd, 0x21cea4aa3d3ed949, 0x0fce6f70303f2304, 0x19557d34b55551be,
     0x4c56f689afc5bbc9, 0xa1e920844334f944, 0xbad66d423d2ec861, 0xf318c785dc9e0479,
     0x99e2032e765ddd81, 0x400ccc9906d66f45, 0xe1197454db2e0dd9},
    {0x84d1ecc4d53d2ff1, 0xd8af8b9ceb4e11b6, 0x335856bb527b52f4, 0xc756f17fb59be595,
     0xc0654e4ea5553a78, 0x9e9a46b61f2ea942, 0x14fc8b5b3b809127, 0xd7009f0f103be413,
     0x3e0ee7b7a9fb4601, 0xa74e888922085ed7, 0xe80a7cde3d4ac526},
    {0x238aa6daa612186d, 0x9137a5c630bad4b4, 0xc7db3817870c5eda, 0x217e4f04e5718dc9,
     0xcae814e2817bd99d, 0xe3292e7ab770a8ba, 0x7bb36ef70b6b9482, 0x3c7835fb85bca2d3,
     0xfe2cdf8ee3c25e86, 0x61b3915ad7274b20, 0xeab75ca7c918e4ef},
    {0xd6e15ffc055e154e, 0xec67881f381a32bf, 0xfbb1196092bf409c, 0xdc9d2e07830ba226,
     0x0698ef3245ff7988, 0x194fae2974f8b576, 0x7a5d9bea6ca4910e, 0x7aebfea95ccdd1c9,
     0xf9bd38a67d5f0e86, 0xfa65539de65492d8, 0xf0dfcbe7653ff787},
    {0x0bd87ad390420258, 0x0ad8617bca9e33c8, 0x0c00ad377a1e2666, 0x0ac6fc58b3f0518f,
     0x0c0cc8a892cc4173, 0x0c210accb117bc21, 0x0b73630dbb46ca18, 0x0c8be4920cbd4a54,
     0x0bfe877a21be1690, 0x0ae790559b0ded81, 0x0bf50db2f8d6ce31},
    {0x000cf29427ff7c58, 0x000bd9b3cf49eec8, 0x000d1dc8aa81fb26, 0x000bc792d5c394ef,
     0x000d2ae0b2266453, 0x000d413f12c496c1, 0x000c84128cfed618, 0x000db5ebd48fc0d4,
     0x000d1b77326dcb90, 0x000beb0ccc145421, 0x000d10e5b22b11d1},
    {0x00000e24c99adad8, 0x00000cf389ed4bc8, 0x00000e580cbf6966, 0x00000cde5fd7e04f,
     0x00000e63628041b3, 0x00000e7e81a87361, 0x00000dabe78f6d98, 0x00000efb14cac554,
     0x00000e5574743b10, 0x00000d05709f42c1, 0x00000e4690c96af1},
    {0x0000000f7157bc98, 0x0000000e3006d948, 0x0000000fa65811e6, 0x0000000e0d127e2f,
     0x0000000fc18bfe53, 0x0000000fd002d901, 0x0000000eed6461d8, 0x0000001068562754,
     0x0000000fa0236f50, 0x0000000e3af13ee1, 0x0000000fa460f6d1},
    {0x0000000011131738, 0x000000000f56d588, 0x0000000011050f86, 0x000000000f848f4f,
     0x00000000111527d3, 0x00000000114369a1, 0x00000000106f2f38, 0x0000000011e2ca94,
     0x00000000110a29f0, 0x000000000fa9f5c1, 0x0000000010f625d1},
    {0x000000000011f718, 0x000000000010b6c8, 0x0000000000134a96, 0x000000000010cf7f,
     0x0000000000124d03, 0x000000000013f8a1, 0x0000000000117c58, 0x0000000000132c94,
     0x0000000000134fc0, 0x000000000010a091, 0x0000000000128961},
    {0x0000000000001300, 0x0000000000001750, 0x000000000000114e, 0x000000000000131f,
     0x000000000000167b, 0x0000000000001371, 0x0000000000001230, 0x000000000000182c,
     0x0000000000001368, 0x0000000000000f31, 0x00000000000015c9},
    {0x0000000000000014, 0x0000000000000022, 0x0000000000000012, 0x0000000000000027,
     0x000000000000000d, 0x000000000000000d, 0x000000000000001c, 0x0000000000000002,
     0x0000000000000010, 0x0000000000000029, 0x000000000000000f}};

__device__ u64 GPU_FAST_PARTIAL_ROUND_W_HATS[N_PARTIAL_ROUNDS][11] = {
    {0x3d999c961b7c63b0, 0x814e82efcd172529, 0x2421e5d236704588, 0x887af7d4dd482328,
     0xa5e9c291f6119b27, 0xbdc52b2676a4b4aa, 0x64832009d29bcf57, 0x09c4155174a552cc,
     0x463f9ee03d290810, 0xc810936e64982542, 0x043b1c289f7bc3ac},
    {0x673655aae8be5a8b, 0xd510fe714f39fa10, 0x2c68a099b51c9e73, 0xa667bfa9aa96999d,
     0x4d67e72f063e2108, 0xf84dde3e6acda179, 0x40f9cc8c08f80981, 0x5ead032050097142,
     0x6591b02092d671bb, 0x00e18c71963dd1b7, 0x8a21bcd24a14218a},
    {0x202800f4addbdc87, 0xe4b5bdb1cc3504ff, 0xbe32b32a825596e7, 0x8e0f68c5dc223b9a,
     0x58022d9e1c256ce3, 0x584d29227aa073ac, 0x8b9352ad04bef9e7, 0xaead42a3f445ecbf,
     0x3c667a1d833a3cca, 0xda6f61838efa1ffe, 0xe8f749470bd7c446},
    {0xc5b85bab9e5b3869, 0x45245258aec51cf7, 0x16e6b8e68b931830, 0xe2ae0f051418112c,
     0x0470e26a0093a65b, 0x6bef71973a8146ed, 0x119265be51812daf, 0xb0be7356254bea2e,
     0x8584defff7589bd7, 0x3c5fe4aeb1fb52ba, 0x9e7cd88acf543a5e},
    {0x179be4bba87f0a8c, 0xacf63d95d8887355, 0x6696670196b0074f, 0xd99ddf1fe75085f9,
     0xc2597881fef0283b, 0xcf48395ee6c54f14, 0x15226a8e4cd8d3b6, 0xc053297389af5d3b,
     0x2c08893f0d1580e2, 0x0ed3cbcff6fcc5ba, 0xc82f510ecf81f6d0},
    {0x94b06183acb715cc, 0x500392ed0d431137, 0x861cc95ad5c86323, 0x05830a443f86c4ac,
     0x3b68225874a20a7c, 0x10b3309838e236fb, 0x9b77fc8bcd559e2c, 0xbdecf5e0cb9cb213,
     0x30276f1221ace5fa, 0x7935dd342764a144, 0xeac6db520bb03708},
    {0x7186a80551025f8f, 0x622247557e9b5371, 0xc4cbe326d1ad9742, 0x55f1523ac6a23ea2,
     0xa13dfe77a3d52f53, 0xe30750b6301c0452, 0x08bd488070a3a32b, 0xcd800caef5b72ae3,
     0x83329c90f04233ce, 0xb5b99e6664a0a3ee, 0x6b0731849e200a7f},
    {0xec3fabc192b01799, 0x382b38cee8ee5375, 0x3bfb6c3f0e616572, 0x514abd0cf6c7bc86,
     0x47521b1361dcc546, 0x178093843f863d14, 0xad1003c5d28918e7, 0x738450e42495bc81,
     0xaf947c59af5e4047, 0x4653fb0685084ef2, 0x057fde2062ae35bf},
    {0xe376678d843ce55e, 0x66f3860d7514e7fc, 0x7817f3dfff8b4ffa, 0x3929624a9def725b,
     0x0126ca37f215a80a, 0xfce2f5d02762a303, 0x1bc927375febbad7, 0x85b481e5243f60bf,
     0x2d3c5f42a39c91a0, 0x0811719919351ae8, 0xf669de0add993131},
    {0x7de38bae084da92d, 0x5b848442237e8a9b, 0xf6c705da84d57310, 0x31e6a4bdb6a49017,
     0x889489706e5c5c0f, 0x0e4a205459692a1b, 0xbac3fa75ee26f299, 0x5f5894f4057d755e,
     0xb0dc3ecd724bb076, 0x5e34d8554a6452ba, 0x04f78fd8c1fdcc5f},
    {0x4dd19c38779512ea, 0xdb79ba02704620e9, 0x92a29a3675a5d2be, 0xd5177029fe495166,
     0xd32b3298a13330c1, 0x251c4a3eb2c5f8fd, 0xe1c48b26e0d98825, 0x3301d3362a4ffccb,
     0x09bb6c88de8cd178, 0xdc05b676564f538a, 0x60192d883e473fee},
    {0x16b9774801ac44a0, 0x3cb8411e786d3c8e, 0xa86e9cf505072491, 0x0178928152e109ae,
     0x5317b905a6e1ab7b, 0xda20b3be7f53d59f, 0xcb97dedecebee9ad, 0x4bd545218c59f58d,
     0x77dc8d856c05a44a, 0x87948589e4f243fd, 0x7e5217af969952c2},
    {0xbc58987d06a84e4d, 0x0b5d420244c9cae3, 0xa3c4711b938c02c0, 0x3aace640a3e03990,
     0x865a0f3249aacd8a, 0x8d00b2a7dbed06c7, 0x6eacb905beb7e2f8, 0x045322b216ec3ec7,
     0xeb9de00d594828e6, 0x088c5f20df9e5c26, 0xf555f4112b19781f},
    {0xa8cedbff1813d3a7, 0x50dcaee0fd27d164, 0xf1cb02417e23bd82, 0xfaf322786e2abe8b,
     0x937a4315beb5d9b6, 0x1b18992921a11d85, 0x7d66c4368b3c497b, 0x0e7946317a6b4e99,
     0xbe4430134182978b, 0x3771e82493ab262d, 0xa671690d8095ce82},
    {0xb035585f6e929d9d, 0xba1579c7e219b954, 0xcb201cf846db4ba3, 0x287bf9177372cf45,
     0xa350e4f61147d0a6, 0xd5d0ecfb50bcff99, 0x2e166aa6c776ed21, 0xe1e66c991990e282,
     0x662b329b01e7bb38, 0x8aa674b36144d9a9, 0xcbabf78f97f95e65},
    {0xeec24b15a06b53fe, 0xc8a7aa07c5633533, 0xefe9c6fa4311ad51, 0xb9173f13977109a1,
     0x69ce43c9cc94aedc, 0xecf623c9cd118815, 0x28625def198c33c7, 0xccfc5f7de5c3636a,
     0xf5e6c40f1621c299, 0xcec0e58c34cb64b1, 0xa868ea113387939f},
    {0xd8dddbdc5ce4ef45, 0xacfc51de8131458c, 0x146bb3c0fe499ac0, 0x9e65309f15943903,
     0x80d0ad980773aa70, 0xf97817d4ddbf0607, 0xe4626620a75ba276, 0x0dfdc7fd6fc74f66,
     0xf464864ad6f2bb93, 0x02d55e52a5d44414, 0xdd8de62487c40925},
    {0xc15acf44759545a3, 0xcbfdcf39869719d4, 0x33f62042e2f80225, 0x2599c5ead81d8fa3,
     0x0b306cb6c1d7c8d0, 0x658c80d3df3729b1, 0xe8d1b2b21b41429c, 0xa1b67f09d4b3ccb8,
     0x0e1adf8b84437180, 0x0d593a5e584af47b, 0xa023d94c56e151c7},
    {0x49026cc3a4afc5a6, 0xe06dff00ab25b91b, 0x0ab38c561e8850ff, 0x92c3c8275e105eeb,
     0xb65256e546889bd0, 0x3c0468236ea142f6, 0xee61766b889e18f2, 0xa206f41b12c30415,
     0x02fe9d756c9f12d1, 0xe9633210630cbf12, 0x1ffea9fe85a0b0b1},
    {0x81d1ae8cc50240f3, 0xf4c77a079a4607d7, 0xed446b2315e3efc1, 0x0b0a6b70915178c3,
     0xb11ff3e089f15d9a, 0x1d4dba0b7ae9cc18, 0x65d74e2f43b48d05, 0xa2df8c6b8ae0804a,
     0xa4e6f0a8c33348a6, 0xc0a26efc7be5669b, 0xa6b6582c547d0d60},
    {0x84afc741f1c13213, 0x2f8f43734fc906f3, 0xde682d72da0a02d9, 0x0bb005236adb9ef2,
     0x5bdf35c10a8b5624, 0x0739a8a343950010, 0x52f515f44785cfbc, 0xcbaf4e5d82856c60,
     0xac9ea09074e3e150, 0x8f0fa011a2035fb0, 0x1a37905d8450904a},
    {0x3abeb80def61cc85, 0x9d19c9dd4eac4133, 0x075a652d9641a985, 0x9daf69ae1b67e667,
     0x364f71da77920a18, 0x50bd769f745c95b1, 0xf223d1180dbbf3fc, 0x2f885e584e04aa99,
     0xb69a0fa70aea684a, 0x09584acaa6e062a0, 0x0bc051640145b19b}};

__device__ u64 GPU_FAST_PARTIAL_ROUND_INITIAL_MATRIX[11][11] = {
    {0x80772dc2645b280b, 0xdc927721da922cf8, 0xc1978156516879ad, 0x90e80c591f48b603,
     0x3a2432625475e3ae, 0x00a2d4321cca94fe, 0x77736f524010c932, 0x904d3f2804a36c54,
     0xbf9b39e28a16f354, 0x3a1ded54a6cd058b, 0x42392870da5737cf},
    {0xe796d293a47a64cb, 0xb124c33152a2421a, 0x0ee5dc0ce131268a, 0xa9032a52f930fae6,
     0x7e33ca8c814280de, 0xad11180f69a8c29e, 0xc75ac6d5b5a10ff3, 0xf0674a8dc5a387ec,
     0xb36d43120eaa5e2b, 0x6f232aab4b533a25, 0x3a1ded54a6cd058b},
    {0xdcedab70f40718ba, 0x14a4a64da0b2668f, 0x4715b8e5ab34653b, 0x1e8916a99c93a88e,
     0xbba4b5d86b9a3b2c, 0xe76649f9bd5d5c2e, 0xaf8e2518a1ece54d, 0xdcda1344cdca873f,
     0xcd080204256088e5, 0xb36d43120eaa5e2b, 0xbf9b39e28a16f354},
    {0xf4a437f2888ae909, 0xc537d44dc2875403, 0x7f68007619fd8ba9, 0xa4911db6a32612da,
     0x2f7e9aade3fdaec1, 0xe7ffd578da4ea43d, 0x43a608e7afa6b5c2, 0xca46546aa99e1575,
     0xdcda1344cdca873f, 0xf0674a8dc5a387ec, 0x904d3f2804a36c54},
    {0xf97abba0dffb6c50, 0x5e40f0c9bb82aab5, 0x5996a80497e24a6b, 0x07084430a7307c9a,
     0xad2f570a5b8545aa, 0xab7f81fef4274770, 0xcb81f535cf98c9e9, 0x43a608e7afa6b5c2,
     0xaf8e2518a1ece54d, 0xc75ac6d5b5a10ff3, 0x77736f524010c932},
    {0x7f8e41e0b0a6cdff, 0x4b1ba8d40afca97d, 0x623708f28fca70e8, 0xbf150dc4914d380f,
     0xc26a083554767106, 0x753b8b1126665c22, 0xab7f81fef4274770, 0xe7ffd578da4ea43d,
     0xe76649f9bd5d5c2e, 0xad11180f69a8c29e, 0x00a2d4321cca94fe},
    {0x726af914971c1374, 0x1d7f8a2cce1a9d00, 0x18737784700c75cd, 0x7fb45d605dd82838,
     0x862361aeab0f9b6e, 0xc26a083554767106, 0xad2f570a5b8545aa, 0x2f7e9aade3fdaec1,
     0xbba4b5d86b9a3b2c, 0x7e33ca8c814280de, 0x3a2432625475e3ae},
    {0x64dd936da878404d, 0x4db9a2ead2bd7262, 0xbe2e19f6d07f1a83, 0x02290fe23c20351a,
     0x7fb45d605dd82838, 0xbf150dc4914d380f, 0x07084430a7307c9a, 0xa4911db6a32612da,
     0x1e8916a99c93a88e, 0xa9032a52f930fae6, 0x90e80c591f48b603},
    {0x85418a9fef8a9890, 0xd8a2eb7ef5e707ad, 0xbfe85ababed2d882, 0xbe2e19f6d07f1a83,
     0x18737784700c75cd, 0x623708f28fca70e8, 0x5996a80497e24a6b, 0x7f68007619fd8ba9,
     0x4715b8e5ab34653b, 0x0ee5dc0ce131268a, 0xc1978156516879ad},
    {0x156048ee7a738154, 0x91f7562377e81df5, 0xd8a2eb7ef5e707ad, 0x4db9a2ead2bd7262,
     0x1d7f8a2cce1a9d00, 0x4b1ba8d40afca97d, 0x5e40f0c9bb82aab5, 0xc537d44dc2875403,
     0x14a4a64da0b2668f, 0xb124c33152a2421a, 0xdc927721da922cf8},
    {0xd841e8ef9dde8ba0, 0x156048ee7a738154, 0x85418a9fef8a9890, 0x64dd936da878404d,
     0x726af914971c1374, 0x7f8e41e0b0a6cdff, 0xf97abba0dffb6c50, 0xf4a437f2888ae909,
     0xdcedab70f40718ba, 0xe796d293a47a64cb, 0x80772dc2645b280b},
};

/*
 * This is based on x = x_hi_hi * 2^96 + x_hi_lo * 2^64 + x_lo (x_lo is 64 bits)
 * Note that EPSILON is 0xFFFFFFFF, that's why we can do &
 */
__device__ INLINE gl64_t PoseidonPermutationGPU::reduce128(u128 x)
{
    u64 x_lo = u128::u128tou64(x);
    u64 x_hi = u128::u128t_hi_ou64(x);

    u64 x_hi_hi = x_hi >> 32;
    u64 x_hi_lo = x_hi & 0xFFFFFFFF; // epsilon

    gl64_t t0 = gl64_t(x_lo) - gl64_t(x_hi_hi);
    u64 t1 = x_hi_lo * 0xFFFFFFFF;
    gl64_t t2 = gl64_t(t0) + gl64_t(t1);

    return t2;
}

__device__ INLINE void PoseidonPermutationGPU::add_u160_u128(u128 *x_lo, u32 *x_hi, u128 y)
{
    u128 M;
    M.hi = (u64)-1;
    M.lo = (u64)-1;
    if (*x_lo > M - y)
    {
        *x_lo = *x_lo + y;
        *x_hi = *x_hi + 1;
    }
    else
    {
        *x_lo = *x_lo + y;
    }
}

__device__ INLINE gl64_t PoseidonPermutationGPU::from_noncanonical_u96(gl64_t n_lo, gl64_t n_hi)
{
    u64 t1 = n_hi.get_val() * 4294967295; // EPSILON
    return n_lo + gl64_t(t1);
}

__device__ INLINE gl64_t PoseidonPermutationGPU::from_noncanonical_u128(u128 n)
{
    gl64_t t1 = (gl64_t)(n.hi) * (gl64_t)0xFFFFFFFF;
    return (gl64_t)(n.lo) + t1;
}

__device__ INLINE gl64_t PoseidonPermutationGPU::reduce_u160(u128 n_lo, u32 n_hi)
{
    u128 reduced_hi = (u128)from_noncanonical_u96(n_lo.hi, n_hi).get_val();
    u128 reduced128 = (reduced_hi << 64) + (u128)n_lo.lo;
    return reduce128(reduced128);
}

__device__ INLINE gl64_t PoseidonPermutationGPU::mds_row_shf(u32 r, gl64_t *v)
{
    // assert(r < SPONGE_WIDTH);
    // The values of `MDS_MATRIX_CIRC` and `MDS_MATRIX_DIAG` are
    // known to be small, so we can accumulate all the products for
    // each row and reduce just once at the end (done by the
    // caller).

    u128 res = 0;
    for (u32 i = 0; i < 12; i++)
    {
        u64 tmp1 = GPU_MDS_MATRIX_CIRC[i];
        u128 tmp2 = v[(i + r) % SPONGE_WIDTH];
        res = res + tmp2 * tmp1;
    }
    u64 tmp1 = GPU_MDS_MATRIX_DIAG[r];
    u128 tmp2 = v[r];
    res = res + tmp2 * tmp1;

    return from_noncanonical_u96(u128::u128tou64(res), u128::u128t_hi_ou64(res) & 0xFFFFFFFF);
}

__device__ INLINE void PoseidonPermutationGPU::mds_layer(gl64_t *state, gl64_t *result)
{
    for (u32 r = 0; r < SPONGE_WIDTH; r++)
    {
        // here the result is already reduced
        result[r] = mds_row_shf(r, state);
    }
}

__device__ INLINE void PoseidonPermutationGPU::constant_layer(gl64_t *state, u32 *round_ctr)
{
    for (int i = 0; i < 12; i++)
    {
        gl64_t tmp = GPU_ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * (*round_ctr)];
        state[i] = state[i] + tmp;
    }
}

__device__ INLINE gl64_t PoseidonPermutationGPU::sbox_monomial(const gl64_t &x)
{
    gl64_t x2 = x * x;
    gl64_t x4 = x2 * x2;
    gl64_t x3 = x * x2;
    gl64_t xr = x3 * x4;
    return xr;
}

__device__ INLINE void PoseidonPermutationGPU::sbox_layer(gl64_t *state)
{
    for (int i = 0; i < 12; i++)
    {
        state[i] = sbox_monomial(state[i]);
    }
}

__device__ INLINE void PoseidonPermutationGPU::full_rounds(gl64_t *state, u32 *round_ctr)
{
    gl64_t res[12] = {(u64)0u};
    gl64_t *pres = res;

    for (int k = 0; k < HALF_N_FULL_ROUNDS; k++)
    {
        constant_layer(state, round_ctr);
        sbox_layer(state);
        mds_layer(state, pres);
        *round_ctr += 1;
        gl64_t *aux = state;
        state = pres;
        pres = aux;
    }
}

__device__ INLINE void PoseidonPermutationGPU::partial_rounds_naive(gl64_t *state, u32 *round_ctr)
{
    gl64_t res[12] = {(u64)0u};
    gl64_t *pres = res;
    for (int k = 0; k < N_PARTIAL_ROUNDS; k++)
    {
        constant_layer(state, round_ctr);
        state[0] = sbox_monomial(state[0]);
        mds_layer(state, pres);
        *round_ctr += 1;
        gl64_t *aux = state;
        state = pres;
        pres = aux;
    }
}

__device__ INLINE void PoseidonPermutationGPU::mds_partial_layer_fast(gl64_t *state, u32 r)
{
    // Set d = [M_00 | w^] dot [state]
    u128 d_sum_lo = 0;
    u32 d_sum_hi = 0;

#pragma unroll
    for (int i = 1; i < 12; i++)
    {
        u128 t = (u128)GPU_FAST_PARTIAL_ROUND_W_HATS[r][i - 1];
        u64 si = state[i].get_val();
        add_u160_u128(&d_sum_lo, &d_sum_hi, t * si);
    }

    u64 s0 = state[0].get_val();
    u128 mds0to0 = (u128)(GPU_MDS_MATRIX_CIRC[0] + GPU_MDS_MATRIX_DIAG[0]);
    add_u160_u128(&d_sum_lo, &d_sum_hi, mds0to0 * s0);

#pragma unroll
    for (int i = 1; i < 12; i++)
    {
        gl64_t t(GPU_FAST_PARTIAL_ROUND_VS[r][i - 1]);
        state[i] = state[i] + state[0] * t;
    }

    state[0] = reduce_u160(d_sum_lo, d_sum_hi);
}

__device__ INLINE void PoseidonPermutationGPU::partial_rounds(gl64_t *state, u32 *round_ctr)
{
    gl64_t res[12];
    // partial_first_constant_layer
#pragma unroll
    for (int i = 0; i < 12; i++)
    {
        state[i] = state[i] + gl64_t(GPU_FAST_PARTIAL_FIRST_ROUND_CONSTANT[i]);
        res[i].make_zero();
    }

    // mds_partial_layer_init
    res[0] = state[0];
#pragma unroll
    for (int r = 1; r < 12; r++)
    {
        for (int c = 1; c < 12; c++)
        {
            gl64_t v = gl64_t(GPU_FAST_PARTIAL_ROUND_INITIAL_MATRIX[r - 1][c - 1]);
            gl64_t t = v * state[r];
            res[c] = res[c] + t;
        }
    }
    for (u32 i = 0; i < 12; i++)
    {
        state[i] = res[i];
    }


    // loop
#pragma unroll
    for (int k = 0; k < N_PARTIAL_ROUNDS; k++)
    {
        state[0] = sbox_monomial(state[0]);
        state[0] = state[0] + gl64_t(GPU_FAST_PARTIAL_ROUND_CONSTANTS[k]);
        mds_partial_layer_fast(state, k);
    }
    *round_ctr += N_PARTIAL_ROUNDS;
}

__device__ INLINE gl64_t *PoseidonPermutationGPU::poseidon_naive(gl64_t *input)
{
    gl64_t *state = input;
    u32 round_ctr = 0;

    full_rounds(state, &round_ctr);
    partial_rounds_naive(state, &round_ctr);
    full_rounds(state, &round_ctr);
    return state;
}

__device__ INLINE gl64_t *PoseidonPermutationGPU::poseidon(gl64_t *input)
{
    gl64_t *state = input;
    u32 round_ctr = 0;

    full_rounds(state, &round_ctr);
    partial_rounds(state, &round_ctr);
    full_rounds(state, &round_ctr);
    return state;
}

__device__ PoseidonPermutationGPU::PoseidonPermutationGPU()
{
    for (int i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i].make_zero();
    }
}

__device__ void PoseidonPermutationGPU::set_from_slice(gl64_t *elts, u32 len, u32 start_idx)
{
    assert(start_idx + len <= SPONGE_WIDTH);
    for (int i = 0; i < len; i++)
    {
        this->state[start_idx + i] = elts[i];
    }
}

__device__ void PoseidonPermutationGPU::set_from_slice_stride(gl64_t *elts, u32 len, u32 start_idx, u32 stride)
{
    assert(start_idx + len <= SPONGE_WIDTH);
    for (int i = 0; i < len; i++)
    {
        this->state[start_idx + i] = elts[i * stride];
    }
}

__device__ void PoseidonPermutationGPU::get_state_as_canonical_u64(u64 *out)
{
    assert(out != 0);
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        out[i] = state[i].get_val();
    }
}

__device__ void PoseidonPermutationGPU::set_state(u32 idx, gl64_t val)
{
    assert(idx < SPONGE_WIDTH);
    state[idx] = val;
}

__device__ void PoseidonPermutationGPU::permute()
{
    poseidon_naive(state);
    // poseidon(state);
}

__device__ gl64_t *PoseidonPermutationGPU::squeeze(u32 size)
{
    // assert(size <= SPONGE_WIDTH);
    return state;
}

__device__ INLINE void gpu_poseidon_hash_one_stride(gl64_t *inputs, u32 num_inputs, gl64_t *hash, u32 stride)
{
    if (num_inputs <= NUM_HASH_OUT_ELTS)
    {
        u32 i = 0;
        for (; i < num_inputs; i++)
        {
            hash[i] = inputs[i * stride];
        }
        for (; i < NUM_HASH_OUT_ELTS; i++)
        {
            hash[i].make_zero();
        }
    }
    else
    {
        PoseidonPermutationGPU perm = PoseidonPermutationGPU();

        // Absorb all input chunks.
        for (u32 idx = 0; idx < num_inputs; idx += SPONGE_RATE)
        {
            perm.set_from_slice_stride(inputs + idx, MIN(SPONGE_RATE, num_inputs - idx), 0, stride);
            perm.permute();
        }
        gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
        for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
        {
            hash[i] = ret[i];
        }
    }
}

__device__ void PoseidonHasher::gpu_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
{
    PoseidonPermutationGPU::gpu_hash_one_with_permutation_template<PoseidonPermutationGPU>(inputs, num_inputs, hash);
}

__device__ void PoseidonHasher::gpu_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
{
    PoseidonPermutationGPU::gpu_hash_two_with_permutation_template<PoseidonPermutationGPU>(hash1, hash2, hash);
}

#ifdef DEBUG
DEVICE  void print_perm(gl64_t *data, int cnt)
    {
        for (int i = 0; i < cnt; i++)
        {
            printf("%lu ", data[i].get_val());
        }
        printf("\n");
    }
#endif
