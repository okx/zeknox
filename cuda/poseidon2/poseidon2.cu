#include "poseidon2.hpp"
#include "poseidon2_constants.hpp"
#include "assert.h"

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_MATRIX_DIAG_12_GOLDILOCKS[12] = {
#else
const u64 MATRIX_DIAG_12_GOLDILOCKS[12] = {
#endif
    0xc3b6c08e23ba9300,
    0xd84b5de94a324fb6,
    0x0d0c371c5b35b84f,
    0x7964f570e7188037,
    0x5daf18bbd996604b,
    0x6743bc47b9595257,
    0x5528b9362c59bb70,
    0xac45e25b7127b68b,
    0xa2077d7dfbb606b5,
    0xf3faac6faee378ae,
    0x0c6388b51545e883,
    0xd27dbb6944917b60};

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_RC12[360] = {
#else
const u64 RC12[360] = {
#endif
    1431286215153372998, 3509349009260703107, 2289575380984896342, 10625215922958251110, 17137022507167291684, 17143426961497010024, 9589775313463224365, 7736066733515538648, 2217569167061322248, 10394930802584583083, 4612393375016695705, 5332470884919453534,
    8724526834049581439, 17673787971454860688, 2519987773101056005, 7999687124137420323, 18312454652563306701, 15136091233824155669, 1257110570403430003, 5665449074466664773, 16178737609685266571, 52855143527893348, 8084454992943870230, 2597062441266647183,
    3342624911463171251, 6781356195391537436, 4697929572322733707, 4179687232228901671, 17841073646522133059, 18340176721233187897, 13152929999122219197, 6306257051437840427, 4974451914008050921, 11258703678970285201, 581736081259960204, 18323286026903235604,
    10250026231324330997, 13321947507807660157, 13020725208899496943, 11416990495425192684, 7221795794796219413, 2607917872900632985, 2591896057192169329, 10485489452304998145, 9480186048908910015, 2645141845409940474, 16242299839765162610, 12203738590896308135,
    5395176197344543510, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    17941136338888340715, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    7559392505546762987, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    549633128904721280, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    15658455328409267684, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    10078371877170729592, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2349868247408080783, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    13105911261634181239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    12868653202234053626, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    9471330315555975806, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4580289636625406680, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    13222733136951421572, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4555032575628627551, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    7619130111929922899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4547848507246491777, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    5662043532568004632, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    15723873049665279492, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    13585630674756818185, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    6990417929677264473, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    6373257983538884779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1005856792729125863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    17850970025369572891, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    14306783492963476045, 12653264875831356889, 10887434669785806501, 7221072982690633460, 9953585853856674407, 13497620366078753434, 18140292631504202243, 17311934738088402529, 6686302214424395771, 11193071888943695519, 10233795775801758543, 3362219552562939863,
    8595401306696186761, 7753411262943026561, 12415218859476220947, 12517451587026875834, 3257008032900598499, 2187469039578904770, 657675168296710415, 8659969869470208989, 12526098871288378639, 12525853395769009329, 15388161689979551704, 7880966905416338909,
    2911694411222711481, 6420652251792580406, 323544930728360053, 11718666476052241225, 2449132068789045592, 17993014181992530560, 15161788952257357966, 3788504801066818367, 1282111773460545571, 8849495164481705550, 8380852402060721190, 2161980224591127360,
    2440151485689245146, 17521895002090134367, 13821005335130766955, 17513705631114265826, 17068447856797239529, 17964439003977043993, 5685000919538239429, 11615940660682589106, 2522854885180605258, 12584118968072796115, 17841258728624635591, 10821564568873127316};

#define ROUNDS_F 8
#define ROUNDS_P 22

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS[8][8] =
#else
constexpr u64 HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS[8][8] =
#endif
    {
        {
            0xdd5743e7f2a5a5d9,
            0xcb3a864e58ada44b,
            0xffa2449ed32f8cdc,
            0x42025f65d6bd13ee,
            0x7889175e25506323,
            0x34b98bb03d24b737,
            0xbdcc535ecc4faa2a,
            0x5b20ad869fc0d033,
        },
        {
            0xf1dda5b9259dfcb4,
            0x27515210be112d59,
            0x4227d1718c766c3f,
            0x26d333161a5bd794,
            0x49b938957bf4b026,
            0x4a56b5938b213669,
            0x1120426b48c8353d,
            0x6b323c3f10a56cad,
        },
        {
            0xce57d6245ddca6b2,
            0xb1fc8d402bba1eb1,
            0xb5c5096ca959bd04,
            0x6db55cd306d31f7f,
            0xc49d293a81cb9641,
            0x1ce55a4fe979719f,
            0xa92e60a9d178a4d1,
            0x002cc64973bcfd8c,
        },
        {
            0xcea721cce82fb11b,
            0xe5b55eb8098ece81,
            0x4e30525c6f1ddd66,
            0x43c6702827070987,
            0xaca68430a7b5762a,
            0x3674238634df9c93,
            0x88cee1c825e33433,
            0xde99ae8d74b57176,
        },
        {
            0x014ef1197d341346,
            0x9725e20825d07394,
            0xfdb25aef2c5bae3b,
            0xbe5402dc598c971e,
            0x93a5711f04cdca3d,
            0xc45a9a5b2f8fb97b,
            0xfe8946a924933545,
            0x2af997a27369091c,
        },
        {
            0xaa62c88e0b294011,
            0x058eb9d810ce9f74,
            0xb3cb23eced349ae4,
            0xa3648177a77b4a84,
            0x43153d905992d95d,
            0xf4e2a97cda44aa4b,
            0x5baa2702b908682f,
            0x082923bdf4f750d1,
        },
        {
            0x98ae09a325893803,
            0xf8a6475077968838,
            0xceb0735bf00b2c5f,
            0x0a1a5d953888e072,
            0x2fcb190489f94475,
            0xb5be06270dec69fc,
            0x739cb934b09acf8b,
            0x537750b75ec7f25b,
        },
        {
            0xe9dd318bae1f3961,
            0xf7462137299efe1a,
            0xb1f6b8eee9adb940,
            0xbdebcc8a809dfe6b,
            0x40fc1f791b178113,
            0x3ac1c3362d014864,
            0x9a016184bdb8aeba,
            0x95f2394459fbc25e,
        }};

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS[22] =
#else
constexpr u64 HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS[22] =
#endif
    {
        0x488897d85ff51f56,
        0x1140737ccb162218,
        0xa7eeb9215866ed35,
        0x9bd2976fee49fcc9,
        0xc0c8f0de580a3fcc,
        0x4fb2dae6ee8fc793,
        0x343a89f35f37395b,
        0x223b525a77ca72c8,
        0x56ccb62574aaa918,
        0xc4d507d8027af9ed,
        0xa080673cf0b7e95c,
        0xf0184884eb70dcf8,
        0x044f10b0cb3d5c69,
        0xe9e3f7993938f186,
        0x1b761c80e772f459,
        0x606cec607a1b5fac,
        0x14a0c2e1d45f03cd,
        0x4eace8855398574f,
        0xf905ca7103eff3e6,
        0xf8c8f8d20862c059,
        0xb524fe8bdd678e5a,
        0xfbb7865901a1ec41,
};

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_MATRIX_DIAG_8_GOLDILOCKS_U64[8] =
#else
constexpr u64 MATRIX_DIAG_8_GOLDILOCKS_U64[8] =
#endif
    {
        0xa98811a1fed4e3a5,
        0x1cc48b54f377e2a0,
        0xe40cd4f6c5609a26,
        0x11de79ebca97a4a3,
        0x9177c73d8b7e929c,
        0x2a6fe8085797e791,
        0x3de6e93329f8d5ad,
        0x3f7af9125da962fe};

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_MATRIX_DIAG_12_GOLDILOCKS_U64[12] =
#else
constexpr u64 MATRIX_DIAG_12_GOLDILOCKS_U64[12] =
#endif
    {
        0xc3b6c08e23ba9300,
        0xd84b5de94a324fb6,
        0x0d0c371c5b35b84f,
        0x7964f570e7188037,
        0x5daf18bbd996604b,
        0x6743bc47b9595257,
        0x5528b9362c59bb70,
        0xac45e25b7127b68b,
        0xa2077d7dfbb606b5,
        0xf3faac6faee378ae,
        0x0c6388b51545e883,
        0xd27dbb6944917b60,
};

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_MATRIX_DIAG_16_GOLDILOCKS_U64[16] =
#else
constexpr u64 MATRIX_DIAG_16_GOLDILOCKS_U64[16] =
#endif
    {
        0xde9b91a467d6afc0,
        0xc5f16b9c76a9be17,
        0x0ab0fef2d540ac55,
        0x3001d27009d05773,
        0xed23b1f906d3d9eb,
        0x5ce73743cba97054,
        0x1c3bab944af4ba24,
        0x2faa105854dbafae,
        0x53ffb3ae6d421a10,
        0xbcda9df8884ba396,
        0xfc1273e4a31807bb,
        0xc77952573d5142c0,
        0x56683339a819b85e,
        0x328fcbd8f0ddc8eb,
        0xb5101e303fce9cb7,
        0x774487b8c40089bb,
};

#ifdef USE_CUDA
__device__ __constant__ u64 GPU_MATRIX_DIAG_20_GOLDILOCKS_U64[20] =
#else
constexpr u64 MATRIX_DIAG_20_GOLDILOCKS_U64[20] =
#endif
    {
        0x95c381fda3b1fa57,
        0xf36fe9eb1288f42c,
        0x89f5dcdfef277944,
        0x106f22eadeb3e2d2,
        0x684e31a2530e5111,
        0x27435c5d89fd148e,
        0x3ebed31c414dbf17,
        0xfd45b0b2d294e3cc,
        0x48c904473a7f6dbf,
        0xe0d1b67809295b4d,
        0xddd1941e9d199dcb,
        0x8cfe534eeb742219,
        0xa6e5261d9e3b8524,
        0x6897ee5ed0f82c1b,
        0x0e7dcd0739ee5f78,
        0x493253f3d0d32363,
        0xbb2737f5845f05c0,
        0xa187e810b06ad903,
        0xb635b995936c4918,
        0x0b3694a940bd2394,
};

#ifdef USE_CUDA
__device__ __constant__ u32 GPU_BABYBEAR_WIDTH_16_EXT_CONST_P3[8 * 16] =
#else
constexpr u32 BABYBEAR_WIDTH_16_EXT_CONST_P3[8 * 16] =
#endif
    {
        0x4EC2680C,
        0x110279CB,
        0x332D1F04,
        0x7DA39A8,
        0x20D60D25,
        0x6837F03,
        0x5C499950,
        0x11D53B04,
        0x4769DFB8,
        0x4D9FB1BF,
        0x21DD1576,
        0x58811E55,
        0x4D9AA329,
        0x2B412236,
        0x3822164A,
        0x1345CE59,
        0x3E8B66A4,
        0x3104F884,
        0x53CF147F,
        0x4C3C5D00,
        0xCCE8ACB,
        0x35EB8AB9,
        0x4EB90117,
        0xB0AB74,
        0x3DC5CED0,
        0x57E20602,
        0x5543D924,
        0x6A06354,
        0x4305BF7B,
        0x516D38B,
        0x288628B2,
        0xE801D9D,
        0x4FC7D3F6,
        0x5AC756E3,
        0x6DBC7A1D,
        0x22768C25,
        0x2D798FD,
        0x6AB7F8C8,
        0x610A3FA1,
        0x664B611A,
        0x61807B01,
        0x4277C420,
        0x429C112E,
        0x2873398E,
        0x74FD36A3,
        0x5180E3B0,
        0xB25B05,
        0x744737A6,
        0x1F828B47,
        0x73760EFB,
        0x571450E7,
        0x471CDEB5,
        0x48D335C3,
        0x46913BAD,
        0x19D6C553,
        0x72F92FBD,
        0x1E25480B,
        0x110700EF,
        0x3E6C0276,
        0x363288DE,
        0x4B0A4DD6,
        0x338B375D,
        0x124CD27E,
        0x1E178795,
        0x5777F011,
        0x68948CEB,
        0x19C2A7BC,
        0x5691B910,
        0x3491DC1C,
        0x20C91A20,
        0x56442FD6,
        0x37FE675F,
        0x6A4822D6,
        0x458DA37C,
        0x6688B5A,
        0x2299FBFF,
        0x776837E8,
        0x32EE44AA,
        0x37B964B7,
        0x3B4B31B4,
        0x6DBD1269,
        0x1A1F74C6,
        0x5C6B1DB1,
        0x3670308A,
        0x23D18114,
        0x22BFE022,
        0x4B432285,
        0x64E58ED,
        0x108AF480,
        0xA298030,
        0x15CE7F03,
        0x5831987C,
        0x56E222BA,
        0x61818A80,
        0x1DC0806D,
        0x2C2E3A27,
        0x26D36BF6,
        0x1BB4B661,
        0x558A76B,
        0x1575D881,
        0xDBA9003,
        0x69AFCEC8,
        0x52436DEB,
        0x2C6805C3,
        0x2F4B7A6E,
        0x512367CC,
        0x560DC002,
        0x3139B8A,
        0x2B987ECA,
        0x40D2C58A,
        0x4BEED74D,
        0x12925C15,
        0x29E264AA,
        0x57CFF845,
        0x7DFD045,
        0x505CD248,
        0x1C6B0406,
        0x55C4A053,
        0x252590,
        0x506BA70E,
        0x747EDFDA,
        0x690819DF,
        0x4BC54E23,
        0x5EF7512C,
        0x33870D43,
        0x84B39D1,
        0x3EC935D6,
        0x19340FFB,
};

#ifdef USE_CUDA
__device__ __constant__ u32 GPU_BABYBEAR_WIDTH_16_INT_CONST_P3[13] =
#else
constexpr u32 BABYBEAR_WIDTH_16_INT_CONST_P3[13] =
#endif
    {0x1801CCD8, 0x131D9E83, 0x42EC25EE, 0x5FC787D, 0xC1356DB, 0x491AAE7C, 0x40E3021A, 0x3D25F0A, 0x68BDAFC, 0x2B32678F, 0x631CE19F, 0x2F8CC233, 0x401CE61F};

#ifdef USE_CUDA
__device__ __constant__ u32 GPU_BABYBEAR_WIDTH_24_EXT_CONST_P3[8 * 24] =
#else
constexpr u32 BABYBEAR_WIDTH_24_EXT_CONST_P3[8 * 24] =
#endif
    {
        0x4EC2680C, 0x110279CB, 0x332D1F04, 0x7DA39A8, 0x20D60D25, 0x6837F03, 0x5C499950, 0x11D53B04, 0x4769DFB8, 0x4D9FB1BF, 0x21DD1576, 0x58811E55, 0x4D9AA329, 0x2B412236, 0x3822164A, 0x1345CE59, 0x3E8B66A4, 0x3104F884, 0x53CF147F, 0x4C3C5D00, 0xCCE8ACB, 0x35EB8AB9, 0x4EB90117, 0xB0AB74, 0x3DC5CED0, 0x57E20602, 0x5543D924, 0x6A06354, 0x4305BF7B, 0x516D38B, 0x288628B2, 0xE801D9D, 0x4FC7D3F6, 0x5AC756E3, 0x6DBC7A1D, 0x22768C25, 0x2D798FD, 0x6AB7F8C8, 0x610A3FA1, 0x664B611A, 0x61807B01, 0x4277C420, 0x429C112E, 0x2873398E, 0x74FD36A3, 0x5180E3B0, 0xB25B05, 0x744737A6, 0x1F828B47, 0x73760EFB, 0x571450E7, 0x471CDEB5, 0x48D335C3, 0x46913BAD, 0x19D6C553, 0x72F92FBD, 0x1E25480B, 0x110700EF, 0x3E6C0276, 0x363288DE, 0x4B0A4DD6, 0x338B375D, 0x124CD27E, 0x1E178795, 0x5777F011, 0x68948CEB, 0x19C2A7BC, 0x5691B910, 0x3491DC1C, 0x20C91A20, 0x56442FD6, 0x37FE675F, 0x6A4822D6, 0x458DA37C, 0x6688B5A, 0x2299FBFF, 0x776837E8, 0x32EE44AA, 0x37B964B7, 0x3B4B31B4, 0x6DBD1269, 0x1A1F74C6, 0x5C6B1DB1, 0x3670308A, 0x23D18114, 0x22BFE022, 0x4B432285, 0x64E58ED, 0x108AF480, 0xA298030, 0x15CE7F03, 0x5831987C, 0x56E222BA, 0x61818A80, 0x1DC0806D, 0x2C2E3A27, 0x26D36BF6, 0x1BB4B661, 0x558A76B, 0x1575D881, 0xDBA9003, 0x69AFCEC8, 0x52436DEB, 0x2C6805C3, 0x2F4B7A6E, 0x512367CC, 0x560DC002, 0x3139B8A, 0x2B987ECA, 0x40D2C58A, 0x4BEED74D, 0x12925C15, 0x29E264AA, 0x57CFF845, 0x7DFD045, 0x505CD248, 0x1C6B0406, 0x55C4A053, 0x252590, 0x506BA70E, 0x747EDFDA, 0x690819DF, 0x4BC54E23, 0x5EF7512C, 0x33870D43, 0x84B39D1, 0x3EC935D6, 0x19340FFB, 0x1801CCD8, 0x131D9E83, 0x42EC25EE, 0x5FC787D, 0xC1356DB, 0x491AAE7C, 0x40E3021A, 0x3D25F0A, 0x68BDAFC, 0x2B32678F, 0x631CE19F, 0x2F8CC233, 0x401CE61F, 0x5DE64A96, 0x15CD53A1, 0x24021AD3, 0x7466B2AB, 0x5DFBA9DE, 0x3F3B5642, 0x4DD9BBC1, 0x9CDE2AA, 0x73827615, 0x677A602B, 0x69BFF5DF, 0x6BCA4452, 0x538EACBB, 0x21695E2F, 0x48FD28A5, 0x53DB5C4C, 0x39AB7ABE, 0x60226CA8, 0x6F39CE27, 0x6DD72702, 0x61C72CA5, 0xB2ABE90, 0x3673352A, 0x36298C76, 0x50DE59D, 0x4169C3EE, 0x63258D2A, 0x59C45549, 0x3EB0408A, 0x72CE8221, 0x7372C616, 0x346F1D76, 0x42B0E84C, 0x271CB214, 0xF64F596, 0x2DEC45DF, 0x27FC1A0, 0x3C938ABF, 0x61BAD871, 0x6E5FD31D, 0x6F36A6D4, 0x544B3F0E, 0x18F27FA1, 0x34451992, 0x2417883F, 0x5157A5B6, 0x2EEB111E, 0x150135D7, 0x355925A3, 0x33329A06, 0x460CB30C};

#ifdef USE_CUDA
__device__ __constant__ u32 GPU_BABYBEAR_WIDTH_24_INT_CONST_P3[21] =
#else
constexpr u32 BABYBEAR_WIDTH_24_INT_CONST_P3[21] =
#endif
    {
        0x24D2AF3B,
        0x766ABFC3,
        0x4A4BBF41,
        0x18DEDD7,
        0x37A4705,
        0x27666A5D,
        0x3475E251,
        0xF0CB909,
        0x68ACF372,
        0x22CEC228,
        0x164774DC,
        0x59034D05,
        0x752865FD,
        0x64E74AD,
        0x5233240A,
        0x39C32A85,
        0x89480D7,
        0x3C439665,
        0x70908112,
        0x2C664E9E,
        0x647B04AC};

#ifdef USE_CUDA
__device__ __forceinline__ void apply_m_4(gl64_t *x)
#else
inline void apply_m_4(GoldilocksField *x)
#endif
{
    auto t0 = x[0] + x[1];
    auto t1 = x[2] + x[3];
    auto t2 = x[1] + x[1] + t1;
    auto t3 = x[3] + x[3] + t0;
    auto t4 = t1 + t1 + t1 + t1 + t3;
    auto t5 = t0 + t0 + t0 + t0 + t2;
    auto t6 = t3 + t5;
    auto t7 = t2 + t4;
    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

#ifdef USE_CUDA
__device__ __forceinline__ void add_rc(gl64_t *state, gl64_t *rc)
#else
inline void add_rc(GoldilocksField *state, GoldilocksField *rc)
#endif
{
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] + rc[i];
    }
}

#ifdef USE_CUDA
__device__ __forceinline__ gl64_t sbox_p(gl64_t &x)
#else
inline GoldilocksField sbox_p(GoldilocksField &x)
#endif
{
    auto x2 = x * x;
    auto x4 = x2 * x2;
    auto x3 = x2 * x;
    return x3 * x4;
}

#ifdef USE_CUDA
__device__ __forceinline__ void sbox(gl64_t *state)
#else
inline void sbox(GoldilocksField *state)
#endif
{
    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = sbox_p(state[i]);
    }
}

#ifdef USE_CUDA
__device__ __forceinline__ void ext_permute_mut(gl64_t *state)
#else
inline void ext_permute_mut(GoldilocksField *state)
#endif
{
    for (u32 i = 0; i < SPONGE_WIDTH; i += 4)
    {
        apply_m_4(state + i);
    }

#ifdef USE_CUDA
    gl64_t sums[4];
#else
    GoldilocksField sums[4];
#endif
    sums[0] = state[0] + state[4] + state[8];
    sums[1] = state[1] + state[5] + state[9];
    sums[2] = state[2] + state[6] + state[10];
    sums[3] = state[3] + state[7] + state[11];

    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] + sums[i % 4];
    }
}

#ifdef USE_CUDA
__device__ __forceinline__ void matmul_internal(gl64_t *state, gl64_t *mat_internal_diag_m_1)
#else
inline void matmul_internal(GoldilocksField *state, GoldilocksField *mat_internal_diag_m_1)
#endif
{
    auto sum = state[0];
    for (u32 i = 1; i < SPONGE_WIDTH; i++)
    {
        sum = sum + state[i];
    }

    for (u32 i = 0; i < SPONGE_WIDTH; i++)
    {
        state[i] = state[i] * mat_internal_diag_m_1[i];
        state[i] = state[i] + sum;
    }
}

#ifdef USE_CUDA
__device__ __forceinline__ void poseidon2(gl64_t *state)
#else
inline void poseidon2(GoldilocksField *state)
#endif
{
#ifdef USE_CUDA
    gl64_t *rc12 = (gl64_t *)GPU_RC12;
    gl64_t *md12 = (gl64_t *)GPU_MATRIX_DIAG_12_GOLDILOCKS;
#else
    GoldilocksField *rc12 = (GoldilocksField *)RC12;
    GoldilocksField *md12 = (GoldilocksField *)MATRIX_DIAG_12_GOLDILOCKS;
#endif

    // The initial linear layer.
    ext_permute_mut(state);

    // The first half of the external rounds.
    u32 rounds = ROUNDS_F + ROUNDS_P;
    u32 rounds_f_beginning = ROUNDS_F / 2;
    for (u32 r = 0; r < rounds_f_beginning; r++)
    {
        add_rc(state, &rc12[12 * r]);
        sbox(state);
        ext_permute_mut(state);
    }

    // The internal rounds.
    u32 p_end = rounds_f_beginning + ROUNDS_P;
    for (u32 r = rounds_f_beginning; r < p_end; r++)
    {
        state[0] = state[0] + rc12[12 * r];
        state[0] = sbox_p(state[0]);
        matmul_internal(state, md12);
    }

    // The second half of the external rounds.
    for (u32 r = p_end; r < rounds; r++)
    {
        add_rc(state, &rc12[12 * r]);
        sbox(state);
        ext_permute_mut(state);
    }
}

template <typename F, const u64 WIDTH>
DEVICE INLINE void DiffusionMatrixGoldilocks<F, WIDTH>::permute_mut(F *state)
{
    u64 *mat = NULL;
    switch (WIDTH)
    {
    case 8:
#ifdef USE_CUDA
        mat = (u64 *)GPU_MATRIX_DIAG_8_GOLDILOCKS_U64;
#else
        mat = (u64 *)MATRIX_DIAG_8_GOLDILOCKS_U64;
#endif
        break;
    case 12:
#ifdef USE_CUDA
        mat = (u64 *)GPU_MATRIX_DIAG_12_GOLDILOCKS_U64;
#else
        mat = (u64 *)MATRIX_DIAG_12_GOLDILOCKS_U64;
#endif
        break;
    case 16:
#ifdef USE_CUDA
        mat = (u64 *)GPU_MATRIX_DIAG_16_GOLDILOCKS_U64;
#else
        mat = (u64 *)MATRIX_DIAG_16_GOLDILOCKS_U64;
#endif
        break;
    case 20:
#ifdef USE_CUDA
        mat = (u64 *)GPU_MATRIX_DIAG_20_GOLDILOCKS_U64;
#else
        mat = (u64 *)MATRIX_DIAG_20_GOLDILOCKS_U64;
#endif
        break;
    default:
        printf("Unsupported width %lu\n", WIDTH);
        assert(0);
    }
    F matf[WIDTH];
    for (u64 i = 0; i < WIDTH; i++)
    {
        matf[i] = F(mat[i]);
    }
    matmul_internal_<F, WIDTH>(state, matf);
}

#ifdef USE_CUDA
__device__ void hl_poseidon2_goldilocks_width_8(gl64_t *input)
{
    const u64 _width = 8;
    const u64 _exponent = 7;
    const u64 _rounds_f = 8;
    const u64 _rounds_p = 22;

    gl64_t ext_const[_rounds_f * _width];
    for (u64 i = 0; i < _rounds_f; i++)
    {
        for (u64 j = 0; j < _width; j++)
        {
            ext_const[i * _width + j] = gl64_t(GPU_HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS[i][j]);
        }
    }
    gl64_t int_const[_rounds_p];
    for (u64 i = 0; i < _rounds_p; i++)
    {
        int_const[i] = gl64_t(GPU_HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS[i]);
    }

    // Our Poseidon2 implementation.
    auto ext_layer = Poseidon2ExternalMatrixHL<gl64_t, _width>();
    auto int_layer = DiffusionMatrixGoldilocks<gl64_t, _width>();

    auto poseidon2 = Poseidon2<
        gl64_t,
        Poseidon2ExternalMatrixHL<gl64_t, _width>,
        DiffusionMatrixGoldilocks<gl64_t, _width>,
        _width,
        _exponent>(
        _rounds_f,
        _rounds_p,
        ext_const,
        int_const,
        ext_layer,
        int_layer);

    poseidon2.permute_mut(input);
}
#else
void hl_poseidon2_goldilocks_width_8(GoldilocksField *input)
{
    const u64 _width = 8;
    const u64 _exponent = 7;
    const u64 _rounds_f = 8;
    const u64 _rounds_p = 22;

    GoldilocksField ext_const[_rounds_f * _width];
    for (u64 i = 0; i < _rounds_f; i++)
    {
        for (u64 j = 0; j < _width; j++)
        {
            ext_const[i * _width + j] = GoldilocksField::from_canonical_u64(HL_GOLDILOCKS_8_EXTERNAL_ROUND_CONSTANTS[i][j]);
        }
    }
    GoldilocksField int_const[_rounds_p];
    for (u64 i = 0; i < _rounds_p; i++)
    {
        int_const[i] = GoldilocksField::from_canonical_u64(HL_GOLDILOCKS_8_INTERNAL_ROUND_CONSTANTS[i]);
    }

    // Our Poseidon2 implementation.
    auto ext_layer = Poseidon2ExternalMatrixHL<GoldilocksField, _width>();
    auto int_layer = DiffusionMatrixGoldilocks<GoldilocksField, _width>();

    auto poseidon2 = Poseidon2<
        GoldilocksField,
        Poseidon2ExternalMatrixHL<GoldilocksField, _width>,
        DiffusionMatrixGoldilocks<GoldilocksField, _width>,
        _width,
        _exponent>(
        _rounds_f,
        _rounds_p,
        ext_const,
        int_const,
        ext_layer,
        int_layer);

    poseidon2.permute_mut(input);
}
#endif

#ifdef USE_CUDA
__forceinline__ __device__ void Poseidon2PermutationGPU::permute()
{
    poseidon2(get_state());
}

__device__ void Poseidon2Hasher::gpu_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
{
    PoseidonPermutationGPU::gpu_hash_one_with_permutation_template<Poseidon2PermutationGPU>(inputs, num_inputs, hash);
}

__device__ void Poseidon2Hasher::gpu_hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
{
    PoseidonPermutationGPU::gpu_hash_two_with_permutation_template<Poseidon2PermutationGPU>(hash1, hash2, hash);
}

#else // USE_CUDA

inline void Poseidon2Permutation::permute()
{
    poseidon2(get_state());
}

void Poseidon2Hasher::cpu_hash_one(u64 *input, u64 input_count, u64 *digest)
{
    PoseidonPermutation::cpu_hash_one_with_permutation_template<Poseidon2Permutation>(input, input_count, digest);
}

void Poseidon2Hasher::cpu_hash_two(u64 *digest_left, u64 *digest_right, u64 *digest)
{
    PoseidonPermutation::cpu_hash_two_with_permutation_template<Poseidon2Permutation>(digest_left, digest_right, digest);
}

#endif // USE_CUDA
