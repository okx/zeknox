#ifndef __POSEIDON_V2_CUH__
#define __POSEIDON_V2_CUH__

#include "types.h"
#include "gl64_t.cuh"
#include "cuda_utils.cuh"
#include "cuda_uint128.cuh"

#define MIN(x, y) (x < y) ? x : y

#define NUM_HASH_OUT_ELTS 4

// From poseidon.rs
#define SPONGE_RATE 8
#define SPONGE_CAPACITY 4
// #define SPONGE_WIDTH SPONGE_RATE + SPONGE_CAPACITY
#define SPONGE_WIDTH 12

// The number of full rounds and partial rounds is given by the
// calc_round_numbers.py script. They happen to be the same for both
// width 8 and width 12 with s-box x^7.
//
// NB: Changing any of these values will require regenerating all of
// the precomputed constant arrays in this file.
#define HALF_N_FULL_ROUNDS 4
// #define N_FULL_ROUNDS_TOTAL 2 * HALF_N_FULL_ROUNDS
#define N_FULL_ROUNDS_TOTAL 8
#define N_PARTIAL_ROUNDS 22
// #define N_ROUNDS N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS
#define N_ROUNDS 30
#define MAX_WIDTH 12
// we only have width 8 and 12, and 12 is bigger. :)

// The MDS matrix we use is C + D, where C is the circulant matrix whose first row is given by
// `MDS_MATRIX_CIRC`, and D is the diagonal matrix whose diagonal is given by `MDS_MATRIX_DIAG`.
//
// WARNING: If the MDS matrix is changed, then the following
// constants need to be updated accordingly:
//  - FAST_PARTIAL_ROUND_CONSTANTS
//  - FAST_PARTIAL_ROUND_VS
//  - FAST_PARTIAL_ROUND_W_HATS
//  - FAST_PARTIAL_ROUND_INITIAL_MATRIX
__device__ u32 GPU_MDS_MATRIX_CIRC[12] = {17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20};
__device__ u32 GPU_MDS_MATRIX_DIAG[12] = {8, (uint64_t)0u, (uint64_t)0u, (uint64_t)0u, (uint64_t)0u, (uint64_t)0u, (uint64_t)0u, (uint64_t)0u, (uint64_t)0u, (uint64_t)0u, (uint64_t)0u, (uint64_t)0u};

__device__ u64 GPU_ALL_ROUND_CONSTANTS[MAX_WIDTH * N_ROUNDS] = {
    // WARNING: The AVX2 Goldilocks specialization relies on all round constants being in
    // 0..0xfffeeac900011537. If these constants are randomly regenerated, there is a ~.6% chance
    // that this condition will no longer hold.
    //
    // WARNING: If these are changed in any way, then all the
    // implementations of Poseidon must be regenerated. See comments
    // in `poseidon_goldilocks.rs`.
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

class PoseidonPermutationGPU
{
public:
    /*
     * This is based on x = x_hi_hi * 2^96 + x_hi_lo * 2^64 + x_lo (x_lo is 64 bits)
     * Note that EPSILON is 0xFFFFFFFF, that's why we can do &
     */
    __device__ inline static gl64_t reduce128(uint128_t x)
    {
        gl64_t x_lo = uint128_t::u128tou64(x);
        gl64_t x_hi = uint128_t::u128t_hi_ou64(x);

        gl64_t x_hi_hi = x_hi >> 32;
        gl64_t x_hi_lo = x_hi & 0xFFFFFFFF;

        gl64_t t0 = x_lo - x_hi_hi;
        gl64_t t1 = x_hi_lo * 0xFFFFFFFF;
        gl64_t t2 = t0 + t1;
        return t2;
    }

    __device__ inline static gl64_t from_noncanonical_u96(gl64_t n_lo, gl64_t n_hi)
    {
        gl64_t t1 = (gl64_t)n_hi * (gl64_t)0xFFFFFFFF;
        return n_lo + t1;
    }

    __device__ gl64_t mds_row_shf(u32 r, gl64_t *v)
    {
        assert(r < SPONGE_WIDTH);
        // The values of `MDS_MATRIX_CIRC` and `MDS_MATRIX_DIAG` are
        // known to be small, so we can accumulate all the products for
        // each row and reduce just once at the end (done by the
        // caller).

        uint128_t res = 0;
        for (u32 i = 0; i < 12; i++)
        {
            uint64_t tmp1 = GPU_MDS_MATRIX_CIRC[i];
            uint128_t tmp2 = v[(i + r) % SPONGE_WIDTH];
            res = res + tmp2 * tmp1;
        }
        uint64_t tmp1 = GPU_MDS_MATRIX_DIAG[r];
        uint128_t tmp2 = v[r];
        res = res + tmp2 * tmp1;

        return from_noncanonical_u96(uint128_t::u128tou64(res), uint128_t::u128t_hi_ou64(res) & 0xFFFFFFFF);
    }

    __device__ void mds_layer(gl64_t *state, gl64_t *result)
    {
        for (u32 i = 0; i < SPONGE_WIDTH; i++)
            result[i] = (uint64_t)0u;

        gl64_t s[SPONGE_WIDTH];

        for (int r = 0; r < SPONGE_WIDTH; r++)
        {
            s[r] = state[r];
        }

        for (u32 r = 0; r < SPONGE_WIDTH; r++)
        {
            // here the result is already reduced
            result[r] = mds_row_shf(r, s);
        }
    }

    __device__ inline void constant_layer(gl64_t *state, u32 *round_ctr)
    {
        for (int i = 0; i < 12; i++)
        {
            gl64_t tmp = GPU_ALL_ROUND_CONSTANTS[i + SPONGE_WIDTH * (*round_ctr)];
            state[i] = state[i] + tmp;
        }
    }

    __device__ static inline gl64_t sbox_monomial(const gl64_t &x)
    {
        gl64_t x2 = x * x;
        gl64_t x4 = x2 * x2;
        gl64_t x3 = x * x2;
        return x3 * x4;
    }

    __device__ inline void sbox_layer(gl64_t *state)
    {
        for (int i = 0; i < 12; i++)
        {
            state[i] = sbox_monomial(state[i]);
        }
    }

    __device__ inline void full_rounds(gl64_t *state, u32 *round_ctr)
    {
        gl64_t res[12] = {(uint64_t)0u};
        gl64_t *pres = res;

        // printf("Before full rounds 1: ");
        // print_perm(state, 12);

        for (int k = 0; k < HALF_N_FULL_ROUNDS; k++)
        {
            constant_layer(state, round_ctr);
            // printf("Constant layer: ");
            // print_perm(state, 12);
            sbox_layer(state);
            // printf("SBox layer: ");
            // print_perm(state, 12);
            mds_layer(state, pres);
            // printf("MDS layer: ");
            // print_perm(state, 12);
            *round_ctr += 1;
            gl64_t *aux = state;
            state = pres;
            pres = aux;
        }
    }

    __device__ void partial_rounds_naive(gl64_t *state, u32 *round_ctr)
    {
        gl64_t res[12] = {(uint64_t)0u};
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

    // Arrys of size SPIONGE_WIDTH
    __device__ gl64_t *poseidon_naive(gl64_t *input)
    {
        // auto state = input;
        // GoldilocksField state[SPONGE_WIDTH] = {GoldilocksField::Zero()};
        // for (int i = 0; i < SPONGE_WIDTH; i++)
        //    state[i] = input[i];
        gl64_t *state = input;
        u32 round_ctr = 0;

        full_rounds(state, &round_ctr);
        // print_perm(state, SPONGE_WIDTH);
        partial_rounds_naive(state, &round_ctr);
        // print_perm(state, SPONGE_WIDTH);
        full_rounds(state, &round_ctr);
        // print_perm(state, SPONGE_WIDTH);
        assert(round_ctr == N_ROUNDS);

        return state;
    }

    gl64_t state[SPONGE_WIDTH];

public:
    __device__ PoseidonPermutationGPU()
    {
        for (int i = 0; i < SPONGE_WIDTH; i++)
            state[i] = (uint64_t)0u;
    }

    __device__ void set_from_slice(gl64_t *elts, u32 len, u32 start_idx)
    {
        assert(start_idx + len <= SPONGE_WIDTH);
        for (int i = 0; i < len; i++)
        {
            this->state[start_idx + i] = elts[i];
        }
    }

    __device__ void permute()
    {
        poseidon_naive(state);
    }

    __device__ gl64_t *squeeze(u32 size)
    {
        // assert(size <= SPONGE_WIDTH);
        return state;
    }
};

__device__ void hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
{    
    PoseidonPermutationGPU perm = PoseidonPermutationGPU();

    // Absorb all input chunks.
    for (u32 idx = 0; idx < num_inputs; idx += SPONGE_RATE)
    {
        perm.set_from_slice(inputs + idx, MIN(SPONGE_RATE, num_inputs - idx), 0);
        perm.permute();
    }

    // Squeeze until we have the desired number of outputs.
    // assert(num_outputs == NUM_HASH_OUT_ELTS);
    gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
    for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
    {
        hash[i] = ret[i];
    }
}

__device__ void hash_two(gl64_t *hash1, gl64_t *hash2, gl64_t *hash)
{
    PoseidonPermutationGPU perm = PoseidonPermutationGPU();
    perm.set_from_slice(hash1, NUM_HASH_OUT_ELTS, 0);
    perm.set_from_slice(hash2, NUM_HASH_OUT_ELTS, NUM_HASH_OUT_ELTS);
    perm.permute();
    gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
    for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
    {
        hash[i] = ret[i];
    }
}

#endif // __POSEIDON_V2_CUH__