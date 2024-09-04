#include "types/int_types.h"
#include "poseidon/element_bn128.cuh"
#include "poseidon/poseidon_bn128.hpp"

#include "utils/cuda_utils.cuh"

#ifdef USE_CUDA
#include "poseidon/poseidon_permutation.cuh"
typedef gl64_t GoldilocksField;
#else
#include <cstring>
#include "poseidon/poseidon.hpp"
#include "poseidon/poseidon_permutation.hpp"
#endif

#include "poseidon/poseidon_bn128_constants.h"

#define NROUNDSF 8

CONST u32 NROUNDSP[16] = {56, 57, 56, 60, 60, 63, 64, 63, 60, 66, 60, 65, 70, 60, 64, 68};

#define SpongeChunkSize 31
#define SpongeInputs 16

#ifdef USE_CUDA
class PoseidonPermutationBN128 : public PoseidonPermutationGPU
#else
class PoseidonPermutationBN128 : public PoseidonPermutation
#endif
{
private:
	INLINE static FFE zero()
	{
		return FFE::NewElement();
	}

	// exp5 performs x^5 mod p
	// https://eprint.iacr.org/2019/458.pdf page 8
	INLINE static void exp5(FFE *a)
	{
		a->Exp5();
	}

	// exp5state perform exp5 for whole state
	INLINE static void exp5state(FFE *state, u32 size)
	{
		for (u32 i = 0; i < size; i++)
		{
			exp5(&state[i]);
		}
	}

	// ark computes Add-Round Key, from the paper https://eprint.iacr.org/2019/458.pdf
	INLINE static void ark(FFE *state, u32 size, const u64 c[100][4], u32 it)
	{
		for (u32 i = 0; i < size; i++)
		{
			FFE cc = FFE(c[it + i]);
			state[i].Add(state[i], cc);
		}
	}

	// mix returns [[matrix]] * [vector]
	DEVICE static void mix(FFE *state, u32 size, u32 t, const u64 m[5][5][4], FFE *newState)
	{
		FFE mul;

#pragma unroll
		for (u32 i = 0; i < size; i++)
		{
			newState[i].SetUint64(0);
#pragma unroll
			for (u32 j = 0; j < size; j++)
			{
				FFE mm = FFE(m[j][i]);
				mul.Mul(mm, state[j]);
				newState[i].Add(newState[i], mul);
			}
		}
#pragma unroll
		for (u32 i = 0; i < size; i++)
		{
			state[i] = newState[i];
		}
	}

#ifdef DEBUG
	static void print_elem(FFE elem)
	{
		printf("%lu %lu %lu %lu\n", elem.z[0], elem.z[1], elem.z[2], elem.z[3]);
	}

	static void print_state(FFE *state, int size)
	{
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				printf("%lu ", state[i].z[j]);
			}
			printf("\n");
		}
		printf("\n");
	}
#endif // DEBUG

public:
	// input and output of size 12
	DEVICE static void permute_fn(u64 *in, u64 *out)
	{
		FFE inp[4];
		for (u32 i = 0; i < 4; i++)
		{
			inp[i].z[0] = in[i * 3 + 2];
			inp[i].z[1] = in[i * 3 + 1];
			inp[i].z[2] = in[i * 3 + 0];
			inp[i].z[3] = 0;
			inp[i].ToMont();
		}

		const u32 t = 5;
		u32 nRoundsF = NROUNDSF;
		u32 nRoundsP = NROUNDSP[t - 2];

		FFE state[5]; // t
		state[1] = inp[0];
		state[2] = inp[1];
		state[3] = inp[2];
		state[4] = inp[3];
		ark(state, t, C, 0);

		FFE newState[5];
		for (u32 i = 0; i < nRoundsF / 2 - 1; i++)
		{
			exp5state(state, t);
			ark(state, t, C, (i + 1) * t);
			mix(state, t, t, M, newState);
		}

		exp5state(state, t);
		ark(state, t, C, (nRoundsF / 2) * t);
		mix(state, t, t, P, newState);

		for (u32 i = 0; i < nRoundsP; i++)
		{
			exp5(&state[0]);
			state[0].Add(state[0], C[(nRoundsF / 2 + 1) * t + i]);

			FFE mul;
			FFE newState0;
			mul.SetZero();
			newState0.SetZero();
			for (u32 j = 0; j < t; j++)
			{
				mul.Mul(S[(t * 2 - 1) * i + j], state[j]);
				newState0.Add(newState0, mul);
			}

			for (u32 k = 1; k < t; k++)
			{
				mul.SetZero();
				mul.Mul(state[0], S[(t * 2 - 1) * i + t + k - 1]);
				state[k].Add(state[k], mul);
			}
			state[0] = newState0;
		}

		for (u32 i = 0; i < nRoundsF / 2 - 1; i++)
		{
			exp5state(state, t);
			ark(state, t, C, (nRoundsF / 2 + 1) * t + nRoundsP + i * t);
			mix(state, t, t, M, newState);
		}
		exp5state(state, t);
		mix(state, t, t, M, newState);

		for (u32 i = 0; i < 4; i++)
		{
			state[i].FromMont();
			out[i * 3] = state[i].z[2];
			out[i * 3 + 1] = state[i].z[1];
			out[i * 3 + 2] = state[i].z[0];
		}
	}

	DEVICE void permute()
	{
		u64 inp[12];
		u64 out[12];
		get_state_as_canonical_u64(inp);
		permute_fn(inp, out);
		for (u32 i = 0; i < SPONGE_WIDTH; i++)
		{
			u64 val = (out[i] >= 0xFFFFFFFF00000001ul) ? out[i] - 0xFFFFFFFF00000001ul : out[i];
			set_state(i, val);
		}
	}
};

#ifdef USE_CUDA
DEVICE void PoseidonBN128Hasher::gpu_hash_one(gl64_t *inputs, u32 num_inputs, gl64_t *hash)
{
	PoseidonPermutationBN128 perm = PoseidonPermutationBN128();
	PoseidonPermutationGPU::gpu_hash_one_with_permutation(inputs, num_inputs, hash, &perm);
}
#else
DEVICE void PoseidonBN128Hasher::cpu_hash_one(u64 *input, u64 input_count, u64 *digest)
{
	PoseidonPermutationBN128 perm = PoseidonPermutationBN128();
	PoseidonPermutation::cpu_hash_one_with_permutation(input, input_count, digest, &perm);
}
#endif

#ifdef USE_CUDA
DEVICE void PoseidonBN128Hasher::gpu_hash_two(gl64_t *digest_left, gl64_t *digest_right, gl64_t *digest)
{
	PoseidonPermutationBN128 perm = PoseidonPermutationBN128();
	PoseidonPermutationGPU::gpu_hash_two_with_permutation(digest_left, digest_right, digest, &perm);
}
#else
DEVICE void PoseidonBN128Hasher::cpu_hash_two(u64 *digest_left, u64 *digest_right, u64 *digest)
{
	PoseidonPermutationBN128 perm = PoseidonPermutationBN128();
	PoseidonPermutation::cpu_hash_two_with_permutation(digest_left, digest_right, digest, &perm);
}
#endif // USE_CUDA
