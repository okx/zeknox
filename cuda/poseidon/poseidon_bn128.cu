#include "int_types.h"
#include "element_bn128.cuh"
#include "poseidon_bn128.hpp"

#include "cuda_utils.cuh"

#ifdef USE_CUDA
#include "poseidon.cuh"
typedef gl64_t GoldilocksField;
#else
#include <stdlib.h>
#include <string.h>
#include "poseidon.hpp"
#endif

#include "poseidon_bn128_constants.h"

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
DEVICE void PoseidonBN128Hasher::gpu_hash_one(gl64_t *data, u32 data_size, gl64_t *digest)
{
	PoseidonPermutationBN128 perm = PoseidonPermutationBN128();

	// Absorb all input chunks.
	for (u32 idx = 0; idx < data_size; idx += SPONGE_RATE)
	{
		perm.set_from_slice(data + idx, MIN(SPONGE_RATE, data_size - idx), 0);
		perm.permute();
	}

	gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
	for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
	{
		digest[i] = ret[i];
	}
}
#else
DEVICE void PoseidonBN128Hasher::cpu_hash_one(u64 *data, u64 data_size, u64 *digest)
{
	assert(data_size > NUM_HASH_OUT_ELTS);

	GoldilocksField *in = (GoldilocksField *)malloc(data_size * sizeof(GoldilocksField));
	for (u32 i = 0; i < data_size; i++)
	{
		in[i] = GoldilocksField(data[i]);
	}

	PoseidonPermutationBN128 perm = PoseidonPermutationBN128();

	u64 idx = 0;
	while (idx < data_size)
	{
		perm.set_from_slice(in + idx, MIN(PoseidonPermutation::RATE, (data_size - idx)), 0);
		perm.permute();
		idx += PoseidonPermutation::RATE;
	}

	HashOut out = perm.squeeze(NUM_HASH_OUT_ELTS);

	for (u64 i = 0; i < NUM_HASH_OUT_ELTS; i++)
	{
		digest[i] = out.elements[i].get_val();
	}
	free(in);
}
#endif

#ifdef USE_CUDA
DEVICE void PoseidonBN128Hasher::gpu_hash_two(gl64_t *digest_left, gl64_t *digest_right, gl64_t *digest)
{
	PoseidonPermutationBN128 perm = PoseidonPermutationBN128();
	perm.set_from_slice(digest_left, NUM_HASH_OUT_ELTS, 0);
	perm.set_from_slice(digest_right, NUM_HASH_OUT_ELTS, NUM_HASH_OUT_ELTS);
	perm.permute();
	gl64_t *ret = perm.squeeze(NUM_HASH_OUT_ELTS);
	for (u32 i = 0; i < NUM_HASH_OUT_ELTS; i++)
	{
		digest[i] = ret[i];
	}
}
#else
DEVICE void PoseidonBN128Hasher::cpu_hash_two(u64 *digest_left, u64 *digest_right, u64 *digest)
{
	HashOut in_l = HashOut(digest_left, NUM_HASH_OUT_ELTS);
	HashOut in_r = HashOut(digest_right, NUM_HASH_OUT_ELTS);

	PoseidonPermutationBN128 perm = PoseidonPermutationBN128();
	perm.set_from_slice(in_l.elements, in_l.n_elements, 0);
	perm.set_from_slice(in_r.elements, in_r.n_elements, NUM_HASH_OUT_ELTS);

	perm.permute();

	HashOut out = perm.squeeze(NUM_HASH_OUT_ELTS);

	for (u64 i = 0; i < NUM_HASH_OUT_ELTS; i++)
	{
		digest[i] = out.elements[i].get_val();
	}
}
#endif

// #define TESTING
#ifdef TESTING

#ifdef USE_CUDA
__global__
#endif
	void
	test(u64 *out)
{
#ifdef USE_CUDA
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	out = out + tid * 12;
#endif
	u64 inp[12] = {8917524657281059100u,
				   13029010200779371910u,
				   16138660518493481604u,
				   17277322750214136960u,
				   1441151880423231822u,
				   0, 0, 0, 0, 0, 0, 0};

	PoseidonPermutationBN128::permute_fn(inp, out);
}

int main2()
{
	u64 cpu_out[12 * 32];

#ifdef USE_CUDA
	u64 *gpu_out;
	CHECKCUDAERR(cudaMalloc(&gpu_out, 12 * 8 * 32));
	test<<<1, 32>>>(gpu_out);
	CHECKCUDAERR(cudaMemcpy(cpu_out, gpu_out, 12 * 8 * 32, cudaMemcpyDeviceToHost));

	for (int k = 0; k < 32; k++)
	{
		printf("Output %d:\n", k);
		for (int i = 0; i < 12; i++)
		{
			printf("%lu\n", cpu_out[12 * k + i]);
		}
	}
#else
	test(cpu_out);

	printf("Output:\n");
	for (int i = 0; i < 12; i++)
	{
		printf("%lu\n", cpu_out[i]);
	}
#endif

	return 0;
}

int main()
{
	u64 out[4] = {0};

#ifdef USE_CUDA
	u64 *gpu_out;
	CHECKCUDAERR(cudaMalloc(&gpu_out, 12 * 8 * 32));
	test<<<1, 32>>>(gpu_out);
	CHECKCUDAERR(cudaMemcpy(out, gpu_out, 4 * 8, cudaMemcpyDeviceToHost));
	CHECKCUDAERR(cudaFree(gpu_out));
#else
	u64 inp[5] = {8917524657281059100u, 13029010200779371910u, 16138660518493481604u, 17277322750214136960u, 1441151880423231822u};
	cpu_poseidon_bn128_hash_one(inp, 5, out);
#endif
	printf("Output:\n");
	for (int i = 0; i < 4; i++)
	{
		printf("%lu\n", out[i]);
	}
	assert(out[0] == 16736853722845225729u);
	assert(out[1] == 1446699130810517790u);
	assert(out[2] == 15445626857806971868u);
	assert(out[3] == 6331160477881736675u);
	printf("Test ok!\n");
	return 0;
}

#endif // TESTING