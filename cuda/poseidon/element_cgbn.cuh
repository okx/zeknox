#include "cgbn_common.cuh"

__device__ env_t::cgbn_t MODULUS;

__global__ void goldilocks_set_constants(cgbn_error_report_t *report)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > 0)
        return;
    context_t bn_context(cgbn_report_monitor, report, tid); // construct a context
    env_t bn_env(bn_context.env<env_t>());                  // construct an environment

    cgbn_mem_t<BITS> o;
    o._limbs[0] = 0x00000001;
    o._limbs[1] = 0xFFFFFFFF;
    o._limbs[2] = 0x00000000;
    o._limbs[3] = 0x00000000;
    cgbn_load(bn_env, MODULUS, &o);

    o._limbs[0] = 0xFFFFFFFF;
    o._limbs[1] = 0x00000000;
    o._limbs[2] = 0x00000000;
    o._limbs[3] = 0x00000000;
    cgbn_load(bn_env, EPSILON, &o);

    o._limbs[0] = 0x00000000;
    o._limbs[1] = 0xFFFFFFFF;
    o._limbs[2] = 0x00000000;
    o._limbs[3] = 0x00000000;
    cgbn_load(bn_env, EPSILON2, &o);

    o._limbs[0] = 0xFFFFFFFF;
    o._limbs[1] = 0xFFFFFFFF;
    o._limbs[2] = 0x00000000;
    o._limbs[3] = 0x00000000;
    cgbn_load(bn_env, MASK, &o);
}

class FFElementGPU {
private:
    env_t *bn_env = NULL; // CGBN environment
    env_t::cgbn_t val;    // the actual value (64 bit)

    __device__ void get_env()
    {
        int tid = threadIdx.x;
        bn_env = cgbn_env[tid];
    }

public:
    __device__ FFElementGPU()
    {
        get_env();
        cgbn_set_ui32(*bn_env, val, 0);
    }

    __device__ FFElementGPU(env_t::cgbn_t val)
    {
        get_env();
        cgbn_set(*bn_env, this->val, val);
    }

    FFElementGPU NewElement() {
        return FFElementGPU();
    }

    __device__ env_t::cgbn_t get_val() const
    {
        return this->val;
    }

    // Modulo Exp
    __device__ FFElementGPU Exp(const FFElementGPU a, u32 p) {        
        env_t::cgbn_t pwr, res;
        cgbn_set_ui32(*bn_env, pwr, p);
        cgbn_modular_power(*bn_env, res, a.get_val(), pwr, MODULUS);
        return FFElementGPU(res);
    }

    // Modulo Add
    __device__ FFElementGPU Add(const FFElementGPU a, const FFElementGPU b) {        
        env_t::cgbn_t tmp, sum;
        cgbn_sub(*bn_env, tmp, MODULUS, b);
        if (cgbn_compare(*bn_env, a, tmp) < 0)
        {
            cgbn_add(*bn_env, tmp, a, b);            
            return FFElementGPU(tmp);
        }
        else {        
            cgbn_sub(*bn_env, tmp, MODULUS, a);
            cgbn_sub(*bn_env, sum, b, tmp);
            return FFElementGPU(sum);
        }
    }
};