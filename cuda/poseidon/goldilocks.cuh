#ifndef __GOLDILOCKS_CUH__
#define __GOLDILOCKS_CUH__

#include "cgbn_common.cuh"

__device__ env_t::cgbn_t ORDER;    // 18446744069414584321
__device__ env_t::cgbn_t EPSILON;  // 4294967295;  // ((u64)1 << 32) - 1 = 2^64 % ORDER
__device__ env_t::cgbn_t EPSILON2; // 18446744069414584320ul;   // 2^96 % ORDER
__device__ env_t::cgbn_t MASK;     // 18446744069414584320ul;   // 2^96 % ORDER

__device__ env_t* cgbn_env[2048];

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
    cgbn_load(bn_env, ORDER, &o);

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

class GoldilocksFieldGPU
{
private:
    env_t *bn_env = NULL; // CGBN environment
    env_t::cgbn_t val;    // the actual value (64 bit)

    __device__ void get_env()
    {
        int tid = threadIdx.x;
        bn_env = cgbn_env[tid];
    }

public:
    __device__ GoldilocksFieldGPU()
    {
        get_env();
        cgbn_set_ui32(*bn_env, val, 0);
    }

    __device__ GoldilocksFieldGPU(u32 val)
    {
        get_env();
        cgbn_set_ui32(*bn_env, this->val, val);
    }

    __device__ GoldilocksFieldGPU(cgbn_mem_t<64> *ptr)
    {
        get_env();
        cgbn_mem_t<BITS> tmp;
        tmp._limbs[0] = ptr->_limbs[0];
        tmp._limbs[1] = ptr->_limbs[1];
        tmp._limbs[2] = 0;
        tmp._limbs[3] = 0;
        cgbn_load(*bn_env, this->val, &tmp);
    }

    __device__ GoldilocksFieldGPU(u32 x0, u32 x1)
    {
        get_env();
        cgbn_mem_t<BITS> tmp;
        tmp._limbs[0] = x0;
        tmp._limbs[1] = x1;
        tmp._limbs[2] = 0;
        tmp._limbs[3] = 0;
        cgbn_load(*bn_env, this->val, &tmp);
    }

    __device__ GoldilocksFieldGPU(u64 x)
    {
        get_env();
        cgbn_mem_t<BITS> tmp;
        tmp._limbs[0] = x & 0xFFFFFFFF;
        tmp._limbs[1] = x >> 32;
        tmp._limbs[2] = 0;
        tmp._limbs[3] = 0;
        cgbn_load(*bn_env, this->val, &tmp);
    }

    __device__ GoldilocksFieldGPU(env_t::cgbn_t val)
    {
        get_env();
        cgbn_set(*bn_env, this->val, val);
    }

    __device__ static GoldilocksFieldGPU Zero()
    {
        return GoldilocksFieldGPU((u32)0);
    };

    __device__ static GoldilocksFieldGPU One()
    {
        return GoldilocksFieldGPU((u32)1);
    }

    __device__ static GoldilocksFieldGPU Two()
    {
        return GoldilocksFieldGPU((u32)2);
    }

    __device__ env_t::cgbn_t get_val() const
    {
        return this->val;
    }

    __device__ inline env_t::cgbn_t modulo_add(env_t::cgbn_t x, env_t::cgbn_t y) const
    {
        env_t::cgbn_t tmp, sum;
        cgbn_sub(*bn_env, tmp, ORDER, y);
        if (cgbn_compare(*bn_env, x, tmp) < 0)
        {
            cgbn_add(*bn_env, tmp, x, y);            
            return tmp;
        }
        else {        
            cgbn_sub(*bn_env, tmp, ORDER, x);
            cgbn_sub(*bn_env, sum, y, tmp);
            return sum;
        }
    }

    __device__ inline env_t::cgbn_t modulo_sub(env_t::cgbn_t x, env_t::cgbn_t y) const
    {
        env_t::cgbn_t sub;
        if (cgbn_compare(*bn_env, x, y) > 0)
        {
            cgbn_sub(*bn_env, sub, x, y);            
            return sub;
        }
        else {        
            env_t::cgbn_t tmp;
            cgbn_sub(*bn_env, tmp, ORDER, y);
            cgbn_add(*bn_env, sub, tmp, x);
            return sub;
        }
    }

    __device__ inline env_t::cgbn_t reduce128(env_t::cgbn_t x) const
    {
        env_t::cgbn_t x_hi, x_lo, x_hi_hi, x_hi_lo, t0, t1;
        cgbn_extract_bits(*bn_env, x_lo, x, 0, 64);
        cgbn_shift_right(*bn_env, x_hi, x, 64);
        cgbn_shift_right(*bn_env, x_hi_hi, x_hi, 32);
        cgbn_bitwise_and(*bn_env, x_hi_lo, x_hi, EPSILON);      
        t0 = modulo_sub(x_lo, x_hi_hi);
        cgbn_mul(*bn_env, t1, x_hi_lo, EPSILON);
        return modulo_add(t0, t1);        
    }

    __device__ GoldilocksFieldGPU operator+(const GoldilocksFieldGPU &rhs) const
    {        
        return GoldilocksFieldGPU(modulo_add(this->val, rhs.get_val()));
    }

    __device__ GoldilocksFieldGPU operator*(const GoldilocksFieldGPU &rhs) const
    {
        env_t::cgbn_t tmp;
        cgbn_mul(*bn_env, tmp, this->val, rhs.get_val());
        return GoldilocksFieldGPU(reduce128(tmp));
    }   

    __device__ GoldilocksFieldGPU from_canonical_u64(env_t::cgbn_t x)
    {
        assert(cgbn_compare(*bn_env, x, ORDER) < 0);
        return GoldilocksFieldGPU(x);
    }

    __device__ env_t::cgbn_t to_noncanonical_u64()
    {
        return this->val;
    }

    __device__ void to_u64(u64* ptr)
    {
        cgbn_mem_t<BITS> tmp;        
        cgbn_store(*bn_env, &tmp, this->val);
        *ptr = (u64)tmp._limbs[0] | (((u64)tmp._limbs[1]) << 32);
    }

    __device__ GoldilocksFieldGPU add_canonical_u64(const env_t::cgbn_t &rhs)
    {

        return *this + from_canonical_u64(rhs);
    }    

    __device__ GoldilocksFieldGPU from_noncanonical_u96(env_t::cgbn_t x)
    {
        env_t::cgbn_t x_lo, x_hi;
        cgbn_extract_bits(*bn_env, x_lo, x, 0, 64);
        cgbn_shift_right(*bn_env, x_hi, x, 64);
        env_t::cgbn_t t1;
        cgbn_mul(*bn_env, t1, x_hi, EPSILON);        
        return GoldilocksFieldGPU(modulo_add(x_lo, t1));
    }

    inline __device__ cgbn_mem_t<64> to_cgbn_u64()
    {
        cgbn_mem_t<64> ret;
        cgbn_mem_t<BITS> tmp;
        cgbn_store(*bn_env, &tmp, this->val);
        ret._limbs[0] = tmp._limbs[0];
        ret._limbs[1] = tmp._limbs[1];
        return ret;
    }

     inline __device__ void to_cgbn_u64(u32* ptr)
    {        
        cgbn_mem_t<BITS> tmp;
        cgbn_store(*bn_env, &tmp, this->val);
        *ptr = tmp._limbs[0];
        ptr++;
        *ptr = tmp._limbs[1];        
    }

    inline __device__ void to_cgbn_u64(cgbn_mem_t<64>* ptr)
    {        
        cgbn_mem_t<BITS> tmp;
        cgbn_store(*bn_env, &tmp, this->val);
        ptr->_limbs[0] = tmp._limbs[0];
        ptr->_limbs[1] = tmp._limbs[1];        
    }
};

#endif // __GOLDILOCKS_CUH__