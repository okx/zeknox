#ifndef __CRYPTO_MSM_PIPPENGER_CUH__
#define __CRYPTO_MSM_PIPPENGER_CUH__

#include <cuda.h>
#include <cooperative_groups.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cassert>

#include <utils/vec2d_t.hpp>
#include <utils/slice_t.hpp>

#include "sort.cuh"
#include "batch_addition.cuh"
#include <utils/exception.cuh>
#include <msm/kernels/pippenger_kernels.cuh>

#ifndef WARP_SZ
#define WARP_SZ 32
#endif
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif

#define MAX_TH 256
#define WINDOW_BIT_SIZE 16
/*
 * Break down |scalars| to signed |wbits|-wide digits.
 */

#ifdef __CUDA_ARCH__
// Transposed scalar_t
template <class scalar_t>
class scalar_T
{
    uint32_t val[sizeof(scalar_t) / sizeof(uint32_t)][WARP_SZ];

public:
    __device__ const uint32_t &operator[](size_t i) const { return val[i][0]; }
    __device__ scalar_T &operator()(uint32_t laneid)
    {
        return *reinterpret_cast<scalar_T *>(&val[0][laneid]);
    }
    __device__ scalar_T &operator=(const scalar_t &rhs)
    {
        for (size_t i = 0; i < sizeof(scalar_t) / sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

template <class scalar_t>
__device__
#ifndef CUDA_DEBUG
    __forceinline__
#endif
    static uint32_t
    get_wval(const scalar_T<scalar_t> &scalar, uint32_t off,
             uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1)
{
    // printf("invoke get_wval \n");
    uint32_t i = off / 32;
    uint64_t ret = scalar[i];

    if (i < top_i)
        ret |= (uint64_t)scalar[i + 1] << 32;

    return ret >> (off % 32);
}

__device__
#ifndef CUDA_DEBUG
    __forceinline__
#endif
    static uint32_t
    booth_encode(uint32_t wval, uint32_t wmask, uint32_t wbits)
{
    uint32_t sign = (wval >> wbits) & 1;
    wval = ((wval + 1) & wmask) >> 1;
    return sign ? 0 - wval : wval;
}
#endif

template <class scalar_t>
__launch_bounds__(1024) __global__
    void breakdown(vec2d_t<uint32_t> digits, const scalar_t scalars[], size_t len,
                   uint32_t nwins, uint32_t wbits, bool mont = true)
{
    assert(len <= (1U << 31) && wbits < 32);

#ifdef __CUDA_ARCH__
    extern __shared__ scalar_T<scalar_t> xchange[];
    const uint32_t tid = threadIdx.x;
    const uint32_t tix = threadIdx.x + blockIdx.x * blockDim.x;

    const uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1;
    const uint32_t wmask = 0xffffffffU >> (31 - wbits); // (1U << (wbits+1)) - 1;

    auto &scalar = xchange[tid / WARP_SZ](tid % WARP_SZ);

#pragma unroll 1
    for (uint32_t i = tix; i < (uint32_t)len; i += gridDim.x * blockDim.x)
    {
        auto s = scalars[i];

        if (mont)
        {
            s.from();
        }

        // clear the most significant bit
        uint32_t msb = s[top_i] >> ((scalar_t::nbits - 1) % 32);
        s.cneg(msb);
        msb <<= 31;

        scalar = s;

#pragma unroll 1
        for (uint32_t bit0 = nwins * wbits - 1, win = nwins; --win;)
        {
            bit0 -= wbits;
            uint32_t wval = get_wval(scalar, bit0, top_i);
            wval = booth_encode(wval, wmask, wbits);
            if (wval)
                wval ^= msb;
            digits[win][i] = wval;
        }

        uint32_t wval = s[0] << 1;
        wval = booth_encode(wval, wmask, wbits);
        if (wval)
            wval ^= msb;
        digits[0][i] = wval;
    }
#endif
}

#ifndef LARGE_L1_CODE_CACHE
#if __CUDA_ARCH__ - 0 >= 800
#define LARGE_L1_CODE_CACHE 1
#define ACCUMULATE_NTHREADS 384
#else
#define LARGE_L1_CODE_CACHE 0
#define ACCUMULATE_NTHREADS (bucket_t::degree == 1 ? 384 : 256)
#endif
#endif

#ifndef MSM_NTHREADS
#define MSM_NTHREADS 256
#endif
#if MSM_NTHREADS < 32 || (MSM_NTHREADS & (MSM_NTHREADS - 1)) != 0
#error "bad MSM_NTHREADS value"
#endif
#ifndef MSM_NSTREAMS
#define MSM_NSTREAMS 8
#elif MSM_NSTREAMS < 2
#error "invalid MSM_NSTREAMS"
#endif

template <class bucket_t,
          class affine_h,
          class bucket_h = class bucket_t::mem_t,
          class affine_t = class bucket_t::affine_t>
__launch_bounds__(ACCUMULATE_NTHREADS) __global__
    void accumulate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits,
                    /*const*/ affine_h points_[], const vec2d_t<uint32_t> digits,
                    const vec2d_t<uint32_t> histogram, uint32_t sid = 0)
{
    vec2d_t<bucket_h> buckets{buckets_, 1U << --wbits};
    const affine_h *points = points_;

    static __device__ uint32_t streams[MSM_NSTREAMS];
    uint32_t &current = streams[sid % MSM_NSTREAMS];
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    const uint32_t lane_id = laneid / degree;

    uint32_t x, y;
#if 1
    __shared__ uint32_t xchg;

    if (threadIdx.x == 0)
        xchg = atomicAdd(&current, blockDim.x / degree);
    __syncthreads();
    x = xchg + threadIdx.x / degree;
#else
    x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
    x = __shfl_sync(0xffffffff, x, 0) + lane_id;
#endif

    while (x < (nwins << wbits))
    {
        y = x >> wbits;
        x &= (1U << wbits) - 1;
        const uint32_t *h = &histogram[y][x];

        uint32_t idx, len = h[0];

        asm("{ .reg.pred %did;"
            "  shfl.sync.up.b32 %0|%did, %1, %2, 0, 0xffffffff;"
            "  @!%did mov.b32 %0, 0;"
            "}" : "=r"(idx) : "r"(len), "r"(degree));

        if (lane_id == 0 && x != 0)
            idx = h[-1];

        if ((len -= idx) && !(x == 0 && y == 0))
        {
            const uint32_t *digs_ptr = &digits[y][idx];
            uint32_t digit = *digs_ptr++;

            affine_t p = points[digit & 0x7fffffff];
            bucket_t bucket = p;
            bucket.cneg(digit >> 31);

            while (--len)
            {
                digit = *digs_ptr++;
                p = points[digit & 0x7fffffff];
                if (sizeof(bucket) <= 128 || LARGE_L1_CODE_CACHE)
                    bucket.add(p, digit >> 31);
                else
                    bucket.uadd(p, digit >> 31);
            }

            buckets[y][x] = bucket;
        }
        else
        {
            buckets[y][x].inf();
        }

        x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
        x = __shfl_sync(0xffffffff, x, 0) + lane_id;
    }

    cooperative_groups::this_grid().sync();

    if (threadIdx.x + blockIdx.x == 0)
        current = 0;
}

template <class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(256) __global__
    void integrate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits, uint32_t nbits)
{
    const uint32_t degree = bucket_t::degree;
    uint32_t Nthrbits = 31 - __clz(blockDim.x / degree);

    assert((blockDim.x & (blockDim.x - 1)) == 0 && wbits - 1 > Nthrbits);

    vec2d_t<bucket_h> buckets{buckets_, 1U << (wbits - 1)};
    extern __shared__ uint4 scratch_[];
    auto *scratch = reinterpret_cast<bucket_h *>(scratch_);
    const uint32_t tid = threadIdx.x / degree;
    const uint32_t bid = blockIdx.x;

    auto *row = &buckets[bid][0];
    uint32_t i = 1U << (wbits - 1 - Nthrbits);
    row += tid * i;

    uint32_t mask = 0;
    if ((bid + 1) * wbits > nbits)
    {
        uint32_t lsbits = nbits - bid * wbits;
        mask = (1U << (wbits - lsbits)) - 1;
    }

    bucket_t res, acc = row[--i];

    if (i & mask)
    {
        if (sizeof(res) <= 128)
            res.inf();
        else
            scratch[tid].inf();
    }
    else
    {
        if (sizeof(res) <= 128)
            res = acc;
        else
            scratch[tid] = acc;
    }

    bucket_t p;

#pragma unroll 1
    while (i--)
    {
        p = row[i];

        uint32_t pc = i & mask ? 2 : 0;
#pragma unroll 1
        do
        {
            if (sizeof(bucket_t) <= 128)
            {
                p.add(acc);
                if (pc == 1)
                {
                    res = p;
                }
                else
                {
                    acc = p;
                    if (pc == 0)
                        p = res;
                }
            }
            else
            {
                if (LARGE_L1_CODE_CACHE && degree == 1)
                    p.add(acc);
                else
                    p.uadd(acc);
                if (pc == 1)
                {
                    scratch[tid] = p;
                }
                else
                {
                    acc = p;
                    if (pc == 0)
                        p = scratch[tid];
                }
            }
        } while (++pc < 2);
    }

    __syncthreads();

    buckets[bid][2 * tid] = p;
    buckets[bid][2 * tid + 1] = acc;
}
#undef asm

#ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
template __global__ void accumulate<bucket_t, affine_t::mem_t>(bucket_t::mem_t buckets_[],
                                                               uint32_t nwins, uint32_t wbits,
                                                               /*const*/ affine_t::mem_t points_[],
                                                               const vec2d_t<uint32_t> digits,
                                                               const vec2d_t<uint32_t> histogram,
                                                               uint32_t sid);
template __global__ void batch_addition<bucket_t>(bucket_t::mem_t buckets[],
                                                  const affine_t::mem_t points[], size_t npoints,
                                                  const uint32_t digits[], const uint32_t &ndigits);
template __global__ void integrate<bucket_t>(bucket_t::mem_t buckets_[], uint32_t nwins,
                                             uint32_t wbits, uint32_t nbits);
template __global__ void breakdown<scalar_t>(vec2d_t<uint32_t> digits, const scalar_t scalars[],
                                             size_t len, uint32_t nwins, uint32_t wbits, bool mont);
#endif

#include <vector>
#include <utils/exception.cuh>
#include <utils/rusterror.h>
#include <utils/gpu_t.cuh>

template <class bucket_t, class point_t, class affine_t, class scalar_t>
class msm_t
{
    typedef typename bucket_t::mem_t bucket_h;
    typedef typename bucket_t::affine_t affine_h;
    const gpu_t &gpu;
    size_t npoints;
    uint32_t wbits, nwins;
    bucket_h *d_buckets;
    vec2d_t<uint32_t> d_hist;

public:
    affine_h *d_points;
    scalar_t *d_scalars;

    template <typename T>
    using vec_t = slice_t<T>;

    class result_t
    {
        bucket_t ret[MSM_NTHREADS / bucket_t::degree][2];

    public:
        result_t() {}
        inline operator decltype(ret) & () { return ret; }
        inline const bucket_t *operator[](size_t i) const { return ret[i]; }
    };

    constexpr static int lg2(size_t n)
    {
        int ret = 0;
        while (n >>= 1)
            ret++;
        return ret;
    }

public:
    msm_t(const affine_t points[], size_t np,
          size_t ffi_affine_sz = sizeof(affine_t), int device_id = -1)
        : gpu(select_gpu(device_id)), d_points(nullptr), d_scalars(nullptr)
    {
        std::cout << "Line: " << __LINE__ << "\n";
        printf("bucket_t::degree: %d\n", bucket_t::degree);
        npoints = (np + WARP_SZ - 1) & ((size_t)0 - WARP_SZ); // make npoints a multiple of WARP_SZ

        // confine the number of bits in each window in the ragne [10,18]
        wbits = 17;
        if (npoints > 192)
        {
            wbits = std::min(lg2(npoints + npoints / 2) - 8, 18);
            if (wbits < 10)
                wbits = 10;
        }
        else if (npoints > 0)
        {
            wbits = 10;
        }
        nwins = (scalar_t::bit_length() - 1) / wbits + 1;

        uint32_t row_sz = 1U << (wbits - 1);

        size_t d_buckets_sz = (nwins * row_sz) + (gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
        size_t d_blob_sz = (d_buckets_sz * sizeof(d_buckets[0])) + (nwins * row_sz * sizeof(uint32_t)) + (points ? npoints * sizeof(d_points[0]) : 0);
        // printf("nwins: %d, wbits: %d, d_buckets_sz: %d, d_blob_sz: %d, sm_count: %d, size_of_bucket: %d, row_sz:%d\n", nwins, wbits, d_buckets_sz, d_blob_sz, gpu.sm_count(), sizeof(d_buckets[0]), row_sz);
        d_buckets = reinterpret_cast<decltype(d_buckets)>(gpu.Dmalloc(d_blob_sz));
        d_hist = vec2d_t<uint32_t>(&d_buckets[d_buckets_sz], row_sz);
        if (points)
        {
            printf("copy host to device within pippenger\n");
            d_points = reinterpret_cast<decltype(d_points)>(d_hist[nwins]);
            gpu.HtoD(d_points, points, np, ffi_affine_sz);
            npoints = np;
        }
        else
        {
            npoints = 0;
        }
    }
    inline msm_t(vec_t<affine_t> points, size_t ffi_affine_sz = sizeof(affine_t),
                 int device_id = -1)
        : msm_t(points, points.size(), ffi_affine_sz, device_id) {
              //   std::cout << "Line: " << __LINE__ << "\n";
          };
    inline msm_t(int device_id = -1)
        : msm_t(nullptr, 0, 0, device_id) {
              //   std::cout << "Line: " << __LINE__ << "\n";
          };
    ~msm_t()
    {
        gpu.sync();
        if (d_buckets)
            gpu.Dfree(d_buckets);
    }

private:
    void digits(const scalar_t d_scalars[], size_t len,
                vec2d_t<uint32_t> &d_digits, vec2d_t<uint2> &d_temps, bool mont)
    {
        // Using larger grid size doesn't make 'sort' run faster, actually
        // quite contrary. Arguably because global memory bus gets
        // thrashed... Stepping far outside the sweet spot has significant
        // impact, 30-40% degradation was observed. It's assumed that all
        // GPUs are "balanced" in an approximately the same manner. The
        // coefficient was observed to deliver optimal performance on
        // Turing and Ampere...
        uint32_t grid_size = gpu.sm_count() / 3;
        while (grid_size & (grid_size - 1))
            grid_size -= (grid_size & (0 - grid_size));

        breakdown<<<2 * grid_size, 1024, sizeof(scalar_t) * 1024, gpu[2]>>>(
            d_digits, d_scalars, len, nwins, wbits, mont);
        CUDA_OK(cudaGetLastError());

        const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;
#if 0
        uint32_t win;
        for (win = 0; win < nwins-1; win++) {
            gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, 0u);
        }
        uint32_t top = scalar_t::bit_length() - wbits * win;
        gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u);
#else
        // On the other hand a pair of kernels launched in parallel run
        // ~50% slower but sort twice as much data...
        uint32_t top = scalar_t::bit_length() - wbits * (nwins - 1);
        uint32_t win;
        for (win = 0; win < nwins - 1; win += 2)
        {
            gpu[2].launch_coop(sort, {{grid_size, 2}, SORT_BLOCKDIM, shared_sz},
                               d_digits, len, win, d_temps, d_hist,
                               wbits - 1, wbits - 1, win == nwins - 2 ? top - 1 : wbits - 1);
        }
        if (win < nwins)
        {
            gpu[2].launch_coop(sort, {{grid_size, 1}, SORT_BLOCKDIM, shared_sz},
                               d_digits, len, win, d_temps, d_hist,
                               wbits - 1, top - 1, 0u);
        }
#endif
    }

public:
    RustError invoke(point_t &out, const affine_t *points_, size_t npoints,
                     const scalar_t *scalars, bool mont = true,
                     size_t ffi_affine_sz = sizeof(affine_t))
    {
        assert(this->npoints == 0 || npoints <= this->npoints);

        uint32_t lg_npoints = lg2(npoints + npoints / 2);

        size_t batch = 1 << (std::max(lg_npoints, wbits) - wbits);
        // printf("npoints: %d, lg_npoints: %d, wbits: %d, batch: %d\n", npoints,lg_npoints, wbits, batch);
        batch >>= 6;
        batch = batch ? batch : 1;
        uint32_t stride = (npoints + batch - 1) / batch;
        stride = (stride + WARP_SZ - 1) & ((size_t)0 - WARP_SZ);

        std::vector<result_t> res(nwins);
        std::vector<bucket_t> ones(gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);

        out.inf();
        point_t p;

        try
        {
            // |scalars| being nullptr means the scalars are pre-loaded to
            // |d_scalars|, otherwise allocate stride.
            size_t temp_sz = scalars ? sizeof(scalar_t) : 0;
            temp_sz = stride * std::max(2 * sizeof(uint2), temp_sz);

            // |points| being nullptr means the points are pre-loaded to
            // |d_points|, otherwise allocate double-stride.
            const char *points = reinterpret_cast<const char *>(points_);
            size_t d_point_sz = points ? (batch > 1 ? 2 * stride : stride) : 0;
            d_point_sz *= sizeof(affine_h);

            size_t digits_sz = nwins * stride * sizeof(uint32_t);

            dev_ptr_t<uint8_t> d_temp{temp_sz + digits_sz + d_point_sz, gpu[2], true, false};

            vec2d_t<uint2> d_temps{&d_temp[0], stride};
            vec2d_t<uint32_t> d_digits{&d_temp[temp_sz], stride};

            scalar_t *d_scalars = scalars ? (scalar_t *)&d_temp[0]
                                          : this->d_scalars;
            affine_h *d_points = points ? (affine_h *)&d_temp[temp_sz + digits_sz]
                                        : this->d_points;

            size_t d_off = 0; // device offset
            size_t h_off = 0; // host offset
            size_t num = stride > npoints ? npoints : stride;
            event_t ev;

            if (scalars)
                gpu[2].HtoD(&d_scalars[d_off], &scalars[h_off], num);
            digits(&d_scalars[0], num, d_digits, d_temps, mont);
            gpu[2].record(ev);

            if (points)
                gpu[0].HtoD(&d_points[d_off], &points[h_off],
                            num, ffi_affine_sz);

            for (uint32_t i = 0; i < batch; i++)
            {
                gpu[i & 1].wait(ev);

                batch_addition<bucket_t><<<gpu.sm_count(), BATCH_ADD_BLOCK_SIZE,
                                           0, gpu[i & 1]>>>(
                    &d_buckets[nwins << (wbits - 1)], &d_points[d_off], num,
                    &d_digits[0][0], d_hist[0][0]);
                CUDA_OK(cudaGetLastError());

                gpu[i & 1].launch_coop(accumulate<bucket_t, affine_h>,
                                       {gpu.sm_count(), 0},
                                       d_buckets, nwins, wbits, &d_points[d_off], d_digits, d_hist, i & 1);
                gpu[i & 1].record(ev);

                integrate<bucket_t><<<nwins, MSM_NTHREADS,
                                      sizeof(bucket_t) * MSM_NTHREADS / bucket_t::degree,
                                      gpu[i & 1]>>>(
                    d_buckets, nwins, wbits, scalar_t::bit_length());
                CUDA_OK(cudaGetLastError());

                if (i < batch - 1)
                {
                    h_off += stride;
                    num = h_off + stride <= npoints ? stride : npoints - h_off;

                    if (scalars)
                        gpu[2].HtoD(&d_scalars[0], &scalars[h_off], num);
                    gpu[2].wait(ev);
                    digits(&d_scalars[scalars ? 0 : h_off], num,
                           d_digits, d_temps, mont);
                    gpu[2].record(ev);

                    if (points)
                    {
                        size_t j = (i + 1) & 1;
                        d_off = j ? stride : 0;
                        gpu[j].HtoD(&d_points[d_off], &points[h_off * ffi_affine_sz],
                                    num, ffi_affine_sz);
                    }
                    else
                    {
                        d_off = h_off;
                    }
                }

                if (i > 0)
                {
                    collect(p, res, ones);
                    out.add(p);
                }

                gpu[i & 1].DtoH(ones, d_buckets + (nwins << (wbits - 1)));
                gpu[i & 1].DtoH(res, d_buckets, sizeof(bucket_h) << (wbits - 1));
                gpu[i & 1].sync();
            }
        }
        catch (const cuda_error &e)
        {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        collect(p, res, ones);
        out.add(p);

        return RustError{cudaSuccess};
    }

    RustError invoke(point_t &out, const affine_t *points, size_t npoints,
                     gpu_ptr_t<scalar_t> scalars, bool mont = true,
                     size_t ffi_affine_sz = sizeof(affine_t))
    {
        d_scalars = scalars;
        return invoke(out, points, npoints, nullptr, mont, ffi_affine_sz);
    }

    RustError invoke(point_t &out, vec_t<scalar_t> scalars, bool mont = true)
    {
        return invoke(out, nullptr, scalars.size(), scalars, mont);
    }

    RustError invoke(point_t &out, vec_t<affine_t> points,
                     const scalar_t *scalars, bool mont = true,
                     size_t ffi_affine_sz = sizeof(affine_t))
    {
        return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);
    }

    RustError invoke(point_t &out, vec_t<affine_t> points,
                     vec_t<scalar_t> scalars, bool mont = true,
                     size_t ffi_affine_sz = sizeof(affine_t))
    {
        return invoke(out, points, points.size(), scalars, mont, ffi_affine_sz);
    }

    RustError invoke(point_t &out, const std::vector<affine_t> &points,
                     const std::vector<scalar_t> &scalars, bool mont = true,
                     size_t ffi_affine_sz = sizeof(affine_t))
    {
        return invoke(out, points.data(),
                      std::min(points.size(), scalars.size()),
                      scalars.data(), mont, ffi_affine_sz);
    }

private:
    point_t integrate_row(const result_t &row, uint32_t lsbits)
    {
        const int NTHRBITS = lg2(MSM_NTHREADS / bucket_t::degree);

        assert(wbits - 1 > NTHRBITS);

        size_t i = MSM_NTHREADS / bucket_t::degree - 1;

        if (lsbits - 1 <= NTHRBITS)
        {
            size_t mask = (1U << (NTHRBITS - (lsbits - 1))) - 1;
            bucket_t res, acc = row[i][1];

            if (mask)
                res.inf();
            else
                res = acc;

            while (i--)
            {
                acc.add(row[i][1]);
                if ((i & mask) == 0)
                    res.add(acc);
            }

            return res;
        }

        point_t res = row[i][0];
        bucket_t acc = row[i][1];

        while (i--)
        {
            point_t raise = acc;
            for (size_t j = 0; j < lsbits - 1 - NTHRBITS; j++)
                raise.dbl();
            res.add(raise);
            res.add(row[i][0]);
            if (i)
                acc.add(row[i][1]);
        }

        return res;
    }

    void collect(point_t &out, const std::vector<result_t> &res,
                 const std::vector<bucket_t> &ones)
    {
        struct tile_t
        {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);

        uint32_t y = nwins - 1, total = 0;

        grid[0].x = 0;
        grid[0].y = y;
        grid[0].dy = scalar_t::bit_length() - y * wbits;
        total++;

        while (y--)
        {
            grid[total].x = grid[0].x;
            grid[total].y = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins); /* zeroed */
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        auto n_workers = min((uint32_t)gpu.ncpus(), total);
        while (n_workers--)
        {
            gpu.spawn([&, this, total, counter]()
                      {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row(res[y], item->dy);
                    if (++row_sync[y] == 1)
                        ch.send(y);
                } });
        }

        point_t one = sum_up(ones);

        out.inf();
        size_t row = 0, ny = nwins;
        while (ny--)
        {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y)
            {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);
                if (y == 0)
                    break;
                for (size_t i = 0; i < wbits; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }
        out.add(one);
    }
};

/**
 * MSM on G1 curve using pippenger algorithm
 * @param out, MSM result; defined in Jacobian format; type is `point_t`, it contains X,Y,Z
 * @param points, the input points; defined in Affine format type is `affine_t`, it contains X,Y
 * @param scalars, the input scalars; Fr scalar field on the pairing curve
 * @param mont, specify whether the input scalars are in montgomery format
 *
 * @class bucket_t, defined as `xyzz_t` the extended jacobian format; contains X,Y,ZZ,ZZZ
 */
template <class bucket_t, class point_t, class affine_t, class scalar_t>
static RustError mult_pippenger(point_t *out, affine_t *points, size_t npoints,
                                scalar_t *scalars, bool mont = true,
                                size_t ffi_affine_sz = sizeof(affine_t))
{
    // printf("invoke mult_pippenger mont: %d, ffi_affine_sz:%d \n", mont, ffi_affine_sz);
    try
    {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        msm.d_points = points; // TODO: need a proper way to set the device pointer
        msm.d_scalars = scalars;
        return msm.invoke(*out, slice_t<affine_t>{points, npoints},
                          scalars, mont, ffi_affine_sz);
    }
    catch (const cuda_error &e)
    {
        out->inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}

#if defined(FEATURE_BN254)

/**
 * @param large_bucket_factor, a factor for determing the large bucket; avg_bucket_size = size / (1 << c);
 * any bucket with count > avg_bucket_size * large_bucket_factor is considerred a large bucket
 * @param msm_size, total number of points
 */
template <typename S, typename P, typename A>
static RustError mult_pippenger_g2_internal(
    P *result,
    A *points,
    S *scalars,
    unsigned msm_size,
    bool mont,
    bool on_device,
    bool big_triangle,
    unsigned large_bucket_factor)
{
    bucket_method_msm(S::NBITS, WINDOW_BIT_SIZE, scalars, points, msm_size, mont, result, on_device, big_triangle, large_bucket_factor);
    RustError r{0};
    return r;
}

/**
 * this function computes msm using the bucket method
 * @param bitsize, bit size of scalar; e.g 254 for BN254 field;
 * @param on_device whetehr put result on device; default is `false`
 */
template <typename S, typename P, typename A>
void bucket_method_msm(
    unsigned bitsize,
    unsigned c,
    S *scalars,
    A *points,
    unsigned size,
    bool mont,
    P *final_result,
    bool on_device,
    bool big_triangle,
    unsigned large_bucket_factor)
{

    cudaStream_t stream;
    CUDA_OK(cudaStreamCreate(&stream));

    printf("bucket_method_msm, bitsize: %d, c: %d, size: %d, large_bucket_factor:%d \n", bitsize, c, size, large_bucket_factor);
    S *d_scalars;
    A *d_points;
    if (!on_device)
    {
        CUDA_OK(cudaMallocAsync(&d_scalars, sizeof(S) * size, stream));
        CUDA_OK(cudaMallocAsync(&d_points, sizeof(A) * size, stream));
        CUDA_OK(cudaMemcpyAsync(d_scalars, scalars, sizeof(S) * size, cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d_points, points, sizeof(A) * size, cudaMemcpyHostToDevice, stream));
    }
    else
    {
        d_scalars = scalars;
        d_points = points;
    }

    P *buckets;

    unsigned num_of_window = (bitsize + c - 1) / c; // number of windows
    unsigned msm_log_size = ceil(log2(size));
    unsigned num_of_windows_bitsize = ceil(log2(num_of_window));
    unsigned total_num_of_buckets = num_of_window << c; // each bucket module contains 1<<c buckets; total is num_of_window << c
    // printf("bucket_method_msm, bitsize: %d, c: %d, num_of_window: %d, total_num_of_buckets:%d \n", bitsize, c, num_of_window, total_num_of_buckets);
    CUDA_OK(cudaMallocAsync(&buckets, sizeof(P) * total_num_of_buckets, stream));

    // launch the bucket initialization kernel with maximum threads
    unsigned NUM_THREADS = 1 << 10; // NOTE: can tune this for better performance;
    unsigned NUM_BLOCKS = (total_num_of_buckets + NUM_THREADS - 1) / NUM_THREADS;
    initialize_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, total_num_of_buckets);

    // indicate which bucket the scalar belongs to within each window; hence the total size is size*num_of_window;
    // additional size is allocated in the front for the following sorting purpose
    unsigned *scalars_window_bucket_indices;
    // binded to scalars scalars_window_bucket_indices
    unsigned *points_window_bucket_indices;

    CUDA_OK(cudaMallocAsync(&scalars_window_bucket_indices, sizeof(unsigned) * size * (num_of_window + 1), stream));
    CUDA_OK(cudaMallocAsync(&points_window_bucket_indices, sizeof(unsigned) * size * (num_of_window + 1), stream));

    // split scalars into digits
    printf("initialize split_scalars_kernel: size: %d, msm_log_size: %d, num_of_window: %d, num_of_windows_bitsize: %d \n", size, msm_log_size, num_of_window, num_of_windows_bitsize);
    if (mont)
    {
        transfrom_scalars_and_points_from_mont<<<(size + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(d_scalars, d_points, size);
    }

    NUM_BLOCKS = (size + NUM_THREADS - 1) / NUM_THREADS;
    // `scalars_window_bucket_indices` & `points_window_bucket_indices` with offset `size`: leaving the first window free for the out of place sort later
    split_scalars_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(scalars_window_bucket_indices + size, points_window_bucket_indices + size, d_scalars, size, msm_log_size, num_of_window, num_of_windows_bitsize, c);

    // sort indices - the indices are sorted from smallest to largest in order to group together the points that belong to each bucket
    unsigned *sort_indices_temp_storage{};
    size_t sort_indices_temp_storage_bytes;
    // Call the function initially with a NULL pointer to determine the required temporary storage size
    cub::DeviceRadixSort::SortPairs(
        sort_indices_temp_storage, sort_indices_temp_storage_bytes, scalars_window_bucket_indices + size, scalars_window_bucket_indices,
        points_window_bucket_indices + size, points_window_bucket_indices, size, 0, sizeof(unsigned) * 8, stream);
    CUDA_OK(cudaMallocAsync(&sort_indices_temp_storage, sort_indices_temp_storage_bytes, stream));
    for (unsigned i = 0; i < num_of_window; i++)
    {
        unsigned offset_out = i * size;
        unsigned offset_in = offset_out + size;
        // sort the pair within each window and put the result in previous window;
        // the third last parameter is `begin_bit`, [optional] The least-significant bit index (inclusive) needed for key comparison
        // the second last parameter is `end_bit`, [optional] The most-significant bit index (exclusive) needed for key comparison (e.g., sizeof(unsigned int) * 8)
        cub::DeviceRadixSort::SortPairs(
            sort_indices_temp_storage, sort_indices_temp_storage_bytes, scalars_window_bucket_indices + offset_in,
            scalars_window_bucket_indices + offset_out, points_window_bucket_indices + offset_in, points_window_bucket_indices + offset_out, size, 0, sizeof(unsigned) * 8,
            stream);
    }
    CUDA_OK(cudaFreeAsync(sort_indices_temp_storage, stream));

    // stores the unique bucket indices;
    // [  0|0, 0|1, 0|2, ..., 0|(1<<c -1), \
    //    1|0, 1|1, 1|2, ..., 1|(1<<c -1), \
    // ...
    // ]
    unsigned *d_unique_bucket_indices;
    unsigned *d_bucket_count;
    unsigned *d_nof_buckets_to_compute;
    CUDA_OK(cudaMallocAsync(&d_unique_bucket_indices, sizeof(unsigned) * total_num_of_buckets, stream));
    CUDA_OK(cudaMallocAsync(&d_bucket_count, sizeof(unsigned) * total_num_of_buckets, stream));
    CUDA_OK(cudaMallocAsync(&d_nof_buckets_to_compute, sizeof(unsigned), stream));
    unsigned *encode_temp_storage{};
    size_t encode_temp_storage_bytes = 0;
    // It identifies consecutive repeated elements (runs) in an input array and outputs unique elements along with the count of occurrences for each run
    cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, scalars_window_bucket_indices, d_unique_bucket_indices, d_bucket_count,
        d_nof_buckets_to_compute, num_of_window * size, stream);
    CUDA_OK(cudaMallocAsync(&encode_temp_storage, encode_temp_storage_bytes, stream));
    cub::DeviceRunLengthEncode::Encode(
        encode_temp_storage, encode_temp_storage_bytes, scalars_window_bucket_indices, d_unique_bucket_indices, d_bucket_count,
        d_nof_buckets_to_compute, num_of_window * size, stream);
    CUDA_OK(cudaFreeAsync(encode_temp_storage, stream));



    unsigned *d_bucket_offsets; // stores where each new bucket begin; new means new `unique`
    CUDA_OK(cudaMalloc(&d_bucket_offsets, sizeof(unsigned) * total_num_of_buckets));
    unsigned *offsets_temp_storage{};
    size_t offsets_temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, d_bucket_count, d_bucket_offsets, total_num_of_buckets, stream);
    CUDA_OK(cudaMalloc(&offsets_temp_storage, offsets_temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(
        offsets_temp_storage, offsets_temp_storage_bytes, d_bucket_count, d_bucket_offsets, total_num_of_buckets, stream);
    CUDA_OK(cudaFreeAsync(offsets_temp_storage, stream));

    unsigned h_nof_buckets_to_compute;
    CUDA_OK(cudaMemcpyAsync(&h_nof_buckets_to_compute, d_nof_buckets_to_compute, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));

    // if all points are 0 just return point 0
    if (h_nof_buckets_to_compute == 0)
    {
        if (!on_device)
            final_result[0] = P::zero();
        else
        {
            P *h_final_result = (P *)malloc(sizeof(P));
            h_final_result[0] = P::zero();
            CUDA_OK(cudaMemcpyAsync(final_result, h_final_result, sizeof(P), cudaMemcpyHostToDevice, stream));
        }

        return;
    }

    // sort by bucket sizes
    unsigned *d_sorted_bucket_count;
    CUDA_OK(cudaMallocAsync(&d_sorted_bucket_count, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
    unsigned *d_sorted_bucket_offsets;
    CUDA_OK(cudaMallocAsync(&d_sorted_bucket_offsets, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
    unsigned *sort_offsets_temp_storage{};
    size_t sort_offsets_temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, d_bucket_count, d_sorted_bucket_count, d_bucket_offsets,
        d_sorted_bucket_offsets, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
    CUDA_OK(cudaMallocAsync(&sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, stream));
    cub::DeviceRadixSort::SortPairsDescending(
        sort_offsets_temp_storage, sort_offsets_temp_storage_bytes, d_bucket_count, d_sorted_bucket_count, d_bucket_offsets,
        d_sorted_bucket_offsets, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
    CUDA_OK(cudaFreeAsync(sort_offsets_temp_storage, stream));

    unsigned *d_sorted_unique_bucket_indices;
    CUDA_OK(cudaMallocAsync(&d_sorted_unique_bucket_indices, sizeof(unsigned) * h_nof_buckets_to_compute, stream));
    unsigned *sort_single_temp_storage{};
    size_t sort_single_temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        sort_single_temp_storage, sort_single_temp_storage_bytes, d_bucket_count, d_sorted_bucket_count, d_unique_bucket_indices,
        d_sorted_unique_bucket_indices, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
    CUDA_OK(cudaMallocAsync(&sort_single_temp_storage, sort_single_temp_storage_bytes, stream));
    cub::DeviceRadixSort::SortPairsDescending(
        sort_single_temp_storage, sort_single_temp_storage_bytes, d_bucket_count, d_sorted_bucket_count, d_unique_bucket_indices,
        d_sorted_unique_bucket_indices, h_nof_buckets_to_compute, 0, sizeof(unsigned) * 8, stream);
    CUDA_OK(cudaFreeAsync(sort_single_temp_storage, stream));

    // find large buckets
    unsigned avarage_size = size / (1 << c);
    unsigned bucket_th = large_bucket_factor * avarage_size;
    unsigned *nof_large_buckets;
    CUDA_OK(cudaMallocAsync(&nof_large_buckets, sizeof(unsigned), stream));
    CUDA_OK(cudaMemset(nof_large_buckets, 0, sizeof(unsigned)));

    unsigned TOTAL_THREADS = 129000; // TODO - device dependant
    unsigned cutoff_run_length = max(2, h_nof_buckets_to_compute / TOTAL_THREADS);
    unsigned cutoff_nof_runs = (h_nof_buckets_to_compute + cutoff_run_length - 1) / cutoff_run_length;

    NUM_THREADS = min(1 << 5, cutoff_nof_runs);
    NUM_BLOCKS = (cutoff_nof_runs + NUM_THREADS - 1) / NUM_THREADS;
    printf("h_nof_buckets_to_compute: %d, cutoff_run_length: %d, cutoff_nof_runs: %d, NUM_BLOCKS: %d, NUM_THREADS: %d bucket_th: %d\n", h_nof_buckets_to_compute, cutoff_run_length, cutoff_nof_runs,
           NUM_BLOCKS, NUM_THREADS, bucket_th);
    find_cutoff_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
        d_sorted_bucket_count, h_nof_buckets_to_compute, bucket_th, cutoff_run_length, nof_large_buckets);

    unsigned h_nof_large_buckets;
    CUDA_OK(cudaMemcpyAsync(&h_nof_large_buckets, nof_large_buckets, sizeof(unsigned), cudaMemcpyDeviceToHost, stream));

    unsigned h_max_res[2];
    CUDA_OK(cudaMemcpyAsync(&h_max_res[0], d_sorted_bucket_count, sizeof(unsigned) , cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaMemcpyAsync(&h_max_res[1], d_sorted_unique_bucket_indices, sizeof(unsigned) , cudaMemcpyDeviceToHost, stream));
    unsigned h_largest_bucket_count = h_max_res[0];
    unsigned h_nof_zero_large_buckets = h_max_res[1];
    unsigned large_buckets_to_compute =
        h_nof_large_buckets > h_nof_zero_large_buckets ? h_nof_large_buckets - h_nof_zero_large_buckets : 0;

    cudaStream_t stream2;
    CUDA_OK(cudaStreamCreate(&stream2));
    P *large_buckets;

    // printf("large_buckets_to_compute: %d, bucket_th: %d \n", large_buckets_to_compute, bucket_th);
    if (large_buckets_to_compute > 0 && bucket_th > 0)
    {
        unsigned threads_per_bucket = min(MAX_TH,
            1 << (unsigned)ceil(log2((h_largest_bucket_count + bucket_th - 1) / bucket_th)));
        unsigned max_bucket_size_run_length = (h_largest_bucket_count + threads_per_bucket - 1) / threads_per_bucket;
        unsigned total_large_buckets_size = large_buckets_to_compute * threads_per_bucket;
        // printf("sizeof(P): %d, total_large_buckets_size: %d\n", sizeof(P), total_large_buckets_size);
        CUDA_OK(cudaMallocAsync(&large_buckets, sizeof(P) * total_large_buckets_size, stream2));

        NUM_THREADS = min(1 << 8, total_large_buckets_size);
        NUM_BLOCKS = (total_large_buckets_size + NUM_THREADS - 1) / NUM_THREADS;
        // printf("NUM_BLOCKS: %d, NUM_THREADS: %d, h_nof_zero_large_buckets: %d, total_num_of_buckets: %d, max_bucket_size_run_length: %d\n",
        // NUM_BLOCKS, NUM_THREADS, h_nof_zero_large_buckets, total_num_of_buckets, max_bucket_size_run_length);
        accumulate_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream2>>>(
            large_buckets, d_sorted_bucket_offsets + h_nof_zero_large_buckets, d_sorted_bucket_count + h_nof_zero_large_buckets,
            d_sorted_unique_bucket_indices + h_nof_zero_large_buckets, points_window_bucket_indices, d_points, total_num_of_buckets,
            large_buckets_to_compute, c + num_of_windows_bitsize, c, threads_per_bucket, max_bucket_size_run_length);

        // reduce
        for (int s = total_large_buckets_size >> 1; s > large_buckets_to_compute - 1; s >>= 1)
        {
            NUM_THREADS = min(MAX_TH, s);
            NUM_BLOCKS = (s + NUM_THREADS - 1) / NUM_THREADS;
            single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream2>>>(
                large_buckets, large_buckets, s * 2, 0, 0, 0, s);

            //   CHECK_LAST_CUDA_ERROR();
        }

        // distribute
        NUM_THREADS = min(MAX_TH, large_buckets_to_compute);
        NUM_BLOCKS = (large_buckets_to_compute + NUM_THREADS - 1) / NUM_THREADS;
        distribute_large_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream2>>>(
            large_buckets, buckets, d_sorted_unique_bucket_indices + h_nof_zero_large_buckets, large_buckets_to_compute);
    }
    else
    {
        h_nof_large_buckets = 0;  // if bucket_th = 0; there is no large buckets
    }

    // launch the accumulation kernel with maximum threads
    if (h_nof_buckets_to_compute > h_nof_large_buckets)
    {
        NUM_THREADS = 1 << 8;
        NUM_BLOCKS = (h_nof_buckets_to_compute - h_nof_large_buckets + NUM_THREADS - 1) / NUM_THREADS;
        accumulate_buckets_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            buckets, d_sorted_bucket_offsets + h_nof_large_buckets, d_sorted_bucket_count + h_nof_large_buckets,
            d_sorted_unique_bucket_indices + h_nof_large_buckets, points_window_bucket_indices, d_points, total_num_of_buckets,
            h_nof_buckets_to_compute - h_nof_large_buckets, c + num_of_windows_bitsize, c);
    }
    // CUDA_OK(cudaFreeAsync(large_buckets, stream2));
    // all the large buckets need to be accumulated before the final summation
    CUDA_OK(cudaStreamSynchronize(stream2));
    CUDA_OK(cudaStreamDestroy(stream2));

    P *d_final_result;
    if (!on_device)
        CUDA_OK(cudaMallocAsync(&d_final_result, sizeof(P), stream));

    P *final_results;
    if (big_triangle)
    {
        CUDA_OK(cudaMallocAsync(&final_results, sizeof(P) * num_of_window, stream));
        // launch the bucket module sum kernel - a thread for each bucket module
        NUM_THREADS = num_of_window;
        NUM_BLOCKS = 1;

        big_triangle_sum_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(buckets, final_results, num_of_window, c);
    }
    else
    {
        unsigned source_bits_count = c;
        unsigned source_windows_count = num_of_window;
        unsigned source_buckets_count = total_num_of_buckets;
        P *source_buckets = buckets;
        buckets = nullptr;
        P *target_buckets;
        P *temp_buckets1;
        P *temp_buckets2;
        /**
         * Assuming C is 4, which represents a window size of 4 bits, and target_c is 2, indicating a target window size of 2 bits.

            Suppose source_windows_count is 64, signifying that there are 64 source windows. In this case, target_windows_count would be 128.

            If source_buckets is an array with a length of 1024:

            In the first round:
            The first call to single_stage_multi_reduction_kernel is made with nof_threads set to 512.
            The launch parameters are (2, 1, 1) and (256, 1, 1).
            The kernel parameters are (source_buckets, temp_buckets1, 16, 0, 0, 0, 512).
            Here, the "jump" variable is 8, and the operation is as follows:

            temp_buckets1[0] = source_buckets[0] + source_buckets[8]
            temp_buckets1[1] = source_buckets[1] + source_buckets[9]
            temp_buckets1[2] = source_buckets[2] + source_buckets[10]
            ...
            temp_buckets1[7] = source_buckets[7] + source_buckets[15]
            temp_buckets1[8] = source_buckets[16] + source_buckets[24]
            ...
            Ultimately, temp_buckets1 consists of 512 elements, where each element is obtained by taking pairs of elements from source_buckets at specific intervals and adding them together.
            In the second call to single_stage_multi_reduction_kernel, nof_threads remains at 512.
            The launch parameters are once again (2, 1, 1) and (256, 1, 1).
            The kernel parameters are (source_buckets, temp_buckets2, 4, 0, 0, 0, 512).
            This time, the "jump" variable is 2, and the operation is as follows:

            temp_buckets2[0] = source_buckets[0] + source_buckets[2]
            temp_buckets2[1] = source_buckets[1] + source_buckets[3]
            temp_buckets2[2] = source_buckets[4] + source_buckets[6]
            temp_buckets2[3] = source_buckets[5] + source_buckets[7]
            ...
            In the end, temp_buckets2 consists of 512 elements, and, similar to the first call, it pairs up elements from source_buckets at specific intervals and adds them together.
            It's important to note that these two calls do not handle low and high bits separately. Instead, they both involve adding pairs of elements with a specific step size from the source_buckets.
        */
        for (unsigned i = 0;; i++)
        {
            const unsigned target_bits_count = (source_bits_count + 1) >> 1;                        // c/2=8
            const unsigned target_windows_count = source_windows_count << 1;                        // nof bms*2 = 32
            const unsigned target_buckets_count = target_windows_count << target_bits_count;        // bms*2^c = 32*2^8
            CUDA_OK(cudaMallocAsync(&target_buckets, sizeof(P) * target_buckets_count, stream));    // 32*2^8*2^7 buckets
            CUDA_OK(cudaMallocAsync(&temp_buckets1, sizeof(P) * source_buckets_count / 2, stream)); // 32*2^8*2^7 buckets
            CUDA_OK(cudaMallocAsync(&temp_buckets2, sizeof(P) * source_buckets_count / 2, stream)); // 32*2^8*2^7 buckets

            if (source_bits_count > 0)
            {
                for (unsigned j = 0; j < target_bits_count; j++)
                {
                    /**
                     * there are two calls to single_stage_multi_reduction_kernel
                     * it splits windows of bitsize c into windows of bitsize c/2, and the first call computes the lower window and the second computes higher.
                     * An example for c=4, it looks like:
                     * 1*P_1 + 2*P_2 + ... + 15*P_15 = ((P_1 + P_2 + P_9 + P_13) + ... + 3*(P_3 + P_7 + P_11 + P_15)) + 4*((P_4 + P_5 + P_6 + P_7) + ... + 3*(P_12 + P_13 + P_14 + P_15))
                     * The first sum is the first single_stage_multi_reduction_kernel and the second sum (that's multiplied by 4) is the second call
                     */
                    unsigned nof_threads = (source_buckets_count >> (1 + j));
                    NUM_THREADS = min(MAX_TH, nof_threads);
                    NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS;
                    single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                        j == 0 ? source_buckets : temp_buckets1, j == target_bits_count - 1 ? target_buckets : temp_buckets1,
                        1 << (source_bits_count - j), j == target_bits_count - 1 ? 1 << target_bits_count : 0, 0, 0, nof_threads);

                    NUM_THREADS = min(MAX_TH, nof_threads);
                    NUM_BLOCKS = (nof_threads + NUM_THREADS - 1) / NUM_THREADS;
                    single_stage_multi_reduction_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
                        j == 0 ? source_buckets : temp_buckets2, j == target_bits_count - 1 ? target_buckets : temp_buckets2,
                        1 << (target_bits_count - j), j == target_bits_count - 1 ? 1 << target_bits_count : 0, 1, 0, nof_threads);
                }
            }
            if (target_bits_count == 1)
            {
                num_of_window = bitsize;
                CUDA_OK(cudaMallocAsync(&final_results, sizeof(P) * num_of_window, stream));
                NUM_THREADS = 32;
                NUM_BLOCKS = (num_of_window + NUM_THREADS - 1) / NUM_THREADS;
                last_pass_kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(target_buckets, final_results, num_of_window);
                c = 1;
                CUDA_OK(cudaFreeAsync(source_buckets, stream));
                CUDA_OK(cudaFreeAsync(target_buckets, stream));
                CUDA_OK(cudaFreeAsync(temp_buckets1, stream));
                CUDA_OK(cudaFreeAsync(temp_buckets2, stream));
                break;
            }
            CUDA_OK(cudaFreeAsync(source_buckets, stream));
            CUDA_OK(cudaFreeAsync(temp_buckets1, stream));
            CUDA_OK(cudaFreeAsync(temp_buckets2, stream));
            source_buckets = target_buckets;
            target_buckets = nullptr;
            temp_buckets1 = nullptr;
            temp_buckets2 = nullptr;
            source_bits_count = target_bits_count;
            source_windows_count = target_windows_count;
            source_buckets_count = target_buckets_count;
        }
    }

    // printf("double and add \n");

    // launch the double and add kernel, a single thread
    final_accumulation_kernel<P, S>
        <<<1, 1, 0, stream>>>(final_results, on_device ? final_result : d_final_result, 1, num_of_window, c);
    //  printf("free \n");

    CUDA_OK(cudaFreeAsync(final_results, stream));
    CUDA_OK(cudaStreamSynchronize(stream));

    if (!on_device)
        CUDA_OK(cudaMemcpyAsync(final_result, d_final_result, sizeof(P), cudaMemcpyDeviceToHost, stream));

    // free memory
    if (!on_device)
    {
        CUDA_OK(cudaFreeAsync(d_points, stream));
        CUDA_OK(cudaFreeAsync(d_scalars, stream));
        CUDA_OK(cudaFreeAsync(d_final_result, stream));
    }
    CUDA_OK(cudaFreeAsync(buckets, stream));

    CUDA_OK(cudaFreeAsync(scalars_window_bucket_indices, stream));
    CUDA_OK(cudaFreeAsync(points_window_bucket_indices, stream));
    CUDA_OK(cudaFreeAsync(d_unique_bucket_indices, stream));
    CUDA_OK(cudaFreeAsync(d_bucket_count, stream));
    CUDA_OK(cudaFreeAsync(d_nof_buckets_to_compute, stream));
    CUDA_OK(cudaFreeAsync(d_bucket_offsets, stream));

    CUDA_OK(cudaFreeAsync(d_sorted_bucket_count, stream));
    CUDA_OK(cudaFreeAsync(d_sorted_bucket_offsets, stream));
    CUDA_OK(cudaFreeAsync(d_sorted_unique_bucket_indices, stream));
    CUDA_OK(cudaFreeAsync(nof_large_buckets, stream));
    //   if (large_buckets_to_compute > 0 && bucket_th > 0) cudaFreeAsync(large_buckets, stream);
    CUDA_OK(cudaStreamSynchronize(stream));
    CUDA_OK(cudaStreamDestroy(stream));
}

#endif

#endif
