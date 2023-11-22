__global__
void generate_partial_twiddles(fr_t (*roots)[WINDOW_SIZE],
                               const fr_t root_of_unity)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    assert(tid < WINDOW_SIZE);
    fr_t root;

    root = root_of_unity^tid;

    roots[0][tid] = root;

    for (int off = 1; off < WINDOW_NUM; off++) {
        for (int i = 0; i < LG_WINDOW_SIZE; i++)
#if defined(__CUDA_ARCH__)
            root.sqr();
#else
            root *= root;
#endif
        roots[off][tid] = root;
    }
}


/**
 * \param d_radixX_twiddles - pointers to the twiddles to be allocated. size is 64 + 128 + 256 + 512 + 32
 * \param root6 - root6^{1<<6} = 1 for forward, inversed for inverse
 * \param root7 - root6^{1<<7} = 1 for forward, inversed for inverse
 * \param root8 -
 * \param root9 - 
 * \param root10 -
*/
__global__
void generate_all_twiddles(fr_t* d_radixX_twiddles, const fr_t root6,
                                                    const fr_t root7,
                                                    const fr_t root8,
                                                    const fr_t root9,
                                                    const fr_t root10)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int pow;
    fr_t root_of_unity;

    if (tid < 64) {
        pow = tid;
        root_of_unity = root7;
    } else if (tid < 64 + 128) {
        pow = tid - 64;
        root_of_unity = root8;
    } else if (tid < 64 + 128 + 256) {
        pow = tid - 64 - 128;
        root_of_unity = root9;
    } else if (tid < 64 + 128 + 256 + 512) {
        pow = tid - 64 - 128 - 256;
        root_of_unity = root10;
    } else if (tid < 64 + 128 + 256 + 512 + 32) {
        pow = tid - 64 - 128 - 256 - 512;
        root_of_unity = root6;
    } else {
        assert(false);
    }

    d_radixX_twiddles[tid] = root_of_unity^pow;
}