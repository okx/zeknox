#ifndef __CRYPTO_NTT_NTT_H__
#define __CRYPTO_NTT_NTT_H__
#include <map>
namespace Ntt_Types
{
    enum InputOutputOrder
    {
        NN,
        NR,
        RN,
        RR
    };
    enum Direction
    {
        forward,
        inverse
    };
    enum Type
    {
        standard,
        coset
    };
    enum Algorithm
    {
        GS,
        CT
    };

}

// namespace ntt {
//         /**
//      * @struct Domain
//      * Struct containing information about the domain on which (i)NTT is evaluated i.e. twiddle factors.
//      * Twiddle factors are private, static and can only be set using [InitDomain](@ref InitDomain) function.
//      * The internal representation of twiddles is prone to change in accordance with changing [NTT](@ref NTT) algorithm.
//      * @tparam S The type of twiddle factors \f$ \{ \omega^i \} \f$. Must be a field.
//      */
//     class Domain
//     {
//     public:
//         static int max_size;
//         static uint64_t *twiddles;
//         static std::map<uint64_t, int> coset_index_map;

//         // public:
//         //     friend class NTT;
//         // friend void NTT::InitDomain(fr_t primitive_root, const gpu_t &gpu);

//         // friend RustError NTT::Batch(const gpu_t &gpu, fr_t *inout, uint32_t lg_domain_size, uint32_t batch_size,
//         //                        InputOutputOrder order, Direction direction,
//         //                        Type type, bool coset_ext_pow = false, bool are_outputs_on_device = false);
//     };
// }

#endif