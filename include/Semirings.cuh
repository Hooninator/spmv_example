#ifndef SEMIRINGS_CUH
#define SEMIRINGS_CUH

#include "common.h"

namespace gasp {

template <typename D>
struct PlusTimesSemiring
{

    inline static __host__ __device__ D mult(D a, D b)
    {
        return a*b;
    }

    inline static __host__ __device__ D add(D a, D b)
    {
        return a + b;
    }
};

}
#endif
