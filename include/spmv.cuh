#ifndef SPMV_CUH
#define SPMV_CUH

#include "common.h"


namespace spmv {



template <typename D, typename I>
__global__ void SpMV(const size_t m, const size_t n,
                    D * d_vals, I * d_colinds, I * d_rowptrs,
                    __constant__ D * d_x, D * d_y)
{
    /**** IMPLEMENT THIS KERNEL ****/
}

template <typename Matrix, typename D>
void SpMV_wrapper(Matrix& A, D * d_x, D * d_y)
{
    //**** CHANGE THESE VALUES ****//
    uint32_t threads_per_block = 256;
    uint32_t blocks = std::ceil((float)A.get_rows() / threads_per_block);

    // Call the kernel
    SpMV<<<blocks, threads_per_block>>>(A.get_rows(), A.get_cols(),
                   A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                   d_x, d_y);

    // Sync w/ the host
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
#endif
