#ifndef GASP_SPMV_CUH
#define GASP_SPMV_CUH

#include "Semirings.cuh"
#include "common.h"


namespace gasp {

template <typename SR, typename D, typename I>
__device__ void SpMV(const size_t m, const size_t n,
                    D * d_vals, I * d_colinds, I * d_rowptrs,
                    __constant__ D * x, D * y)
{
    using WarpReduce = cub::WarpReduce<D>;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t wid = tid / 32;
    const uint32_t lid = threadIdx.x % 32;

    if (wid >= m) return;

    const I row_start = d_rowptrs[wid];
    const I row_end = d_rowptrs[wid+1];

    D thread_data = 0;

    for (I j = row_start + lid; j < row_end; j += 32) {
        const I colidx = d_colinds[j];
        const D r = SR::mult(d_vals[j], x[colidx]);
        thread_data = SR::add(thread_data, r);
    }

    D result = WarpReduce(temp_storage).Sum(thread_data);

    if (lid==0) {
        y[wid] = result;
    }

}

template <typename SR, typename D, typename I>
__global__ void SpMV_wrapper(const size_t m, const size_t n,
                            D * d_vals, I * d_colinds, I * d_rowptrs,
                            __constant__ D * x, D * y)
{
    SpMV<SR>(m, n, d_vals, d_colinds, d_rowptrs, x, y);
}

template <typename SR, typename D, typename I>
void SpMV_host( GaspCsr<D, I>& A, D * x, D * y)
{
    const uint32_t tpb = 256;
    const uint32_t wpb = 256 / 32;
    const uint32_t blocks = std::ceil((double)A.get_nnz() / (double)wpb);
    SpMV_wrapper<SR, D, I><<<blocks, tpb>>>
                    (A.get_rows(), A.get_cols(), 
                     A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                     x, y);
}


}
#endif
