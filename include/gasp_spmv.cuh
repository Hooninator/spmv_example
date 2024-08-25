#ifndef GASP_SPMV_CUH
#define GASP_SPMV_CUH

#include "Semirings.cuh"
#include "common.h"


namespace gasp {


template <typename SR>
struct SpMVScalar {

    template <typename D, typename I>
    static __device__ void SpMV(const size_t m, const size_t n,
                        D * d_vals, I * d_colinds, I * d_rowptrs,
                        __constant__ D * x, D * y)
    {
        const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid>=m) return;

        const I row_start = d_rowptrs[tid];
        const I row_end = d_rowptrs[tid+1];

        D thread_data = 0;

#pragma unroll
        for (I j = row_start; j < row_end; j++) 
        {
            const I colidx = d_colinds[j];
            const D r = SR::mult(d_vals[j], x[colidx]);
            thread_data = SR::add(thread_data, r);
        }

        y[tid] = SR::add(y[tid], thread_data);
    }


    template <typename Matrix>
    static std::pair<uint32_t, uint32_t> get_params(Matrix& A)
    {
        const uint32_t tpb = 512;
        const uint32_t blocks = std::ceil((double)A.get_rows() / (double)tpb);
        return {blocks, tpb};
    }

};

template <typename SR>
struct SpMVWarp {

    template <typename D, typename I>
    static __device__ void SpMV(const size_t m, const size_t n,
                        D * d_vals, I * d_colinds, I * d_rowptrs,
                        __constant__ D * x, D * y)
    {
        using WarpReduce = cub::WarpReduce<D>;
        __shared__ typename WarpReduce::TempStorage temp_storage;

        const uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint64_t wid = tid / warpSize;
        const uint64_t lid = threadIdx.x % warpSize;

        if (wid >= m) return;

        const I row_start = d_rowptrs[wid];
        const I row_end = d_rowptrs[wid+1];

        D thread_data = 0;

#pragma unroll
        for (I j = row_start + lid; j < row_end; j += warpSize) 
        {
            const I colidx = d_colinds[j];
            const D r = SR::mult(d_vals[j], x[colidx]);
            thread_data = SR::add(thread_data, r);
        }

        D result = WarpReduce(temp_storage).Sum(thread_data);

        if (lid==0) {
            //y[wid] = result;
            y[wid] = SR::add(y[wid], result);
        }

    }


    template <typename Matrix>
    static std::pair<uint32_t, uint32_t> get_params(Matrix& A)
    {
        const uint32_t tpb = 512;
        const uint32_t wpb = tpb / 32;
        const uint32_t blocks = std::ceil((double)A.get_rows() / (double)wpb);
        return {blocks, tpb};
    }
};


template <typename Kernel, typename D, typename I>
__global__ void SpMV_wrapper(const size_t m, const size_t n,
                                D * d_vals, I * d_colinds, I * d_rowptrs,
                                __constant__ D * x, D * y)
{
    Kernel::SpMV(m, n, d_vals, d_colinds, d_rowptrs, x, y);
}


template <typename Kernel,
          typename Matrix,
          typename D>
void SpMV_host(Matrix& A, D * x, D * y)
{
    auto spmv_params = Kernel::get_params(A);
    SpMV_wrapper<Kernel><<<spmv_params.first, spmv_params.second>>>
                    (A.get_rows(), A.get_cols(), 
                     A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                     x, y);
}


}
#endif
