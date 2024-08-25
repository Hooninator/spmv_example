#ifndef GASP_SPGEMM_CUH
#define GASP_SPGEMM_CUH

#include "Semirings.cuh"
#include "common.h"

namespace gasp {

template <typename SR>
struct SpGEMMWarp {

    template <typename D, typename I>
    static __device__ void SpGEMM_expand(const I m, const I n, const I k)
    {
    }
                                  

};

template <typename D, typename I>
__global__ void SpGEMM_count_flops(const size_t m, const size_t n, const size_t k,
                                  const D * d_A_vals, 
                                  const I * d_A_colinds, const I * d_A_rowptrs,
                                  const D * d_B_vals, 
                                  const I * d_B_colinds, const I * d_B_rowptrs,
                                  size_t * d_flops_rows)
{
    using WarpReduce = cub::WarpReduce<size_t>;
    __shared__ typename WarpReduce::TempStorage temp_storage;

    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t wid = tid / 32;
    const size_t lid = tid % 32;

    if (wid >= m) return;

    const I row_start = d_A_rowptrs[wid];
    const I row_end = d_A_rowptrs[wid];

    size_t thread_data = 0;

#pragma unroll
    for (I j = row_start + lid; j < row_end; j += 32)
    {
        const I colidx = d_A_colinds[j];
        const size_t nnz_row = d_B_rowptrs[colidx+1] - d_B_rowptrs[colidx];
        thread_data += nnz_row;
    }

    size_t result = WarpReduce(temp_storage).Sum(thread_data);

    if (lid==0) {
        d_flops_rows[wid] = result;
    }

}

struct MemEstimatorFLOPS
{

    template <typename Matrix>
    static size_t get_output_mem(Matrix& A, Matrix& B, size_t * d_flops_rows)
    {
        const size_t tpb = 256;
        const size_t wpb = tpb / 32;
        const size_t blocks = std::ceil( (double)A.get_rows() / (double)wpb );

        SpGEMM_count_flops<<<blocks, tpb>>>(A.get_rows(), B.get_cols(), A.get_cols(),
                                            A.get_vals(), A.get_colinds(), A.get_rowptrs(),
                                            B.get_vals(), B.get_colinds(), B.get_rowptrs(),
                                            d_flops_rows);
        CUDA_CHECK(cudaDeviceSynchronize());


        thrust::device_ptr<size_t> d_flops_rows_ptr(d_flops_rows);

        size_t total_flops = thrust::reduce(d_flops_rows_ptr, d_flops_rows_ptr+A.get_rows(), 0);

        return total_flops * A.get_tuple_bytes(); 
    }


};


template <typename D, typename I>
struct OutputTuple
{
    D val;
    I i;
    I j;
};


template <typename Kernel, typename Tuple, typename Matrix, 
            typename MemEstimator=MemEstimatorFLOPS>
void SpGEMM_host(Matrix& A, Matrix& B)
{
    /* First, estimate memory for intermediate products */ 
    size_t * d_flops_rows;
    CUDA_CHECK(cudaMalloc(&d_flops_rows, sizeof(size_t)*A.get_rows()));
    size_t output_mem = MemEstimator::get_output_mem(A, B, d_flops_rows);

    /* Allocate memory for intermediate products */
    Tuple * d_intermediate_products;
    CUDA_CHECK(cudaMalloc(&d_intermediate_products, output_mem));


    CUDA_FREE_SAFE(d_flops_rows);
    CUDA_FREE_SAFE(d_intermediate_products);
}

}
#endif
