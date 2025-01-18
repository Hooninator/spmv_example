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
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid>=m) return;

	const I row_start = d_rowptrs[tid];
	const I row_end = d_rowptrs[tid+1];

	D thread_data = 0;

#pragma unroll
	for (I j = row_start; j < row_end; j++) 
	{
		const I colidx = d_colinds[j];
		const D r = (d_vals[j] * d_x[colidx]);
		thread_data = thread_data + r;
	}

	d_y[tid] = d_y[tid] + thread_data;
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
