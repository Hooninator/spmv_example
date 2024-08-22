

#include <chrono>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusparse.h>

#include <cub/cub.cuh>

#include "test_common.h"

using namespace testing;


int main(int argc, char ** argv)
{
    cusparseHandle_t cusparseHandle;
    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    size_t m = std::atol(argv[1]);
    size_t n = std::atol(argv[2]);
    size_t nnz = std::atol(argv[3]);
    size_t n_iters = std::atoi(argv[4]);

    float * d_vals, * d_vec, * d_result;
    uint32_t * d_colinds, * d_rowptrs;

    init_sparse_mat_csr<float, RandomCsrInitializer>
                        (m, n, nnz, 
                        &d_vals, &d_colinds, &d_rowptrs, 
                        RandomCsrInitializer());
    init_dense_vec(n, &d_vec);
    init_dense_vec(m, &d_result);

    cusparseDnVecDescr_t x;
    cusparseDnVecDescr_t y;
    cusparseSpMatDescr_t A;

    CUSPARSE_CHECK(cusparseCreateDnVec(&x, n, d_vec, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&y, m, d_result, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(&A,
                                     m, n, nnz,
                                     d_rowptrs,
                                     d_colinds,
                                     d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    float alpha = 1.0;
    float beta = 0.0;

    size_t buf_size;
    void * buf;

    const char * label_cusparse = "SpMV_cusparse";

    start_timer(label_cusparse);
    for (int i=0; i<n_iters; i++) {

        /* BENCHMARK CUSPARSE */
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(cusparseHandle,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, A, x,
                                                &beta, y,
                                                CUDA_R_32F,
                                                CUSPARSE_SPMV_ALG_DEFAULT,
                                                &buf_size));

        CUDA_CHECK(cudaMalloc(&buf, buf_size));
        CUSPARSE_CHECK(cusparseSpMV(cusparseHandle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, A, x,
                                    &beta, y,
                                    CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT,
                                    buf));



        CUDA_CHECK(cudaFree(buf));
    }

    end_timer(label_cusparse);
    measure_gflops(label_cusparse, 2*nnz*n_iters);

    print_time(label_cusparse);
    print_gflops(label_cusparse);

    CUSPARSE_CHECK(cusparseDestroySpMat(A));
    CUSPARSE_CHECK(cusparseDestroyDnVec(x));
    CUSPARSE_CHECK(cusparseDestroyDnVec(y));


    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_colinds));
    CUDA_CHECK(cudaFree(d_rowptrs));

    CUDA_CHECK(cudaFree(d_vec));
    CUDA_CHECK(cudaFree(d_result));

    CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));


    return 0;
}
