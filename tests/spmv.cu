

#include <chrono>
#include <fstream>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusparse.h>

#include <cub/cub.cuh>

#include "gasp.h"
#include "test_common.h"


using namespace testing;
using namespace gasp;

template <typename SR, typename D, typename I>
void check_correctness(const size_t m,
                        const size_t n,
                        const size_t nnz,
                        cusparseSpMatDescr_t& A,
                        cusparseDnVecDescr_t& x, 
                        cusparseDnVecDescr_t& y, 
                        GaspCsr<D,I>& gasp_A,
                        D * d_x, D * d_y)
{
    cusparseHandle_t cusparseHandle;
    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    const float alpha = 1.0;
    const float beta = 0.0;

    size_t buf_size;
    void * buf;

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

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(buf));
    D * h_correct = new D[m];
    CUDA_CHECK(cudaMemcpy(h_correct, d_y, sizeof(D)*m, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_y, 0, sizeof(D)*m));

    SpMV_host<PlusTimesSemiring<float>, float, uint32_t>(gasp_A, d_x, d_y);
    CUDA_CHECK(cudaDeviceSynchronize());

    D * h_computed = new D[m];
    CUDA_CHECK(cudaMemcpy(h_computed, d_y, sizeof(D)*m, cudaMemcpyDeviceToHost));

    for (int i=0; i<m; i++) {
        assert(fabs((h_computed[i] - h_correct[i])) < EPS);
    }
    std::cout<<BRIGHT_GREEN<<"Correctness for SpMV passed!"<<RESET<<std::endl;

    CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));

    delete[] h_correct;
    delete[] h_computed;
}

int main(int argc, char ** argv)
{
    cusparseHandle_t cusparseHandle;
    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    const size_t m = std::atol(argv[1]);
    const size_t n = std::atol(argv[2]);
    const size_t nnz = std::atol(argv[3]);
    const size_t n_iters = std::atoi(argv[4]);
    const std::string action(argv[5]);

    float * d_vals, * d_x, * d_y;
    uint32_t * d_colinds, * d_rowptrs;

    init_sparse_mat_csr<float, RandomCsrInitializer>
                        (m, n, nnz, 
                        &d_vals, &d_colinds, &d_rowptrs, 
                        RandomCsrInitializer());
    init_dense_vec(n, &d_x);
    init_dense_vec(m, &d_y);

    cusparseDnVecDescr_t x;
    cusparseDnVecDescr_t y;
    cusparseSpMatDescr_t A;

    CUSPARSE_CHECK(cusparseCreateDnVec(&x, n, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&y, m, d_y, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateCsr(&A,
                                     m, n, nnz,
                                     d_rowptrs,
                                     d_colinds,
                                     d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));

    GaspCsr gasp_A(m, n, nnz,
                   d_vals, d_colinds, d_rowptrs);

    if (action.compare("correctness")==0) { 
        check_correctness<PlusTimesSemiring<float>, float, uint32_t>
            (m, n, nnz, A, x, y, gasp_A, d_x, d_y);
    } else if (action.compare("benchmark")==0) {
        const float alpha = 1.0;
        const float beta = 0.0;

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

            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaFree(buf));
        }

        end_timer(label_cusparse);
        measure_gflops(label_cusparse, 2*nnz*n_iters);

        print_time(label_cusparse);
        print_gflops(label_cusparse);


        const char * label_gasp = "SpMV_gasp";

        start_timer(label_gasp);
        for (int i=0; i<n_iters; i++) {
            SpMV_host<PlusTimesSemiring<float>, float, uint32_t>(gasp_A, d_x, d_y);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        end_timer(label_gasp);
        measure_gflops(label_gasp, 2*nnz*n_iters);

        print_time(label_gasp);
        print_gflops(label_gasp);
    }

    CUSPARSE_CHECK(cusparseDestroySpMat(A));
    CUSPARSE_CHECK(cusparseDestroyDnVec(x));
    CUSPARSE_CHECK(cusparseDestroyDnVec(y));

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));


    return 0;
}
