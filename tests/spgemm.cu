#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusparse.h>

#include <cub/cub.cuh>

#include "gasp.h"
#include "test_common.h"

using namespace testing;
using namespace gasp;


int64_t cusparse_spgemm(cusparseHandle_t& handle,
                     cusparseSpMatDescr_t& matA,
                     cusparseSpMatDescr_t& matB,
                     cusparseSpMatDescr_t& matC)
{
    cusparseSpGEMMDescr_t spgemmDesc;
    CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));

    float alpha = 1.0;
    float beta = 0.0;

    size_t buf_size1 = 0;
    void * d_buf1 = nullptr;
    CUSPARSE_CHECK(
            cusparseSpGEMM_workEstimation(handle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                spgemmDesc, &buf_size1, nullptr));

    CUDA_CHECK(cudaMalloc((void**)&d_buf1, buf_size1));
    
    CUSPARSE_CHECK(
            cusparseSpGEMM_workEstimation(handle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                spgemmDesc, &buf_size1, d_buf1));

    size_t buf_size2 = 0;
    void * d_buf2 = nullptr;
    int64_t flops = 0;

    /*
    CUSPARSE_CHECK(cusparseSpGEMM_getNumProducts(spgemmDesc, &flops));

    size_t buf_size3 = 0;
    void * d_buf3 = nullptr;
    CUSPARSE_CHECK(
            cusparseSpGEMM_estimateMemory(handle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                spgemmDesc, 0.2,
                &buf_size3, nullptr, nullptr));

    CUDA_CHECK(cudaMalloc((void**)&d_buf3, buf_size3));

    CUSPARSE_CHECK(
            cusparseSpGEMM_estimateMemory(handle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                spgemmDesc, 0.2,
                &buf_size3, d_buf3, &buf_size2));

    CUDA_FREE_SAFE(d_buf3);
    */

    CUSPARSE_CHECK(
            cusparseSpGEMM_compute(handle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                spgemmDesc, &buf_size2, nullptr));

    CUDA_CHECK(cudaMalloc(&d_buf2, buf_size2));

    CUSPARSE_CHECK(
            cusparseSpGEMM_compute(handle, 
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matB, &beta, matC,
                CUDA_R_32F, CUSPARSE_SPGEMM_ALG1,
                spgemmDesc, &buf_size2, d_buf2));


    CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemmDesc));

    cudaFree(d_buf1); 
    cudaFree(d_buf2); 

                                            
    return flops;

}



int main(int argc, char ** argv)
{

    cusparseHandle_t cusparseHandle;
    CUSPARSE_CHECK(cusparseCreate(&cusparseHandle));

    const size_t n = std::atol(argv[1]);
    const size_t nnz = std::atol(argv[2]);
    const size_t n_iters = std::atoi(argv[3]);
    const std::string action(argv[4]);

    float * d_A_vals, * d_B_vals, * d_C_vals;
    int32_t * d_A_colinds, * d_A_rowptrs;
    int32_t * d_B_colinds, * d_B_rowptrs;
    int32_t * d_C_colinds, * d_C_rowptrs;

    init_sparse_mat_csr<float, RandomCsrInitializer>
                        (n, n, nnz, 
                        &d_A_vals, &d_A_colinds, &d_A_rowptrs, 
                        RandomCsrInitializer());

    init_sparse_mat_csr<float, RandomCsrInitializer>
                        (n, n, nnz, 
                        &d_B_vals, &d_B_colinds, &d_B_rowptrs, 
                        RandomCsrInitializer());

    init_sparse_mat_csr<float, RandomCsrInitializer>
                        (n, n, 1, 
                        &d_C_vals, &d_C_colinds, &d_C_rowptrs, 
                        RandomCsrInitializer());


    cusparseSpMatDescr_t A;
    cusparseSpMatDescr_t B;
    cusparseSpMatDescr_t C;

    make_cusparse_csr(n, n, nnz, d_A_vals, d_A_colinds, d_A_rowptrs, &A);
    make_cusparse_csr(n, n, nnz, d_B_vals, d_B_colinds, d_B_rowptrs, &B);
    make_cusparse_csr(n, n, nnz, d_C_vals, d_C_colinds, d_C_rowptrs, &C);

    GaspCsr gasp_A(n, n, nnz, d_A_vals, d_A_colinds, d_A_rowptrs);
    GaspCsr gasp_B(n, n, nnz, d_B_vals, d_B_colinds, d_B_rowptrs);
    GaspCsr gasp_C(n, n, nnz, d_C_vals, d_C_colinds, d_C_rowptrs);

    std::ofstream A_out("A.out");
    gasp_A.dump(A_out);
    A_out.close();

    const char * label_cusparse = "SpGEMM_cusparse";

    start_timer(label_cusparse);

    int64_t flops = 0;
    for (int i=0; i<n_iters; i++) {
        flops = cusparse_spgemm(cusparseHandle, A, B, C);
    }

    end_timer(label_cusparse);
    measure_gflops(label_cusparse, flops);

    print_time(label_cusparse);
    print_gflops(label_cusparse);

    const char * label_gasp = "SpGEMM_gasp";

    start_timer(label_gasp);

    for (int i=0; i<n_iters; i++) {
        SpGEMM_ESC<SpGEMM_ESC_Warp<PlusTimesSemiring<float>>,
                    OutputTuple<float, int64_t>>(gasp_A, gasp_B);
    }

    end_timer(label_gasp);
    print_time(label_gasp);

    CUSPARSE_CHECK(cusparseDestroySpMat(A));
    CUSPARSE_CHECK(cusparseDestroySpMat(B));
    CUSPARSE_CHECK(cusparseDestroySpMat(C));

    CUSPARSE_CHECK(cusparseDestroy(cusparseHandle));

    return 0;
}
