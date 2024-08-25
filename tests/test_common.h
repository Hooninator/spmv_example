
#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <map>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <random>
#include <cassert>


#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusparse_v2.h>

#include <cub/cub.cuh>

#include "colors.h"
#include "utils.cuh"


#define CUDA_CHECK(call) {                                                 \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " ("     \
                  << __FILE__ << ":" << __LINE__ << ")" << std::endl;      \
        exit(err);                                                         \
    }                                                                      \
}

#define CUSPARSE_CHECK(call) do {                                    \
    cusparseStatus_t err = call;                                     \
    if (err != CUSPARSE_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuSPARSE error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cusparseGetErrorString(err));    \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

#define EPS 1

namespace testing {
/***************************************************************************
 *                                                                         *
 *                           CORRECTNESS TEST UTILS                        *
 *                                                                         *
 ***************************************************************************/


/***************************************************************************
 *                                                                         *
 *                           MATRIX AND VECTOR INITIALIZATION              *
 *                                                                         *
 ***************************************************************************/
std::random_device rd;
std::mt19937 gen(rd());

template <typename T>
void init_random_buffer(std::vector<T>& buf,
                        const T lower, 
                        const T upper)
{
    std::uniform_real_distribution<T> distr(lower, upper);
    std::generate(buf.begin(), buf.end(), [&]()
    {
        T elem = distr(gen);
        return elem;
    }); 

}

void init_random_buffer(std::vector<int64_t>& buf,
                        const int64_t lower, 
                        const int64_t upper)
{
    std::uniform_int_distribution<uint64_t> distr(lower, upper);
    std::generate(buf.begin(), buf.end(), [&](){return distr(gen);}); 
}

void init_random_buffer(std::vector<int32_t>& buf,
                        const int32_t lower, 
                        const int32_t upper)
{
    std::uniform_int_distribution<uint32_t> distr(lower, upper);
    std::generate(buf.begin(), buf.end(), [&](){return distr(gen);}); 
}


template <typename T>
void init_dense_mat(const size_t m,
					const size_t n,
                    T ** d_vals)
{
    std::vector<T> h_vals(m*n);
    init_random_buffer(h_vals, -1e5, 1e5);

    CUDA_CHECK(cudaMalloc(d_vals, sizeof(T)*m*n));
    CUDA_CHECK(cudaMemcpy(*d_vals, h_vals.data(), sizeof(T)*m*n, cudaMemcpyHostToDevice));
}


template <typename T>
void init_dense_vec(const size_t n,
                    T ** d_vals)
{
    std::vector<T> h_vals(n);
    init_random_buffer(h_vals, static_cast<T>(-1e5), static_cast<T>(1e5));

    CUDA_CHECK(cudaMalloc(d_vals, sizeof(T)*n));
    CUDA_CHECK(cudaMemcpy(*d_vals, h_vals.data(), sizeof(T)*n, cudaMemcpyHostToDevice));
}


struct RandomCsrInitializer
{
    template <typename T, typename I>
    void init(const size_t m,
             const size_t n,
             const size_t nnz,
             std::vector<T>& vals,
             std::vector<I>& colinds,
             std::vector<I>& rowptrs)
    {

        std::unordered_set<I> found;
        found.reserve(m);

        std::uniform_int_distribution<I> distr(1, n);
        std::uniform_real_distribution<T> real_distr(-5, 5);

		// Distribute nnz elements randomly among the rows
		for (int i = 0; i < nnz; ++i) {
			I row = distr(gen) % m;  // Random row index
			rowptrs[row + 1]++;       // Increment the count of non-zeros in this row
		}

		for (int i = 0; i < m; ++i) {
			rowptrs[i + 1] += rowptrs[i];
		}
		
		// Fill the column indices and values randomly
		for (int i = 0; i < m; ++i) {
			I rowStart = rowptrs[i];
			I rowEnd = rowptrs[i + 1];
			I numElementsInRow = rowEnd - rowStart;

			// Generate unique random column indices
			std::vector<I> columns;
			while (columns.size() < numElementsInRow) {
				I col = distr(gen) % n;  // Random column index
				if (std::find(columns.begin(), columns.end(), col) == columns.end()) {
					columns.push_back(col);
				}
			}

			// Sort the column indices for CSR format
			std::sort(columns.begin(), columns.end());

			// Fill in the column indices and values
			for (I j = 0; j < numElementsInRow; ++j) {
				colinds[rowStart + j] = columns[j];
				vals[rowStart + j] = static_cast<T>(real_distr(gen));			
            }
		}
    }

};



template <typename T, typename Initializer, typename I>
void init_sparse_mat_csr(const size_t m,
                         const size_t n,
                         const size_t nnz,
                         T ** d_vals,
                         I ** d_colinds,
                         I ** d_rowptrs,
                         Initializer initializer)
{
    std::vector<T> h_vals(nnz);
    std::vector<I> h_colinds(nnz);
    std::vector<I> h_rowptrs(m+1);

    initializer.init(m, n, nnz, h_vals, h_colinds, h_rowptrs);

    CUDA_CHECK(cudaMalloc(d_vals, sizeof(T)*nnz));
    CUDA_CHECK(cudaMalloc(d_colinds, sizeof(I)*nnz));
    CUDA_CHECK(cudaMalloc(d_rowptrs, sizeof(I)*(m+1)));

    CUDA_CHECK(cudaMemcpy(*d_vals, h_vals.data(), sizeof(T)*nnz,
                                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_colinds, h_colinds.data(), sizeof(I)*nnz,
                                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_rowptrs, h_rowptrs.data(), sizeof(I)*(m+1),
                                cudaMemcpyHostToDevice));
}


template <typename D, typename I>
void make_cusparse_csr(const size_t m,
                         const size_t n,
                         const size_t nnz,
                         D * d_vals,
                         I * d_colinds,
                         I * d_rowptrs,
                         cusparseSpMatDescr_t * A)
{
    CUSPARSE_CHECK(cusparseCreateCsr(A,
                                     m, n, nnz,
                                     d_rowptrs,
                                     d_colinds,
                                     d_vals,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F));
}



/***************************************************************************
 *                                                                         *
 *                           PROFILING                                     *
 *                                                                         *
 ***************************************************************************/

std::map<std::string, cudaEvent_t> stimes;
std::map<std::string, cudaEvent_t> etimes;

std::map<std::string, size_t> gflops;
std::map<std::string, float> times;

void start_timer(const char * label)
{
    std::string label_str(label);
    cudaEvent_t start, end;

    stimes[label_str] = start;
    etimes[label_str] = end;

    cudaEventCreate(&stimes[label_str]);
    cudaEventCreate(&etimes[label_str]);

    cudaEventRecord(stimes[label_str]);
}


void end_timer(const char * label)
{
    std::string label_str(label);

    cudaEventRecord(etimes[label_str]);
    cudaEventSynchronize(etimes[label_str]);

    float t = 0;
    cudaEventElapsedTime(&t, stimes[label_str], etimes[label_str]);

    times[label] = t;

    cudaEventDestroy(stimes[label_str]);
    cudaEventDestroy(etimes[label_str]);
}


void measure_gflops(const char * label, size_t flops)
{
    std::string label_str(label);
    size_t gflops_per_sec = (flops / 1e9) / (times[label] / 1e3);
    gflops[label] = gflops_per_sec;
}


void print_time(const char * label)
{
    std::string label_str(label);
    std::cout<<BRIGHT_RED<<"["<<label_str<<"]: "<<times[label_str]/1e3<<"s"<<RESET<<std::endl;
}


void print_gflops(const char * label)
{
    std::string label_str(label);
    std::cout<<BRIGHT_BLUE<<"["<<label_str<<"]: "<<gflops[label_str]<<" gflops/s"<<RESET<<std::endl;
}


}

#endif
