
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
#include <cusparse.h>

#include <cub/cub.cuh>

#include "colors.h"


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

#define EPS 1e-3

namespace testing {


/***************************************************************************
 *                                                                         *
 *                           MATRIX AND VECTOR INITIALIZATION              *
 *                                                                         *
 ***************************************************************************/
std::random_device rd;
std::mt19937 gen(rd());

template <typename T>
void init_random_buffer(std::vector<T>& buf,
                        const int32_t lower, 
                        const int32_t upper)
{
    std::uniform_real_distribution<T> distr(lower, upper);
    std::generate(buf.begin(), buf.end(), [&](){return distr(gen);}); 
}

void init_random_buffer(std::vector<uint32_t>& buf,
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
    init_random_buffer(h_vals, -1e5, 1e5);

    CUDA_CHECK(cudaMalloc(d_vals, sizeof(T)*n));
    CUDA_CHECK(cudaMemcpy(*d_vals, h_vals.data(), sizeof(T)*n, cudaMemcpyHostToDevice));
}


struct RandomCsrInitializer
{
    template <typename T>
    void init(const size_t m,
             const size_t n,
             const size_t nnz,
             std::vector<T>& vals,
             std::vector<uint32_t>& colinds,
             std::vector<uint32_t>& rowptrs)
    {
        init_random_buffer(vals, -1e5, 1e5);
        init_random_buffer(colinds, 0, n);

        std::unordered_set<uint32_t> found;
        found.reserve(m);

        std::uniform_int_distribution<uint32_t> distr(1, n);

        rowptrs[0] = 0;
        size_t count = 1;
        while (count < m) {
            uint32_t idx = distr(gen);
            if (found.find(idx)==found.end()) {
                found.insert(idx);
                rowptrs[count] = idx;
                count++;
            }
        }

        std::sort(rowptrs.begin(), rowptrs.end());
    }

};



template <typename T, typename Initializer>
void init_sparse_mat_csr(const size_t m,
                         const size_t n,
                         const size_t nnz,
                         T ** d_vals,
                         uint32_t ** d_colinds,
                         uint32_t ** d_rowptrs,
                         Initializer initializer)
{
    std::vector<T> h_vals(nnz);
    std::vector<uint32_t> h_colinds(nnz);
    std::vector<uint32_t> h_rowptrs(m+1);

    initializer.init(m, n, nnz, h_vals, h_colinds, h_rowptrs);

    CUDA_CHECK(cudaMalloc(d_vals, sizeof(T)*nnz));
    CUDA_CHECK(cudaMalloc(d_colinds, sizeof(T)*nnz));
    CUDA_CHECK(cudaMalloc(d_rowptrs, sizeof(T)*(m+1)));

    CUDA_CHECK(cudaMemcpy(*d_vals, h_vals.data(), sizeof(T)*nnz,
                                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_colinds, h_colinds.data(), sizeof(uint32_t)*nnz,
                                cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*d_rowptrs, h_rowptrs.data(), sizeof(uint32_t)*(m+1),
                                cudaMemcpyHostToDevice));
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
