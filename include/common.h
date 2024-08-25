#ifndef COMMON_N
#define COMMON_H

#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <map>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <random>


#include <cuda.h>
#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusparse.h>

#include <cub/cub.cuh>

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

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

#define CUDA_FREE_SAFE(ptr) if (ptr!=nullptr) CUDA_CHECK(cudaFree(ptr));

namespace gasp {

typedef unsigned long long int ull_t;

}
#endif
