cmake_minimum_required(VERSION 3.3)
project(SpMV VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_compile_options(-w -O3 --use_fast_math)
set(CMAKE_CUDA_ARCHITECTURES "80")

find_package(CUDAToolkit)

add_executable(spmv tests/spmv.cu)

target_include_directories(spmv PUBLIC include)

target_link_libraries(spmv PUBLIC CUDA::cusparse CUDA::cudart CUDA::cublas )

