cmake_minimum_required(VERSION 3.3)
project(GASP VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_compile_options(-w -O3)
set(CMAKE_CUDA_ARCHITECTURES "80")

find_package(CUDA)

add_executable(spmv tests/spmv.cu)
add_executable(spgemm tests/spgemm.cu)
add_executable(spmm tests/spmm.cu)

target_link_libraries(spmv PUBLIC "-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/" -lcuda -lcudart -lcusparse -lcublas)
target_link_libraries(spmm PUBLIC "-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/" -lcuda -lcudart -lcusparse -lcublas)
target_link_libraries(spgemm PUBLIC "-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64/" -lcuda -lcudart -lcusparse -lcublas)

