# SpMV Exercise #
This repository contains infrastructure for implementing and benchmarking a SpMV kernel with CUDA.

## Build Instructions ##
To build the driver program, run the following

`mkdir build && cd build`

`cmake ..`

`make spmv`

This will generate a binary called `spmv`, which can be run with a command like this

`./spmv /path/to/matrix`

where `/path/to/matrix` is a relative path to a matrix stored in matrix market (`.mtx`) format.

Once you have built the executable, please run `sh get_matrices.sh`, which will fetch two large test matrices from the [SuiteSparse matrix collection](https://sparse.tamu.edu/) and place them in the `matrices` directory.
The `matrices` directory also contains a small matrix called `n16.mtx`, which can be used for debugging.

The `spmv` executable does two things.
First, it will run a correctness check that compares the output of your SpMV kernel to the output of the SpMV kernel found in the [cuSPARSE library](https://docs.nvidia.com/cuda/cusparse/contents.html). 
Second, it runs a simple benchmark that compares the throughput of your SpMV kernel to the throughput of cuSPARSE's SpMV kernel.

## Important Files ## 
The main file that you'll need to work with is `include/spmv.cuh`. This file contains the template SpMV kernel that you'll need to implement.

The driver program can be found in `tests/spmv.cu`, although you should not have to edit this file at all.

## References ## 
* Bell, Nathan, and Michael Garland. "Implementing sparse matrix-vector multiplication on throughput-oriented processors." Proceedings of the conference on high performance computing networking, storage and analysis. 2009. [https://doi.org/10.1145/1654059.1654078](https://doi.org/10.1145/1654059.1654078)

* [cuSPARSE SpMV API](https://docs.nvidia.com/cuda/cusparse/#cusparsespmv)
