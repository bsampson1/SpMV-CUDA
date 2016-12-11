# SpMV-CUDA

## CPEG 655 Course Project

### About
This repository contains a sparse matrix-vector (SpMV) multiplication
that has been optimized in CUDA for the GPU as a submission for
the course project in CPEG655: High Performance Computing with
Commodity Hardware.

Project Features:
        - Sparse matrix generation up to 2^15x2^15
        - Several SpMV cuda kernel "flavors"
        - CPU SpMV implementation
        - Tests for sparse matrix generation, timing, RMSE
          calculation, and SpMV correctness

### Compiling and Executing Code
Suggested that you compile and run using the Makefile. You
need to have the nvcc compiler to run this code and a 
CUDA-enabled device to run it on.

#### Compilation Options:
1. make (compiles main.cu)
2. make all-tests (compiles options 3-6)
3. make test-spmatrix-generator
4. make test-RMSE
5. make test-spmv-correctness
6. make test-spmv-timing

#### Execution Options:
1. make run (runs main)
2. make run-all-tests (runs options 3-6)
3. make run-test-spmatrix-generator
4. make run-test-RMSE
5. make run-test-spmv-correctness
6. make run-test-spmv-timing

#### Clean Options:
1. make clean (cleans all executables from project)

### Authors
Ben Sampson & Mecheal Greene


