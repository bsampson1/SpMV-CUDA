# sparse-matrix-vector-multiplication
CPEG 655 Course Project
Ben Sampson & Mecheal Greene

About
--------------------------------------------------------------

This repository contains a sparse matrix-vector multiplication
that has been optimized in CUDA for the GPU as a submission for
the course project in CPEG655: High Performance Computing with
Commodity Hardware.

Compiling and Executing Code
---------------------------------------------------------------

Type the following commands into terminal:
    1. make
    2. make run

Description of Files Within This Repository
---------------------------------------------------------------

main.cu - contains main function which is executed when run
spmv.c - contains implementations for SpMV functions & structs
spmv.h - contains declarations for SpMV functions & structs
