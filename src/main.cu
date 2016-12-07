#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
int main()
{
        double p_diag = 0.8;
        double p_nondiag = 0.05;
        int N = 16;

        // Create sparse matix A
        SpMatrix A = generateSquareSpMatrix(N, p_diag, p_nondiag);
        printf("Sparse Matrix A: \n");
        printSpMatrix(A);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Generate dense vector x
        float* x = (float *)malloc(sizeof(float)*N);
        fillDenseVector(x, N);
        printf("Dense vector x: "); printArray(x, N);

        // Define output vector y
        float* y = (float *)malloc(sizeof(float)*N);

        // Compute spmv multiplication
        cudaEventRecord(start);
        cpuSpMV(y, A, x);
        cudaEventRecord(stop);

        // Print result

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Output vector y: "); printArray(y, N);
        printf("Elapsed time (ms): %f\n", milliseconds);


        // Free memory
        free(A.IA);
        free(A.JA);
        free(A.A);
        free(x);
	return 0;
}
