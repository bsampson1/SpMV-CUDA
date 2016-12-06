#include "spmv.h"
#include <stdio.h>

int main()
{
        double p_diag = 0.8;
        double p_nondiag = 0.05;
        int N = 16;

        // Create sparse matix A
        SpMatrix A = generateSquareSpMatrix(N, p_diag, p_nondiag);
        printf("Sparse Matrix A: \n");
        printSpMatrix(A);

        // Generate dense vector x
        float* x = (float *)malloc(sizeof(float)*N);
        fillDenseVector(x, N);
        printf("Dense vector x: "); printArray(x, N);

        // Define output vector y
        float* y = (float *)malloc(sizeof(float)*N);

        // Compute spmv multiplication
        cpuSpMV(y, A, x);

        // Print result
        printf("Output vector y: "); printArray(y, N);
        

        // Free memory
        free(A.IA);
        free(A.JA);
        free(A.A);
        free(x);
	return 0;
}
