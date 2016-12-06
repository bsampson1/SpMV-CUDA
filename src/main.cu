#include "spmv.h"
#include <stdio.h>

int main()
{
        double p_diag = 0.8;
        double p_nondiag = 0.05;
        int N = 8;

        SpMatrix S = generateSquareSpMatrix(N, p_diag, p_nondiag);
        printf("Sparse Matrix S: \n");
        printSpMatrix(S);

        float* v = (float *)malloc(sizeof(float)*N);
        fillDenseVector(v, N);

        // Free memory
        free(S.IA);
        free(S.JA);
        free(S.A);

	return 0;
}
