#include "spmv.h"
#include <stdio.h>

int main()
{
        double p_diag = 0.9;
        double p_nondiag = 0.001;
        float *A;
        int *IA, *JA;
        int NNZ;

        // seed random number generator
        time_t t; srand((unsigned) time(&t));
        
        int N;
        for (N = 2; N <= (1 << 15); N*=2)
        {
                // allocates and changes A, IA, JA, & NNZ!
                generateSquareSpMatrix(&A, &IA, &JA, &NNZ, N, p_diag, p_nondiag);
                //printSpMatrix(A, IA, JA, N, N);

                printf("Generated sparse matrix of size %ix%i\n", N, N);

                // Free memory
                free(A);
                free(IA);
                free(JA);
        }

        return 0;
}
