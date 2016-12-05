#include "spmv.h"
#include <stdio.h>

int main()
{

        
        // Initialize sparse matrix in CSR format
        int M = 5;
        int N = 5;
        int NNZ = 9;

        float* A;
        int* IA; 
        int* JA;

        A = (float *)malloc(sizeof(float)*NNZ);
        IA = (int *)malloc(sizeof(int)*(M+1));
        JA = (int *)malloc(sizeof(int)*NNZ);

        int i;
        for (i = 0; i < NNZ; ++i)
        {
                A[i] = i+1;
                JA[i] = i%N;
        }

        IA[0] = 0; IA[1] = 2; IA[2] = 4; IA[3] = 5;
        IA[4] = 7; IA[5] = 9;
        
        // Initialie sparse matrix on stack pointing to CSR arrays
        SpMatrix S = {.M = M, .N = N, .NNZ = NNZ, .A = A, .IA = IA, .JA = JA};

        // Print arrays for CSR sparse matrix
        printf("A: "); printArray(A, NNZ);
        printf("IA: "); printArray(IA, M+1);
        printf("JA: "); printArray(JA, NNZ);

        // Print corresponding sparse matrix
        printf("Sparse Matrix S:\n");
        printSpMatrix(S);

        // Free memory
        free(A);
        free(IA);
        free(JA);
        return 0;
}
