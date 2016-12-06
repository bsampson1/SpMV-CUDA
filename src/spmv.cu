#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>

void cpuSpMV(Vector* y, const SpMatrix* A, const Vector* x)
{
    int i, j, y, x;
    for (i=0; i<S.M; ++i)
    {
        for (j=IA[i]; j<IA[i+1]; ++j)
        {
            y[i] += values[j]*x[JA[j]];
        }
    }
}

bool areEqual(const Vector*a, const Vector*b)
{
    for (i = 0; i<row1; ++i)
    {
        for (j=0; j<column2; ++j)
        {
            if (a[i][j] != b[i][j])
            {
                pass;
            }
        }
    }
}
bool areEqual(const SpMatrix*A, const SpMatrix*B)
{
    for (i = 0; i<row1; ++i)
    {
        for (j=0; j<column2; ++j)
        {
            if (a[i][j] != b[i][j])
            {
                pass;
            }
        }
    }
}
void printArray(const float* arr, const int l)
{
        int i;
        printf("[ ");
        for (i = 0; i < l; ++i)
                printf("%g ", arr[i]);
        printf("]\n");
}

void printArray(const int* arr, const int l)
{
        int i;
        printf("[ ");
        for (i = 0; i < l; ++i)
                printf("%i ", arr[i]);
        printf("]\n");
}

void printSpMatrix(const SpMatrix S)
{
        int v = 0;

        int i, j;
        for (i = 0; i < S.M; ++i)
        {
                for (j = 0; j < S.N; ++j)
                {
                        if(v < S.IA[i+1] && j == S.JA[v])
                        {
                                printf("%g\t", S.A[v]);
                                v++;
                        }
                        else
                        {
                                printf("0\t");
                        }
                }
                printf("\n");
        }
}

SpMatrix generateSquareSpMatrix(const int N, const double p_diag, const double p_nondiag)
{
        // estimate size of A, JA arrays because they vary between realization
        // but are same for a given realization
        int estSize = N*p_diag + N*(N-1)*p_nondiag;

        //printf("Estimate size of A: %i\n", estSize);
        //printf("Size of IA: %i\n", N+1);
        //printf("Estimate size of JA: %i\n", estSize);

        // allocate IA because size is fixed (size of IA = N + 1)
        int* IA = (int *)malloc(sizeof(int)*(N+1));

        // define buffer space for undetermined arrays
        const int bufferSize = 1.5*estSize;

        //printf("Buffer size = %i\n", bufferSize);

        // allocate buffer*estSize for A & JA so we can probably fit everything in those
        float* A_temp = (float *)malloc(sizeof(float)*bufferSize);
        int* JA_temp = (int *)malloc(sizeof(float)*bufferSize);

        // seed random number generator
        time_t t; double randProb; double randNum;
        srand((unsigned) time(&t));

        // Create NNZ variable for SpMatrix struct
        int NNZ = 0;

        SpMatrix S = {.M = N, .N = N, .NNZ = NNZ, .A = NULL, .IA = NULL, .JA = NULL};

        // Initial condition: first element of IA[1] = 0
        IA[1] = 0;

        // Iterate through all elements placing nonzero terms in sparse matrix
        int i,j;
        for (i = 0; i < N; ++i)
        {
                IA[i+1] = IA[i];

                for (j = 0; j < N; ++j)
                {
                        randProb = ((double)rand())/RAND_MAX;
                        if (i == j) //on diagonal - use p_diag
                        {
                                if (randProb < p_diag)
                                {
                                        if (NNZ == bufferSize) // EXCEEDED ALLOWED BUFFER
                                        {
                                                printf("Error: Exceeded allowed buffer size. Failed to create sparse matrix\n");
                                                return S;
                                        }
                                        
                                        // Add new random number to sparse matrix
                                        randNum = (double)(rand()%10)+1;
                                        A_temp[NNZ] = randNum;
                                        JA_temp[NNZ] = j;
                                        IA[i+1]++;
                                        NNZ++;
                                }
                        }
                        else // not on diagoal - use p_nondiag
                        {
                                if (randProb < p_nondiag)
                                {
                                        if (NNZ == bufferSize) // EXCEEDED ALLOWED BUFFER
                                        {
                                                printf("Error: Exceeded allowed buffer size. Failed to create sparse matrix\n");
                                                return S;
                                        }
                                        
                                        // Add new random number to sparse matrix
                                        randNum = (double)(rand()%10)+1;
                                        A_temp[NNZ] = randNum;
                                        JA_temp[NNZ] = j;
                                        IA[i+1]++;
                                        NNZ++;  
                                }
                        }
                }
        }

        //printf("A_temp: "); printArray(A_temp, bufferSize);
        //printf("IA: "); printArray(IA, N+1);
        //printf("JA_temp: "); printArray(JA_temp, bufferSize);

        // By this point we have not exceeded memory limit so lets create
        // actual A and IA array now that we have determined the size
        float* A = (float *)malloc(sizeof(float)*NNZ);
        int* JA = (int *)malloc(sizeof(float)*NNZ);

        // Add elements from temp arrays to actual arrays
        for (i = 0; i < NNZ; ++i)
        {
                A[i] = A_temp[i];
                JA[i] = JA_temp[i];
        }

        //printf("A: "); printArray(A, NNZ);
        //printf("IA: "); printArray(IA, N+1);
        //printf("JA: "); printArray(JA, NNZ);
        //printf("NNZ: %i/n", NNZ);

        // Create SpMatrix struct
        S = (SpMatrix) {.M = N, .N = N, .NNZ = NNZ, .A = A, .IA = IA, .JA = JA};
        //printf("Successfully created sparse matrix\n");
        //printf("Sparse Matrix S:\n");
        //printSpMatrix(S);

        // set excess pointers to NULL
        A = NULL;
        A_temp = NULL;
        IA = NULL;
        JA = NULL;
        
        // return SpMatrix
        return S;
}

void fillDenseVector(float* v, const int N)
{
        int i;
        for (i = 0; i < N; ++i)
                v[i] = (float)(rand()%10) - 5;
}










