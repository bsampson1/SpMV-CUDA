#include "spmv.h"
#include <stdio.h>

void printSpMatrix(const SpMatrix S)
{
        int v = 0;
        int count;

        int i, j;
        for (i = 0; i < S.M; ++i)
        {
                count = 0;
                for (j = 0; j < S.N; ++j)
                {
                        if(count < (S.IA[i+1]-S.IA[i]) && j == S.JA[v])
                        {
                                printf("%g\t", S.A[v]);
                                v++;
                                count++;
                        }
                        else
                        {
                                printf("0\t");
                        }
                }
                printf("\n");
        }
}
