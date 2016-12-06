#include "spmv.h"

int main()
{
        double p_diag = 0.9;
        double p_nondiag = 0.01;
        int N = 24;

        SpMatrix S = generateSquareSpMatrix(N, p_diag, p_nondiag);
        printSpMatrix(S);

        // Free memory
        free(S.IA);
        free(S.JA);
        free(S.A);
        return 0;
}
