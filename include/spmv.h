#ifndef SPMV_H
#define SPMV_H

typedef struct
{
        int M, N, NNZ;
        float* A;
        int* IA;
        int* JA;
} SpMatrix;
/* Sparse Matrix Struct - CSR Format
 *      M - number of rows
 *      N - number of columns
 *      NNZ - number of nonzero elements
 *      A - pointer to float array containing nonzero elements (length NNZ)
 *      IA - pointer to row pointer array (length M+1)
 *      JA - pointer to column index array (length NNZ)
 */

void printSpMatrix(const SpMatrix S);
/* Takes in a sparse matrix in CSR format and prints the entire matrix using printf
 * WARNING: DO NOT USE THIS FUNCTION FOR LARGE MATRICES
 *      S - input sparse matrix
 */

SpMatrix generateSpMatrix(const int M, const int N, const double p_diag, const double p_nondiag);
/* Takes in a MxN size and two probabilities and generates one realization of a sparse matrix
 *      M - number of rows
 *      N - number of columns
 *      p_diag - probability of non-zero element on main diagonal
 *      p_nondiag - probability of non-zero element not on main diagonal
 */

#endif
