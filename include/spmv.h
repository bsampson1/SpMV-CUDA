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
 */

#endif
