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

void printArray(const float* arr, const int l);
/* Used to print A array from CSR formatted sparse matrix
 */

void printArray(const int* arr, const int l);
/* Used to print IA & JA array from CSR formatted sparse matrix
 */

void cpuSpMV(Vector* y, const SpMatrix* A, const Vector* x)
/*For testing and correctness purposes.
 *CPU implementation of matrix-vector multiplication for sparse
*/

bool areEqual(const Vector*a, const Vector*b)
/*Tests whether or not two vectors are equal
*/

bool areEqual(const SpMatrix*A, const SpMatrix*B)
/*Tests whether a sparse matrix is equivalent to a regular matrix (for correctness
 purposes
*/
void printSpMatrix(const SpMatrix S);
/* Takes in a sparse matrix in CSR format and prints the entire matrix using printf
 * WARNING: DO NOT USE THIS FUNCTION FOR LARGE MATRICES
 *      S - input sparse matrix
 */

SpMatrix generateSquareSpMatrix(const int N, const double p_diag, const double p_nondiag);
/* Takes in a MxN size and two probabilities and generates one realization of a sparse matrix
 *      N - number of row (and columns)
 *      p_diag - probability of non-zero element on main diagonal
 *      p_nondiag - probability of non-zero element not on main diagonal
 */

#endif
