#ifndef SPMV_H
#define SPMV_H

void printArray(const float* arr, const int l);
/* Used to print A array from CSR formatted sparse matrix
 */

void printArray(const int* arr, const int l);
/* Used to print IA & JA array from CSR formatted sparse matrix
 */

void cpuSpMV(float *y, float *A, int *IA, int *JA, const int M, const float *x);
/* For testing and correctness purposes.
 * CPU implementation of matrix-vector multiplication for sparse
*/

bool areEqualRMSE(const float *a, const float *b, const int N);
/* Checks whether the rmse between float array a and b are below
 * some given rmse threshold. If it is, we can consider a and b 
 * equal and if not, we can consider they not equal. Internal
 * RMSE_THRESHOLD = 1e-5
 *      a, b - input float vectors
 *      N - length of vector
 */

__global__
void spmvSimple(float * y, const float *A, const int *IA, const int *JA, const float *x);
/* Sparse matrix-vector multiplication for GPU
 * One thread computes one element of output vector
 *      y - output vector (M x 1)
 *      A - sparse matrix (M x N)
 *      x - input vector  (N x 1)
 */

void printSpMatrix(const float* A, const int* IA, const int* JA, const int M, const int N);
/* Takes in a sparse matrix in CSR format and prints the entire matrix using printf
 * WARNING: DO NOT USE THIS FUNCTION FOR LARGE MATRICES
 *      S - input sparse matrix
 */

// WARNING: Generating sparse matrices with size N > 2^15 is prohibitively expensive for memory & time! Generally avoid doing this.
void generateSquareSpMatrix(float **A_p, int **IA_p, int **JA_p, int *NNZ_p, const int N, const double p_diag, const double p_nondiag);
/* Takes in a MxN size and two probabilities and generates one realization of a sparse matrix
 *      N - number of row (and columns)
 *      p_diag - probability of non-zero element on main diagonal
 *      p_nondiag - probability of non-zero element not on main diagonal
 */

void resizeSpMatrixArraysAndCopy(float **A_temp_p, int **JA_temp_p, int *bufferSize_p, double RESIZE_FACTOR);
/* Called from within generateSquareSpMatrix when we've exceeded the allowed buffer size.
 * The function will resize the temp arrays to length bufferSize * RESIZE_FACTOR provided in
 * the general case that RESIZE_FACTOR > 1. If you provide a RESIZE_FACTOR <= 1, it will be
 * resized by a factor of 1.33 by default. By the end of the function, bufferSize will point
 * to an integer which is the new length of the temporary sparse matrix arrays.
 *      A_temp - array that holds the float data of sparse matrix in CSR format
 *      JA_temp - array that holds the column indicies of sparse matrix in CSR format
 *      bufferSize - lengths of A_temp and JA_temp
 *      RESIZE_FACTOR - a constant determining the length of new temp arrays (should be > 1!)
 */

float getRandomFloat(const float min, const float max);
/* Returns a quasi-uniformly distributed random float between min and max
 */

void fillDenseVector(float* v, const int N);
/* Takes in a pointer, v, to a 1-D vector and a length N and fills it with random values
 * v - input vector pointer
 * N - input vector length
 */

//typedef struct
//{
//        int M, N, NNZ;
//        float* A;
//        int* IA;
//        int* JA;
//} SpMatrix;
/* Sparse Matrix Struct - CSR Format
 *      M - number of rows
 *      N - number of columns
 *      NNZ - number of nonzero elements
 *      A - pointer to float array containing nonzero elements (length NNZ)
 *      IA - pointer to row pointer array (length M+1)
 *      JA - pointer to column index array (length NNZ)
 */

#endif
