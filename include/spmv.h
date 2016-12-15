#ifndef SPMV_H
#define SPMV_H

#define BLOCK_SIZE 1024

void printArray(const float* arr, const int l);
/* Prints float array of length l
 */

void printArray(const int* arr, const int l);
/* Prints int array of length l
 */

void cpuSpMV(float *y, float *A, int *IA, int *JA, const int M, const float *x);
/* CPU implementation of matrix-vector multiplication for SpMV multiplcation
*/

bool areEqualRMSE(const float *a, const float *b, const int N);
/* Checks whether the rmse between float array a and b are below
 * some given rmse threshold. If it is, we can consider a and b 
 * equal and if not, we can consider they not equal. Internal
 * RMSE_THRESHOLD = 1e-3
 */

__global__
void spmvChocolate(float *y, const float *A, const int *IA, const int *JA, const float *x);
/* Sparse matrix-vector multiplication for GPU
 * One thread computes one element of output vector y
 * but after loading to shared memory first
 */

__global__
void spmvVanilla(float *y, const float *A, const int *IA, const int *JA, const float *x);
/* Sparse matrix-vector multiplication for GPU
 * One thread computes one element of output vector using global memory
 */

// WARNING: DO NOT USE THIS FUNCTION FOR LARGE MATRICES
void printSpMatrix(const float* A, const int* IA, const int* JA, const int M, const int N);
/* Prints a CSR-formatted sparse matrix
 */

// WARNING: Generating sparse matrices with size N > 2^15 is prohibitively expensive for memory & time! Generally avoid doing this.
void generateSquareSpMatrix(float **A_p, int **IA_p, int **JA_p, int *NNZ_p, const int N, const double p_diag, const double p_nondiag);
/* Generates a square sparse matrix of dimension N in CSR format with elements of the diagonal according to probability p_diag and
 * elements off the diagonal according to probability p_nondiag
 */

void resizeSpMatrixArraysAndCopy(float **A_temp_p, int **JA_temp_p, int *bufferSize_p, double RESIZE_FACTOR);
/* Called from within generateSquareSpMatrix function when we have run out of room to store new elements
 * in A & JA. Will create new arrays of length RESIZE_FACTOR times the original length (bufferSize) and 
 * copy the elements into the new larger array.
 */

float getRandomFloat(const float min, const float max);
/* Returns a quasi-uniformly distributed random float between min and max
 */

void fillDenseVector(float* v, const int N);
/* Takes in a pointer to a vector and fills it with random floats
 * using getRandomFloat
 */

#endif
