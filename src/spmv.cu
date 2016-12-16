#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ 
void spmvStrawberry(    float *y, 
                        const float *A, 
                        const int *IA,
                        const int *JA,
                        const int M,
                        const float *x)
{
        
        __shared__ float t_sum[BLOCK_SIZE]; // thread sum
        int j;

        int t_id = threadIdx.x + blockDim.x * blockIdx.x; // thread id
        int w_id = t_id / 32; // warp id
        int t_warp = t_id & 31; // thread number within a given warp
        
        // one warp per row
        int row = w_id;

        // don't compute for a row value greater than the total in our matrix!
        if (row < M){

                // compute running sum per thread
                t_sum[threadIdx.x] = 0;

                for (j = IA[row] + t_warp; j < IA[row+1]; j += 32)
                        t_sum[threadIdx.x] += A[j] * x[JA[j]];

                // Add individual thread sums within a warp together
                // Final result for a row will be stored in the first thread
                // within each warp
                if (t_warp < 16) t_sum[threadIdx.x] += t_sum[threadIdx.x+16];
                if (t_warp < 8) t_sum[threadIdx.x] += t_sum[threadIdx.x+8];
                if (t_warp < 4) t_sum[threadIdx.x] += t_sum[threadIdx.x+4];
                if (t_warp < 2) t_sum[threadIdx.x] += t_sum[threadIdx.x+2];
                if (t_warp < 1) t_sum[threadIdx.x] += t_sum[threadIdx.x+1];
                
                // first thread within warp writes result to y
                if (t_warp == 0)
                        y[row] = t_sum[threadIdx.x];
        
        }
}

__global__
void spmvChocolate(     float* y,
                        const float *A,
                        const int *IA,
                        const int *JA,
                        const int M,
                        const float *x)
{
        __shared__ int IA_sh[BLOCK_SIZE+1];
        __shared__ float y_sh[BLOCK_SIZE];

        int j;
        int row = threadIdx.x + blockDim.x * blockIdx.x;

        // don't compute for a row value greater than the total in our matrix!
        if (row < M)
        {
                // Load shared data from global to shared memory
                IA_sh[threadIdx.x] = IA[row];
                IA_sh[BLOCK_SIZE] = IA[blockDim.x*(blockIdx.x+1)];
                y_sh[threadIdx.x] = y[row];
                __syncthreads();

                y_sh[threadIdx.x] = 0;
                for (j = IA_sh[threadIdx.x]; j < IA_sh[threadIdx.x+1]; ++j)
                        y_sh[threadIdx.x] += A[j]*x[JA[j]];

                // Return result from shared memory to global memory
                y[row] = y_sh[threadIdx.x];
        }
        __syncthreads();
}

__global__
void spmvVanilla(       float* y,
                        const float *A,
                        const int *IA,
                        const int *JA,
                        const int M,
                        const float *x)
{
        int j;
        int row = threadIdx.x + blockDim.x * blockIdx.x;
        
        // don't compute for a row greater than the total number in our matrix!
        if (row < M)
        {
                y[row] = 0;
                for (j = IA[row]; j < IA[row+1]; ++j)
                        y[row] += A[j]*x[JA[j]];
        }
        __syncthreads();
}

void spmvCPU(float *y, float *A, int *IA, int *JA, const int M, const float *x)
{
        int i, j;
        float sum;
        for (i = 0; i < M; ++i)
        {
                sum = 0;
                for (j = IA[i]; j < IA[i+1]; ++j)
                {
                        sum += A[j]*x[JA[j]];
                }
                y[i] = sum;
        }
}

void printArray(const float* arr, const int l)
{
        int i;
        printf("[");
        for (i = 0; i < l; ++i)
        {
                printf("%g", arr[i]);
                if (i != l-1)
                        printf("; ");
        }
        printf("];\n");
}

void printArray(const int* arr, const int l)
{
        int i;
        printf("[");
        for (i = 0; i < l; ++i)
        {
                printf("%i", arr[i]);
                if (i != l-1)
                        printf("; ");
        }
        printf("];\n");
}

bool areEqualRMSE(const float *a, const float *b, const int N)
{
        double RMSE_THRESHOLD = 1e-3;
        double sq_err_sum = 0; double rmse;
        int i;
        for (i = 0; i < N; ++i)
        {
                sq_err_sum += (a[i] - b[i])*(a[i] - b[i]);
        }
        rmse = sqrt(sq_err_sum/N);
        return rmse < RMSE_THRESHOLD;
}

void printRMSE(const float *a, const float *b, const int N)
{
        double sq_err_sum = 0; double rmse;
        int i;
        for (i = 0; i < N; ++i)
        {
                sq_err_sum += (a[i] - b[i])*(a[i] - b[i]);
        }
        rmse = sqrt(sq_err_sum/N);
        printf("RMSE = %g\n", rmse);
}

void printSpMatrix(const float* A, const int* IA, const int* JA, const int M, const int N)
{
        int v = 0;

        int i, j;
        for (i = 0; i < M; ++i)
        {
                for (j = 0; j < N; ++j)
                {
                        if(v < IA[i+1] && j == JA[v])
                        {
                                printf("%g\t", A[v]);
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

void generateSquareSpMatrix(float **A_p, int **IA_p, int **JA_p, int *NNZ_p, const int N, const double p_diag, const double p_nondiag)
{
        // estimate size of A, JA arrays because they vary between realization
        // but are same for a given realization
        int estSize = N*p_diag + N*(N-1)*p_nondiag;
        
        //printf("Estimate size of A: %i\n", estSize);
        //printf("Size of IA: %i\n", N+1);
        //printf("Estimate size of JA: %i\n", estSize);
        
        // allocate IA because size is fixed (size of IA = N + 1)
        *IA_p = (int *)malloc(sizeof(int)*(N+1));
        
        // define buffer space for undetermined arrays
        int bufferSize = (int)ceil(1.33*estSize);
        //printf("Buffer size = %i\n", bufferSize);
        
        // allocate buffer*estSize for A & JA so we can probably fit everything in those
        float* A_temp = (float *)malloc(sizeof(float)*bufferSize);
        int* JA_temp = (int *)malloc(sizeof(float)*bufferSize);
        
        double randProb; float randNum;

        // Setup inital conditions for sparse matrix
        *NNZ_p = 0; (*IA_p)[0] = 0;

        int i,j;
        for (i = 0; i < N; ++i)
        {
                (*IA_p)[i+1] = (*IA_p)[i];
                
                for (j = 0; j < N; ++j)
                {
                        randProb = ((double)rand())/RAND_MAX;
                        if (i == j) // on diagonal - use p_diag
                        {
                                if (randProb < p_diag) // insert non-zero element
                                {
                                        if((*NNZ_p) == bufferSize) // Placing element will exceed allowed buffer!
                                        {
                                                resizeSpMatrixArraysAndCopy(&A_temp, &JA_temp, &bufferSize, 1.33); // resize arrays so we can insert element!
                                                //printf("Error: Exceeded allowed buffer size. Failed to create sparse matrix!\n");
                                                //return;
                                        }
                                        
                                        // Place random non-zero element into sparse matrix
                                        randNum = getRandomFloat(0, 1);
                                        A_temp[(*NNZ_p)] = randNum;
                                        JA_temp[(*NNZ_p)] = j;
                                        (*IA_p)[i+1]++;
                                        (*NNZ_p)++;
                                }
                        }
                        else // not on diagonal - use p_nondiag
                        {
                                if (randProb < p_nondiag)
                                {
                                        if((*NNZ_p) == bufferSize) // Placing element will exceed allowed buffer!
                                        {
                                                resizeSpMatrixArraysAndCopy(&A_temp, &JA_temp, &bufferSize, 1.33); // resize arrays so we can insert element!
                                                //printf("Error: Exceeded allowed buffer size. Failed to create sparse matrix!\n");
                                                //return;
                                        }
                                        
                                        // Place random non-zero element into sparse matrix
                                        randNum = getRandomFloat(0, 1);;
                                        A_temp[(*NNZ_p)] = randNum;
                                        JA_temp[(*NNZ_p)] = j;
                                        (*IA_p)[i+1]++;
                                        (*NNZ_p)++;
                                        
                                }
                        }
                }
        }

        //printf("A_temp: "); printArray(A_temp, bufferSize);
        //printf("IA: "); printArray(*IA_p, N+1);
        //printf("JA_temp: "); printArray(JA_temp, bufferSize);

        // By this point we have not exceeded memory limit so lets create
        // actual A and IA array now that we have determined the size
        *A_p = (float *)malloc(sizeof(float)*(*NNZ_p));
        *JA_p = (int *)malloc(sizeof(float)*(*NNZ_p));
        
        // Add elements from temp arrays to actual arrays
        for (i = 0; i < (*NNZ_p); ++i)
        {
                (*A_p)[i] = A_temp[i];
                (*JA_p)[i] = JA_temp[i];
        }
        
        //printf("A: "); printArray(*A_p, *NNZ_p);
        //printf("IA: "); printArray(*IA_p, N+1);
        //printf("JA: "); printArray(*JA_p, *NNZ_p);
        //printf("NNZ: %i\n", *NNZ_p);
       
        // free no longer used temp arrays
        free(A_temp); A_temp = NULL;
        free(JA_temp); JA_temp = NULL;
        
        return;
}

void resizeSpMatrixArraysAndCopy(float **A_temp_p, int **JA_temp_p, int *bufferSize_p, double RESIZE_FACTOR)
{

        printf("Executing resize!!\n");
        if (RESIZE_FACTOR <= 1) // RESIZE_FACTOR should not be less than one!
                RESIZE_FACTOR = 1.33; // if so, set to default value of 1.33

        int oldLength = (*bufferSize_p);
        int newLength = (int)ceil((*bufferSize_p)*RESIZE_FACTOR);
        float *A_temp_new;
        int *JA_temp_new;

        // allocate the new resized memory
        A_temp_new = (float *)malloc(sizeof(float)*newLength);
        JA_temp_new = (int *)malloc(sizeof(int)*newLength);

        // copy old elements into new array
        int i;
        for (i = 0; i < oldLength; ++i)
        {
                A_temp_new[i] = (*A_temp_p)[i];
                JA_temp_new[i] = (*JA_temp_p)[i];
        }

        // free memory from old arrays
        free(*A_temp_p);
        free(*JA_temp_p);

        // update pointers
        *A_temp_p = A_temp_new; A_temp_new = NULL;
        *JA_temp_p = JA_temp_new; A_temp_new = NULL;

        // update bufferSize
        *bufferSize_p = newLength;
}

float getRandomFloat(const float min, const float max)
{
        return ((((float)rand())/RAND_MAX)*(max-min)+min);
}

void fillDenseVector(float* v, const int N)
{
        int i;
        for (i = 0; i < N; ++i)
                v[i] = getRandomFloat(0, 1);
}










