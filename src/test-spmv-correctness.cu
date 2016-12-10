#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{
        // PARAMETERS
        double p_diag = 0.9;
        double p_nondiag = 0.001;
        float *A_cpu, *A_gpu, *x_cpu, *x_gpu, *y_cpu, *y_gpu, *y_correct;
        int *IA_cpu, *IA_gpu, *JA_cpu, *JA_gpu;
        int NNZ;

        // seed random number generator
        time_t t; srand((unsigned) time(&t));

        const int NUM_ITERS = 1;

        // Define cuda events
        int N, iter;
        for (N = 2; N <= (1 << 15); N=N*2)
        {
                for (iter = 0; iter < NUM_ITERS; ++iter)
                {
                        // Create sparse matrix
                        generateSquareSpMatrix(&A_cpu, &IA_cpu, &JA_cpu, &NNZ, N, p_diag, p_nondiag); // allocates!

                        // Generate dense vector x
                        x_cpu = (float *)malloc(sizeof(float)*N);
                        fillDenseVector(x_cpu, N);
                        
                        // Define output vector y
                        y_cpu = (float *)malloc(sizeof(float)*N);
                        y_correct = (float *)malloc(sizeof(float)*N);

                        // Setup memory on the GPU
                        cudaMalloc((void**) &A_gpu, NNZ*sizeof(float));
                        cudaMalloc((void**) &IA_gpu, (N+1)*sizeof(int)); // N = M
                        cudaMalloc((void**) &JA_gpu, NNZ*sizeof(int));
                        cudaMalloc((void**) &x_gpu, N*sizeof(float));
                        cudaMalloc((void**) &y_gpu, N*sizeof(float)); // N = M
        
                        // Transfer to device
                        cudaMemcpy(A_gpu, A_cpu, NNZ*sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(IA_gpu, IA_cpu, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(JA_gpu, JA_cpu, NNZ*sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(x_gpu, x_cpu, N*sizeof(float), cudaMemcpyHostToDevice);
                        
                        // CUDA kernel parameters
                        int threadsPerBlock, blocksPerGrid;
                        if (N < 1024)
                        {
                                threadsPerBlock = N;
                                blocksPerGrid = 1;
                        }
                        else
                        {
                                threadsPerBlock = 1024;
                                blocksPerGrid = N / 1024;
                        }

                        // Simple SpMV CUDA kernel
                        spmvSimple<<<blocksPerGrid, threadsPerBlock>>>(y_gpu, A_gpu, IA_gpu, JA_gpu, x_gpu);
                        
                        // Transfer result back to host
                        cudaMemcpy(y_cpu, y_gpu, N*sizeof(float), cudaMemcpyDeviceToHost);

                        // Test correctness of CUDA kernel vs "golden" cpu spmv function
                        cpuSpMV(y_correct, A_cpu, IA_cpu, JA_cpu, N, x_cpu);
                        if (areEqualRMSE(y_correct, y_cpu, N))
                                printf("GPU kernel result is correct for a (%ix%i)*(%ix1) spmv multiplication\n", N, N, N);
                        else
                                printf("GPU kernel result is NOT correct for a (%ix%i)*(%ix1) spmv multiplication\n", N, N, N);


                        // Free memory
                        free(A_cpu);
                        free(IA_cpu);
                        free(JA_cpu);
                        free(x_cpu);
                        free(y_cpu);
                        free(y_correct);
                        cudaFree(A_gpu);
                        cudaFree(IA_gpu);
                        cudaFree(JA_gpu);
                        cudaFree(x_gpu);
                        cudaFree(y_gpu);
                }
        }
        
        cudaDeviceReset();
	return 0;
}
