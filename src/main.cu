#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{
        // PARAMETERS
        double p_diag = 0.9;
        double p_nondiag = 0.1;
        int N = (1 << 12);
        float *A_cpu, *A_gpu, *x_cpu, *x_gpu, *y_cpu, *y_gpu, *y_correct;
        int *IA_cpu, *IA_gpu, *JA_cpu, *JA_gpu;
        int NNZ;
 
        // seed random number generator
        time_t t; srand((unsigned) time(&t));

        printf("Computing spmv for %ix%i sparse matrix\n", N, N);

        // Define cuda events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Create sparse matrix
        generateSquareSpMatrix(&A_cpu, &IA_cpu, &JA_cpu, &NNZ, N, p_diag, p_nondiag); // allocates!

        // Verify correctness by hand
        //printf("Sparse Matrix S: \n"); printSpMatrix(S);
        //printf("A: "); printArray(A_cpu, NNZ);
        //printf("IA: "); printArray(IA_cpu, N+1);
        //printf("JA: "); printArray(JA_cpu, NNZ);

        // Generate dense vector x
        x_cpu = (float *)malloc(sizeof(float)*N);
        fillDenseVector(x_cpu, N);
        //printf("Dense vector x: "); printArray(x_cpu, N);
        
        // Define output vector y
        y_cpu = (float *)malloc(sizeof(float)*N);

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
        int dB, dG;
        if (N < 1024)
        {
                dB = N;
                dG = 1;
        }
        else
        {
                dB = BLOCK_SIZE;
                dG = N / BLOCK_SIZE;
        }

        // Start cudaEvent timing
        cudaEventRecord(start);
        
        // Vanilla SpMV CUDA Kernel
        spmvVanilla<<< dG, dB>>>(y_gpu, A_gpu, IA_gpu, JA_gpu, x_gpu);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Print result
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Elapsed time (ms) = %f\n", milliseconds);

        // Transfer to host
        cudaMemcpy(y_cpu, y_gpu, N*sizeof(float), cudaMemcpyDeviceToHost);
        //printf("Output vector y: "); printArray(y_cpu, N);

        // Verify correctness of CUDA kernel
        y_correct = (float *)malloc(sizeof(float)*N);
        cpuSpMV(y_correct, A_cpu, IA_cpu, JA_cpu, N, x_cpu);
        //printf("Correct output vector y_correct: "); printArray(y_correct, N);

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

        cudaDeviceReset();
	return 0;
}
