#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>

int main()
{

        printf("\n============================== TEST: SPMV CORRECTNESS =====================================\n\n");

        // PARAMETERS
        double p_diag = 0.9;
        double p_nondiag = 0.1;
        float *A_cpu, *A_gpu, *x_cpu, *x_gpu;
        float *y_cpu_correct, *y_cpu_vanilla, *y_cpu_chocolate, *y_cpu_strawberry;
        float *y_gpu_vanilla, *y_gpu_chocolate, *y_gpu_strawberry;
        int *IA_cpu, *IA_gpu, *JA_cpu, *JA_gpu;
        int NNZ;

        // seed random number generator
        time_t t; srand((unsigned) time(&t));

        const int NUM_ITERS = 1;

        // Define cuda events
        int M, N, iter;
        for (M = (1 << 10); M <= (1 << 15); M=M*2)
        {
                for (iter = 0; iter < NUM_ITERS; ++iter)
                {
                        // Create sparse matrix
                        generateSquareSpMatrix(&A_cpu, &IA_cpu, &JA_cpu, &NNZ, M, p_diag, p_nondiag); // allocates!
                        N = M; // for square matrices

                        // Generate dense vector x
                        x_cpu = (float *)malloc(sizeof(float)*N);
                        fillDenseVector(x_cpu, N);
                        
                        // Define output vector y
                        y_cpu_correct = (float *)malloc(sizeof(float)*M);
                        y_cpu_vanilla = (float *)malloc(sizeof(float)*M);
                        y_cpu_chocolate = (float *)malloc(sizeof(float)*M);
                        y_cpu_strawberry = (float *)malloc(sizeof(float)*M);

                        // Setup memory on the GPU
                        cudaMalloc((void**) &A_gpu, NNZ*sizeof(float));
                        cudaMalloc((void**) &IA_gpu, (M+1)*sizeof(int)); // N = M
                        cudaMalloc((void**) &JA_gpu, NNZ*sizeof(int));
                        cudaMalloc((void**) &x_gpu, N*sizeof(float));
                        cudaMalloc((void**) &y_gpu_vanilla, M*sizeof(float)); // N = M
                        cudaMalloc((void**) &y_gpu_chocolate, M*sizeof(float)); // N = M
                        cudaMalloc((void**) &y_gpu_strawberry, M*sizeof(float)); // N = M
        
                        // Transfer to device
                        cudaMemcpy(A_gpu, A_cpu, NNZ*sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(IA_gpu, IA_cpu, (M+1)*sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(JA_gpu, JA_cpu, NNZ*sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(x_gpu, x_cpu, N*sizeof(float), cudaMemcpyHostToDevice);
                        
                        // CUDA kernel parameters
                        int dB = BLOCK_SIZE;
                        int dG = M / BLOCK_SIZE;
                        int dB_strawberry = BLOCK_SIZE;
                        int dG_strawberry = M / BLOCK_SIZE * 32;

                        // CPU SpMV kernel
                        spmvCPU(y_cpu_correct, A_cpu, IA_cpu, JA_cpu, M, x_cpu);

                        // Vanilla  SpMV CUDA kernel
                        spmvVanilla<<< dG, dB >>>(y_gpu_vanilla, A_gpu, IA_gpu, JA_gpu, M, x_gpu);
                        
                        // Chocolate SpMV CUDA kernel
                        spmvChocolate<<< dG, dB >>>(y_gpu_chocolate, A_gpu, IA_gpu, JA_gpu, M, x_gpu);

                        // Strawberry SpMV CUDA kernel
                        spmvStrawberry<<< dG_strawberry, dB_strawberry >>>(y_gpu_strawberry, A_gpu, IA_gpu, JA_gpu, M, x_gpu);

                        // Transfer result back to host
                        cudaMemcpy(y_cpu_vanilla, y_gpu_vanilla, M*sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(y_cpu_chocolate, y_gpu_chocolate, M*sizeof(float), cudaMemcpyDeviceToHost);
                        cudaMemcpy(y_cpu_strawberry, y_gpu_strawberry, M*sizeof(float), cudaMemcpyDeviceToHost);

                        // Test correctness of SpMV CUDA kernel flavors  vs "golden" cpu spmv function
                        if (areEqualRMSE(y_cpu_correct, y_cpu_vanilla, M))
                                printf("spmvVanilla is correct for a (%ix%i)*(%ix1) SpMV multiplication\n", M, N, N);
                        else
                                printf("spmvVanilla is NOT correct for a (%ix%i)*(%ix1) SpMV multiplication\n", M, N, N);
                        if (areEqualRMSE(y_cpu_correct, y_cpu_chocolate, M))
                                printf("spmvChocolate is correct for a (%ix%i)*(%ix1) SpMV multiplication\n", M, N, N);
                        else
                                printf("spmvChocolate is NOT correct for a (%ix%i)*(%ix1) SpMV multiplication\n", M, N, N);
                        if (areEqualRMSE(y_cpu_correct, y_cpu_strawberry, M))
                                printf("spmvStrawberry is correct for a (%ix%i)*(%ix1) SpMV multiplication\n", M, N, N);
                        else
                                printf("spmvStrawberry is NOT correct for a (%ix%i)*(%ix1) SpMV multiplication\n", M, N, N);

                        printf("\n");

                        // Free memory
                        free(A_cpu);
                        free(IA_cpu);
                        free(JA_cpu);
                        free(x_cpu);
                        free(y_cpu_correct);
                        free(y_cpu_vanilla);
                        free(y_cpu_chocolate);
                        free(y_cpu_strawberry);
                        cudaFree(A_gpu);
                        cudaFree(IA_gpu);
                        cudaFree(JA_gpu);
                        cudaFree(x_gpu);
                        cudaFree(y_gpu_vanilla);
                        cudaFree(y_gpu_chocolate);
                        cudaFree(y_gpu_strawberry);
                }
        }
        
        cudaDeviceReset();
	
        
        printf("\n===========================================================================================\n\n");
        return 0;
}
