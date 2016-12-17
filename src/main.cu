#include "spmv.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

int main()
{
        // Set CUDA to prefer L1 cache over shared memory
        // Will set L1 cache to 48K and shared memory to 16K if possible
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        // Do gettimeofday timing for estimating execution time of main
        struct timeval t1_main, t2_main;//, t1, t2;
        double elapsedMain;
        gettimeofday(&t1_main, NULL);

        printf("Running main using spmvStrawberry with block size %i\n", BLOCK_SIZE);

        // PARAMETERS
        double p_diag = 0.9;
        double p_nondiag = 0.1;
        float *A_cpu, *A_gpu, *x_cpu, *x_gpu, *y_cpu, *y_gpu, *y_correct;
        int *IA_cpu, *IA_gpu, *JA_cpu, *JA_gpu;
        int NNZ;

        int expMmin = 10;
        int expMmax = 15;
        int Mmin = (1 << expMmin);
        int Mmax = (1 << expMmax);
        int L = expMmax-expMmin+1; //printf("L = %i\n", L);
        float *t_arr = (float *)malloc(sizeof(float)*L);
        int *M_arr = (int *)malloc(sizeof(float)*L);
        int i; int idx = 0;
        for (i = 0; i < L; ++i)
                M_arr[i] = (1 << (i+expMmin));

        // seed random number generator
        time_t t; srand((unsigned) time(&t));

        const int NUM_ITERS = 1;

        // Define cuda events for GPU timing
        float milliseconds;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Define CudaError
        cudaError_t err;

        // Define CUDA kernel parameters
        //int dB, dG;
        int dB_strawberry, dG_strawberry;

        int M, N, iter; double elapsed;
        for (M = Mmin; M <= Mmax; M=M*2)
        {
                elapsed = 0;
                for (iter = 0; iter < NUM_ITERS; ++iter)
                {
                        // Create sparse matrix
                        generateSquareSpMatrix(&A_cpu, &IA_cpu, &JA_cpu, &NNZ, M, p_diag, p_nondiag); // allocates!
                        N = M; // due to generation of square sparse matrix
                        
                        // Generate dense vector x
                        x_cpu = (float *)malloc(sizeof(float)*N);
                        fillDenseVector(x_cpu, N);
                        
                        // Define output vector y and y_correct
                        y_cpu = (float *)malloc(sizeof(float)*M);
                        y_correct = (float *)malloc(sizeof(float)*M);

                        // Setup memory on the GPU
                        cudaMalloc((void**) &A_gpu, NNZ*sizeof(float));
                        cudaMalloc((void**) &IA_gpu, (M+1)*sizeof(int)); // N = M
                        cudaMalloc((void**) &JA_gpu, NNZ*sizeof(int));
                        cudaMalloc((void**) &x_gpu, N*sizeof(float));
                        cudaMalloc((void**) &y_gpu, M*sizeof(float)); // N = M
        
                        // Transfer to device
                        cudaMemcpy(A_gpu, A_cpu, NNZ*sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy(IA_gpu, IA_cpu, (M+1)*sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(JA_gpu, JA_cpu, NNZ*sizeof(int), cudaMemcpyHostToDevice);
                        cudaMemcpy(x_gpu, x_cpu, N*sizeof(float), cudaMemcpyHostToDevice);
                        
                        // Set CUDA kernel parameters
                        //dB = BLOCK_SIZE;
                        //dG = N / 1024;
                        dB_strawberry = BLOCK_SIZE;
                        dG_strawberry = M / BLOCK_SIZE * 32;
                        
                        // Do CPU timing
                        //gettimeofday(&t1, NULL);
                        //spmvCPU(y_cpu, A_cpu, IA_cpu, JA_cpu, M, x_cpu);
                        //gettimeofday(&t2, NULL);
                        //elapsed += (t2.tv_sec-t1.tv_sec)*1000.0 + (t2.tv_usec-t1.tv_usec)/1000.0; // in ms

                        // Start cudaEvent timing
                        cudaEventRecord(start);
                        
                        // CUDA Vanilla SpMV Kernel
                        //spmvVanilla<<<dG, dB>>>(y_gpu, A_gpu, IA_gpu, JA_gpu, M, x_gpu);

                        // CUDA Chocolate SpMV Kernel
                        //spmvChocolate<<<dG, dB>>>(y_gpu, A_gpu, IA_gpu, JA_gpu, M, x_gpu);
                       
                        // CUDA Strawberry SpMV Kernel
                        spmvStrawberry<<< dG_strawberry, dB_strawberry >>>(y_gpu, A_gpu, IA_gpu, JA_gpu, M, x_gpu);

                        // Stop cudaEvent timing
                        cudaEventRecord(stop);
                        cudaEventSynchronize(stop);

                        // Check to make sure that cuda kernel was successful
                        err = cudaGetLastError();
                        if (err != cudaSuccess)
                                printf("Error: %s\n", cudaGetErrorString(err));

                        // Record timing result
                        milliseconds = 0;
                        cudaEventElapsedTime(&milliseconds, start, stop);
                        elapsed += milliseconds;

                        // Transfer result back to host
                        cudaMemcpy(y_cpu, y_gpu, N*sizeof(float), cudaMemcpyDeviceToHost);

                        // Test correctness of CUDA kernel vs "golden" cpu spmv function
                        spmvCPU(y_correct, A_cpu, IA_cpu, JA_cpu, N, x_cpu);
                        if (!areEqualRMSE(y_correct, y_cpu, N))
                        {
                                printf("Not correct result for a (%ix%i)*(%ix1) spmv multiplication\n", M, N, N);
                                printRMSE(y_correct, y_cpu, N);
                        }

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
                t_arr[idx++] = (float)elapsed/NUM_ITERS;
        }

        printf("Results averaged over %i iterations with time in ms:\n", NUM_ITERS);
        printf("M = "); printArray(M_arr, L);
        printf("t = "); printArray(t_arr, L);
        
        free(t_arr);
        free(M_arr);

        cudaDeviceReset();

        gettimeofday(&t2_main, NULL);
        //elapsed += (t2.tv_sec-t1.tv_sec)*1000.0 + (t2.tv_usec-t1.tv_usec)/1000.0; // in ms
        elapsedMain = (t2_main.tv_sec-t1_main.tv_sec) + (t2_main.tv_usec-t1_main.tv_usec)/1000000.0;
        printf("Total time to execute main() : %g seconds\n", elapsedMain);
	return 0;
}
