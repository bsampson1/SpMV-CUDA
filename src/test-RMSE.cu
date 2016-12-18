#include "spmv.h"
#include <stdio.h>

int main()
{

        printf("\n============================== TEST: RMSE =================================================\n\n");

        float *a, *b;
        int N = 10;
        float diff1 = 1e-5;
        float diff2 = 1e-3;

        // seed random number generator
        time_t t; srand((unsigned) time(&t));

        a = (float *)malloc(sizeof(float)*N);
        b = (float *)malloc(sizeof(float)*N);

        // TEST CASE 1: a and b assigned same values, EXPECTED RESULT: PASS
        int i;
        for (i = 0; i < N; ++i)
        {
                a[i] = i;
                b[i] = i;
        }

        printf("a = "); printArray(a, N);
        printf("b = "); printArray(b, N);
        printRMSE(a, b, N);

        if(areEqualRMSE(a, b, N))
                printf("They are equal in the sense that the RMSE of a-b is below RMSE error threshold\n\n");
        else
                printf("They are NOT equal in the sense that the RMSE of a-b is NOT below RMSE error threshold\n\n");

        // TEST CASE 2: a and b are different by a value less than the RMSE threshold of 1e-4, EXPECTED RESULT: PASS
        for (i = 0; i < N; ++i)
                b[i] += diff1;

        printf("a = "); printArray(a, N);
        printf("b = "); printArray(b, N);
        printRMSE(a, b, N);

        if(areEqualRMSE(a, b, N))
                printf("They are equal in the sense that the RMSE of a-b is below RMSE error threshold\n\n");
        else
                printf("They are NOT equal in the sense that the RMSE of a-b is NOT below RMSE error threshold\n\n");

        // TEST CASE 3: a and b are different by a value greater than the RMSE threshold of 1e-4, EXPECTED RESULT: FAIL
        for (i = 0; i < N; ++i)
                b[i] = a[i] + diff2;

        printf("a = "); printArray(a, N);
        printf("b = "); printArray(b, N);
        printRMSE(a, b, N);

        if(areEqualRMSE(a, b, N))
                printf("They are equal in the sense that the RMSE of a-b is below RMSE error threshold\n\n");
        else
                printf("They are NOT equal in the sense that the RMSE of a-b is NOT below RMSE error threshold\n\n");
        
        // TEST CASE 4: a and b are random vectors drawn from the uniform distribution, EXPECTED RESULT: FAIL
        fillDenseVector(a, N);
        fillDenseVector(b, N);

        printf("a = "); printArray(a, N);
        printf("b = "); printArray(b, N);
        printRMSE(a, b, N);

        if(areEqualRMSE(a, b, N))
                printf("They are equal in the sense that the RMSE of a-b is below RMSE error threshold\n\n");
        else
                printf("They are NOT equal in the sense that the RMSE of a-b is NOT below RMSE error threshold\n\n");

        free(a);
        free(b);


        printf("===========================================================================================\n\n");

        return 0;
}
