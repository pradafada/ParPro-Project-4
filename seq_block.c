#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 64
#define FactorIntToDouble 1.1

double **A;
double **B;
double **C;

void allocate_matrices(int N)
{
    A = (double**)malloc(N * sizeof(double*));
    B = (double**)malloc(N * sizeof(double*));
    C = (double**)malloc(N * sizeof(double*));

    for(int i = 0; i < N; i++)
    {
        A[i] = (double*)malloc(N * sizeof(double));
        B[i] = (double*)malloc(N * sizeof(double));
        C[i] = (double*)malloc(N * sizeof(double));
    }
}

void matrixInit(int N)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            srand(i + j);
            A[i][j] = (rand() % 10) * FactorIntToDouble;
            B[i][j] = (rand() % 10) * FactorIntToDouble;
            C[i][j] = 0.0;
        }
    }
}

void matrixMultiplyBlocked(int N)
{
    for(int ii = 0; ii < N; ii += BLOCK_SIZE)
    {
        for(int jj = 0; jj < N; jj += BLOCK_SIZE)
        {
            for(int kk = 0; kk < N; kk += BLOCK_SIZE)
            {
                for(int i = ii; i < ii + BLOCK_SIZE && i < N; i++)
                {
                    for(int j = jj; j < jj + BLOCK_SIZE && j < N; j++)
                    {
                        double sum = C[i][j];

                        for(int k = kk; k < kk + BLOCK_SIZE && k < N; k++)
                        {
                            sum += A[i][k] * B[k][j];
                        }

                        C[i][j] = sum;
                    }
                }
            }
        }
    }
}

void free_matrices(int N)
{
    for(int i = 0; i < N; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }

    free(A);
    free(B);
    free(C);
}

int main()
{
    int sizes[3] = {1024, 2048, 4096};

    for(int s = 0; s < 3; s++)
    {
        int N = sizes[s];

        printf("Running matrix size: %d x %d\n", N, N);

        allocate_matrices(N);
        matrixInit(N);

        clock_t start = clock();

        matrixMultiplyBlocked(N);

        clock_t end = clock();

        double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

        printf("Execution time: %f seconds\n\n", time_taken);

        free_matrices(N);
    }

    return 0;
}