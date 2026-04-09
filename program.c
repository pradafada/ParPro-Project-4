#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define FactorIntToDouble 1.1
#define NUM_RUNS 5

double **firstMatrix;
double **secondMatrix;
double **matrixMultiResult;

int N;

/* Allocate matrix memory */
double** allocateMatrix(int n) {
    double **matrix = malloc(n * sizeof(double*));
    for(int i = 0; i < n; i++)
        matrix[i] = malloc(n * sizeof(double));
    return matrix;
}

/* Initialize matrices */
void matrixInit() {
    srand(time(NULL));

    for(int row = 0; row < N; row++) {
        for(int col = 0; col < N; col++) {
            firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            matrixMultiResult[row][col] = 0.0;
        }
    }
}

/* Parallel Matrix Multiplication */
void matrixMulti() {

    #pragma omp parallel for schedule(static)
    for(int row = 0; row < N; row++) {

        for(int col = 0; col < N; col++) {

            double resultValue = 0.0;

            for(int transNumber = 0; transNumber < N; transNumber++) {
                resultValue += firstMatrix[row][transNumber] *
                               secondMatrix[transNumber][col];
            }

            matrixMultiResult[row][col] = resultValue;
        }
    }
}

int main(int argc, char *argv[]) {

    if(argc != 2){
        printf("Usage: %s matrix_size\n", argv[0]);
        printf("Example: %s 2048\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);

    printf("Matrix Size: %d x %d\n", N, N);
    printf("Threads: %d\n", omp_get_max_threads());

    firstMatrix = allocateMatrix(N);
    secondMatrix = allocateMatrix(N);
    matrixMultiResult = allocateMatrix(N);

    matrixInit();

    double total_time = 0;

    for(int run = 0; run < NUM_RUNS; run++) {

        double start = omp_get_wtime();

        matrixMulti();

        double end = omp_get_wtime();

        double exec_time = end - start;

        printf("Run %d Time: %f seconds\n", run + 1, exec_time);

        total_time += exec_time;
    }

    printf("\nAverage Time: %f seconds\n", total_time / NUM_RUNS);

    return 0;
}