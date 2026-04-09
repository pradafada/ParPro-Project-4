#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define MAX_N 4096
#define BLOCK_SIZE 64
#define FactorIntToDouble 1.1

double firstMatrix[MAX_N][MAX_N];
double secondMatrix[MAX_N][MAX_N];
double matrixMultiResult[MAX_N][MAX_N];

int N;


/* Initialize matrices */
void matrixInit()
{
    for(int row = 0 ; row < N ; row++ ) {
        for(int col = 0 ; col < N ; col++){
            srand(row+col);
            firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            matrixMultiResult[row][col] = 0.0;
        }
    }
}


/* Block optimized multiplication for small submatrices */
void smallMatrixMult(int upperOfRow , int bottomOfRow ,
                     int leftOfCol , int rightOfCol ,
                     int transLeft , int transRight)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = upperOfRow; i <= bottomOfRow; i++)
    {
        for(int j = leftOfCol; j <= rightOfCol; j++)
        {
            double sum = matrixMultiResult[i][j];

            for(int k = transLeft; k <= transRight; k++)
            {
                sum += firstMatrix[i][k] * secondMatrix[k][j];
            }

            matrixMultiResult[i][j] = sum;
        }
    }
}


/* Recursive block division */
void matrixMulti(int upperOfRow , int bottomOfRow ,
                 int leftOfCol , int rightOfCol ,
                 int transLeft , int transRight )
{
    if ((bottomOfRow - upperOfRow) < 512)
    {
        smallMatrixMult(upperOfRow, bottomOfRow,
                        leftOfCol, rightOfCol,
                        transLeft, transRight);
    }
    else
    {
        int midRow = (upperOfRow + bottomOfRow) / 2;
        int midCol = (leftOfCol + rightOfCol) / 2;
        int midTrans = (transLeft + transRight) / 2;

        #pragma omp task
        {
            matrixMulti(upperOfRow, midRow,
                        leftOfCol, midCol,
                        transLeft, midTrans);

            matrixMulti(upperOfRow, midRow,
                        leftOfCol, midCol,
                        midTrans + 1, transRight);
        }

        #pragma omp task
        {
            matrixMulti(upperOfRow, midRow,
                        midCol + 1, rightOfCol,
                        transLeft, midTrans);

            matrixMulti(upperOfRow, midRow,
                        midCol + 1, rightOfCol,
                        midTrans + 1, transRight);
        }

        #pragma omp task
        {
            matrixMulti(midRow + 1, bottomOfRow,
                        leftOfCol, midCol,
                        transLeft, midTrans);

            matrixMulti(midRow + 1, bottomOfRow,
                        leftOfCol, midCol,
                        midTrans + 1, transRight);
        }

        #pragma omp task
        {
            matrixMulti(midRow + 1, bottomOfRow,
                        midCol + 1, rightOfCol,
                        transLeft, midTrans);

            matrixMulti(midRow + 1, bottomOfRow,
                        midCol + 1, rightOfCol,
                        midTrans + 1, transRight);
        }

        #pragma omp taskwait
    }
}



int main()
{
    int sizes[3] = {1024, 2048, 4096};

    for(int s = 0; s < 3; s++)
    {
        N = sizes[s];

        printf("Running matrix size: %d x %d\n", N, N);

        matrixInit();

        double t1 = omp_get_wtime();

        #pragma omp parallel
        {
            #pragma omp single
            {
                matrixMulti(0, N-1, 0, N-1, 0, N-1);
            }
        }

        double t2 = omp_get_wtime();

        printf("execution time: %f seconds\n\n", t2 - t1);
    }

    return 0;
}