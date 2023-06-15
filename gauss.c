#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#define N 10 // Size of the matrix
#define COL 11
// 复制矩阵，使得同个进程中需要的内存是连续的
int rank, numproc;
int main(int argc, char* argv[]) {
    int i,j;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    // 线性方程组 Ax = b , A和b合并成增广矩阵M
    double *M, *x; // x是解向量
    double *M_buffer; // 每个进程需要的矩阵M的一部分
    if(rank == 0){
        // 为了保证有解，先生成解向量，b的值由随机矩阵A和x计算得到
        M = (double*)malloc(sizeof(double)*N*COL);
        x = (double*)malloc(sizeof(double)*N); 
        for (i = 0; i < N; i++) {
            x[i] = (double)rand() / RAND_MAX;
            M[i*COL + N] = 0.0;
            for (j = 0; j < N; j++) {
                M[i*COL+j] = (double)rand() / RAND_MAX;
                M[i*COL+N] += M[i*COL+j] * x[i];
            }
        }
    }
    int bsize = N*COL/numproc;
    M_buffer = (double*)malloc(sizeof(double)*bsize);
    MPI_Scatter(M,N*COL/numproc,MPI_DOUBLE,M_buffer,bsize,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // 打印局部矩阵
    printf("rank = %d\n",rank);
    for(i=0;i<bsize/COL;i++){
        for(j=0;j<COL;j++){
            printf("%lf ",M_buffer[i*COL+j]);
        }
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}