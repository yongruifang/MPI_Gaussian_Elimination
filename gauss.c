#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#define N 10 // Size of the matrix
#define COL 11
// 复制矩阵，使得同个进程中需要的内存是连续的
int rank, numproc;
void copyMemory(const double *M, double *M_remake)
{
    int unit = N/numproc; // 每个进程需要的行数
    for(int i = 0; i < N; i++){
        int proc = i%numproc; //属于哪个进程的管辖
        int offset = i/numproc; // 在该进程中的行偏移量
        int row = proc*unit + offset;
        for(int j = 0; j <= N; j++){
            M_remake[row*COL+j] = M[i*COL+j];
        }        
    }   
}
int main(int argc, char* argv[]) {
    int i,j;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    // 线性方程组 Ax = b , A和b合并成增广矩阵M
    double *M, *M_remake, *x; // x是解向量
    double *M_buffer; // 每个进程需要的矩阵M的一部分
    if(rank == 0){
        // 为了保证有解，先生成解向量，b的值由随机矩阵A和x计算得到
        M = (double*)malloc(sizeof(double)*N*COL);
        M_remake = (double*)malloc(sizeof(double)*N*COL);
        x = (double*)malloc(sizeof(double)*N); 
        for (i = 0; i < N; i++) {
            x[i] = (double)rand() / RAND_MAX;
            M[i*COL + N] = 0.0;
            for (j = 0; j < N; j++) {
                M[i*COL+j] = (double)rand() / RAND_MAX;
                M[i*COL+N] += M[i*COL+j] * x[i];
            }
        }
        copyMemory(M,M_remake); // 复制矩阵，使得同个进程中需要的内存是连续的
    }
    int bsize = N*COL/numproc;
    M_buffer = (double*)malloc(sizeof(double)*bsize);
    MPI_Scatter(M_remake,N*COL/numproc,MPI_DOUBLE,M_buffer,bsize,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // Perform Gaussian elimination with partial pivoting
    // 引入双向映射，另类实现行交换
    int *cmap = (int*)malloc(N*sizeof(int)); //cmap[i]=j 表示第i列的主元在第j行
    int *rmap = (int*)malloc(N*sizeof(int)); //rmap[i]=j 表示第i行的主元在第j列
    for (i = 0; i < N; i ++){
        cmap[i] = -1;
        rmap[i] = -1;
    }
    double local_pivot_val, global_pivot_val;
    int local_pivot_idx, global_pivot_idx;
    for (i = 0; i < N; i ++) {// 逐列消元
        local_pivot_val = -1e10;
        local_pivot_idx = -1;
        // Find the local pivot row with the maximum absolute value
        for(int row = 0; row < N/numproc; row++){ //逐行找主元
            int gRow = row*numproc + rank; // gRow是全局行号 
            if(rmap[gRow] < 0){ //第J行的主元为空
                if(fabs(M_buffer[row*COL+i]) > local_pivot_val){
                    local_pivot_val = fabs(M_buffer[row*COL+i]);
                    local_pivot_idx = row;
                }                
            }
        }
        MPI_Allreduce(&local_pivot_val,&global_pivot_val,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        if(fabs(global_pivot_val-local_pivot_val)<0.0001){
            printf("rank %d, global_pivot_val = %f\n",rank,global_pivot_val);
        }
    }    
    if(rank==0){
        // 释放内存
        free(M);
        free(M_remake) ;
        free(M_buffer);
        free(x);
        free(cmap);
        free(rmap);
    }
    MPI_Finalize();
    return 0;
}