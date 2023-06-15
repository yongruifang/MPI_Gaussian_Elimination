#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#define N 1000 // Size of the matrix
#define COL 1001
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
    double t1 , t2;
    MPI_Init(&argc, &argv);
    t1 = MPI_Wtime();
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
            for (j = 0; j < N; j ++) {
                M[i*COL+j] = (double)rand() / RAND_MAX;
            }
            M[i*COL+N] = 0.0;
        }
        for (i = 0; i < N; i ++)
            for(j = 0; j < N; j ++) 
                    M[i*COL+N] += M[i*COL+j] * x[j];
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
    int local_pivot_row, global_pivot_row;
    for (i = 0; i < N; i ++) {// 逐列消元
        local_pivot_val = -1e10;
        local_pivot_row = -1;
        // Find the local pivot row with the maximum absolute value
        for(int row = 0; row < N/numproc; row++){ //逐行找主元
            int gRow = row*numproc + rank; // gRow是全局行号 
            if(rmap[gRow] < 0){ //第J行的主元为空
                if(M_buffer[row*COL+i] > local_pivot_val){ 
                    local_pivot_val = M_buffer[row*COL+i];
                    local_pivot_row = row;
                }                
            }
        }
        // Find the global pivot row with the maximum absolute value
        MPI_Allreduce(&local_pivot_val,&global_pivot_val,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        int local_pivot_proc = -1;// 记录主元所在的进程号
        double *temp_row = (double*)malloc(sizeof(double)*COL); // 临时存储行
        if(fabs(global_pivot_val-local_pivot_val)<0.0001){ 
            local_pivot_proc = rank; 
            for(j = 0 ; j < N+1; j ++){
                temp_row[j] = M_buffer[local_pivot_row*COL+j]; // 保存该行
            }
            global_pivot_row = local_pivot_row * numproc + rank; // 该行在全局矩阵中的位置
        }
        int global_pivot_proc; // 因为有可能有多个进程的主元值相同，所以需要再次通信
        MPI_Allreduce(&local_pivot_proc, &global_pivot_proc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD); // 取最大的进程号
        MPI_Bcast(&global_pivot_proc,1,MPI_INT,0,MPI_COMM_WORLD); // 广播主元所在的进程
        MPI_Bcast(temp_row,COL,MPI_DOUBLE,global_pivot_proc,MPI_COMM_WORLD); // 广播主元所在的行
        MPI_Bcast(&global_pivot_row,1,MPI_INT,global_pivot_proc,MPI_COMM_WORLD); // 广播主元所在的行号
        cmap[i] = global_pivot_row; // 第i列的主元在第global_pivot_row行
        rmap[global_pivot_row] = i; // 第global_pivot_row行的主元在第i列
        for(int row = 0; row < N/numproc; row ++){
            if(rmap[row*numproc+rank]<0){ // 第row行的主元为空，需要消元
                double temp = M_buffer[row*COL+i] / temp_row[i]; // 计算消元系数
                for(j = i; j < N+1; j ++){ // 逐列消元
                    M_buffer[row*COL+j] -= temp_row[j]*temp; // 消元
                }
            }
        }
    }
    // 收集矩阵
    MPI_Gather(M_buffer,N*COL/numproc,MPI_DOUBLE,M,N*COL/numproc,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // 打印上三角矩阵
    if(rank == 0){
        // 记录最后一行的映射
        for(i = 0 ; i < N; i ++){
            if(rmap[i] == -1){ 
                cmap[N-1] = i; // 记录最后一行的映射
            }
        }        
        // Perform back substitution 回代，求解x
        for (i = N-1; i >= 0; i --) {//自下而上
            int row = cmap[i]; 
            int index = row % numproc*(N/numproc) + row / numproc; // 该行在全局矩阵中的位置
            double sum = 0.0;
            for (j = i+1; j < N; j ++) {
                sum += M[index*COL+j] * x[j];
            }
            x[i] = (M[index*COL+N] - sum) / M[index*COL+i];
        }
        // Max error
        double max_error = 0.0;
        for (i = 0; i < N; i ++) {
            double sum = 0.0;
            for (j = 0; j < N; j ++) {
                sum += M[i*COL+j] * x[j];
            }
            double error = fabs(sum - M[i*COL+N]);
            if (error > max_error) max_error = error;
        }
        printf("\nMax error: %lf\n", max_error);
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
    t2 = MPI_Wtime();
    printf("it takes %.2lfs\n",t2-t1);
    MPI_Finalize();
    return 0;
}