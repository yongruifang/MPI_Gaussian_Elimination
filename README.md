# MPI_Gaussian_Elimination
高斯消元法-MPI并行程序
# 实验内容
用MPI编写用选主元的高斯消去法(有回代)求解n元线性方程组Ax=b的并行程序。A、x、b采用行块带状划分存储，初始时，每个进程将所在的A、x子块中的每个数都初始化为一个0到1之间的随机double型值：rand()/double(RAND_MAX)。为了验证结果正确性，将求解得到的x带入原方程组，求出最大误差：向量Ax-b中绝对值最大的分量。执行时间不包括A、x、b的初始化时间和验证时间，通过函数MPI_Wtime()计算。在下面写出完整的程序代码，并添加必要的注释
# 实现行交换的小技巧
用维护两个一维数组map的方式来取代行交换，减少时间开销。
cmap[i]=j 指的是 第i列的主元行是j 
rmap[i]=j 指的是 第i行的主元列是i
# 算法设计
1. 初始化，将A和b列组合成增广矩阵M
2. 寻找第i列主元，记录到映射数组
3. 通过主元行消去映射数组中未定义的行，让在映射数组中未定义的行的第i个元素为0
4. 迭代 直到i=N-1，化成上三角矩阵
5. 回代求解x
# 实现难点
- 如何采用交叉分配的方式来分配内存
- 怎么并行寻找主元行
# 算法改进
经过改进，<kbd>行数%进程数!=0</kbd>也可以正常运行
# 程序运行
- 环境：Linux
命令：
```bash
mpicc -O3 gauss.c -o gauss
mpirun -np <进程数> ./gauss
```
- 环境：Windows
命令：
```bash
gcc  -O3 gauss.c -o gauss.c -l msmpi -L "D:\Microsoft SDKs\MPI\Lib\x64" -I "D:\Microsoft SDKs\MPI\Include"
mpiexec -np <进程数> ./gauss
```
# 问题
- 无回代的高斯-约旦法应该如何实现？
