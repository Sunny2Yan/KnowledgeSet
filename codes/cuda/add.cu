#include <stdio.h>
#include <sys/time.h>

// 计时函数
double get_walltime(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double) (tp.tv_sec + tp.tv_usec * 1e-6);
}

// 初始化向量数据
void initCpu(float *hostA, float *hostB, int n){
    for (int i = 0; i < n; i++){
        hostA[i] = i;
        hostB[i] = i;
    }
}

// CPU串行实现向量加法
void addCpu(float *hostA, float * hostB, float *hostC, int n){
    for (int i = 0; i < n; i++){
        hostC[i] = hostA[i] + hostB[i];
    }
}

// GPU并行实现向量加法
__global__ void addKernel(float *deviceA, float *deviceB, float *deviceC, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;  // 线程索引
    if (index < n){  // 没有for循环，每个线程做一个加法
        deviceC[index] = deviceA[index] + deviceB[index];
    }
}