#include <stdio.h>
#include <math.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main(void){
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for (int n = 0; n < N; ++n){
        h_x[n] = a;
        h_y[n] = b;
    }

    double *d_x, *d_y, *d_z;  // 定义三个双精度变量指针
    cudaMalloc((void **)&d_x, M);  // 指针指向定义并分配显存的数组
    cudaMalloc((void **)&d_y, M);  // &d_y表示d_y的地址
    cudaMalloc((void **)&d_z, M);  // (void **) 是一个强制类型转换操作，可省略
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);  // 将主机中存放在h_x 和h_y 中的数据复制到设备中的相应变量d_x 和d_y 所指向的缓冲区中去。
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);  // (目标地址，源地址，复制字节数，方向)

    const int block_size = 128;
    const int grid_size = N / block_size;
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);  // 每个线程计算一个元素加法

    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);  // device上的数据复制会host
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}

void __global__ add(const double *x, const double *y, double *z){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(const double *z, const int N){
    // 在判断两个浮点数是否相等时，不能用运算符==，只能求差的绝对值
    bool has_error = false;
    for (int n = 0; n < N; n++){
        if (fabs(z[n] - c) > EPSILON){
            has_error = true;
        }
    }

    printf("%s\n", has_error ? "Has errors" : "No errors");
}





















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