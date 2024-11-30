# CUDA 基本框架

## 加法运算
1. 设置 GPU 设备
2. 分配 host 与 device 内存
3. 初始化 host 中的数据
4. 数据从 host 复制到 device
5. 调用核函数在 device 中进行计算
6. 将计算得到的数据从 device 中传给 host
7. 释放 host 与 device 内存

```cuda
#include <stdio>

int main(void)
{
    int iDeviceCount = 0;  // 查看GPU设备
    cudaError_t error = cudaGetDeviceCount(&iDeviceCount);
    
    if (error != cudaSuccess || iDeviceCount == 0)
    {
        printf("No CUDA campatable GPU found!\n");
        exit(-1)
    }
    else
    {
        printf("The count of GPU is %d.\n", iDeviceCount);
    }
    
    // 设置执行
    int iDev = 0;
    error = cudaSetDevice(iDev);  // 设置GPU设备
    if (error != cudaSuccess)
    {
        printf("Fail to set GPU 0 for computing.\n");
        exit(-1)
    }
    else
    {
        printf("Set GPU 0 for computing.\n");
    }
    
    return 0;
}
```

## 自定义设备函数

函数执行空间标识符：
`--global__`：核函数，由 host 调用，在 device 中执行；
`__device__`：设备函数，只能被核函数和其他设备函数调用；
`__host__`：普通C++函数，在主机中被调用，可以省略。可以和 `__device__` 共同使用；
`__device__ __host__`：既是一个 C++ 中的普通函数，又是一个设备函数。译器将针对主机和设备分别编译该函数。

```C++
// eg 1: 定义设备函数，并在核函数中调用
double __device__ add1_device(const double x, const double y){
    return (x + y);
}

void __global__ add1(const double *x, const double *y, double *z, const int N){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        z[n] = add1_device(x[n], y[n]);
    }
}

// eg 2: 用指针的设备函数
void __device__ add2_device(const double x, const double y, double *z){
    *z = x + y;
}

void __global__ add2(const double *x, const double *y, double *z, const int N){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
    add2_device(x[n], y[n], &z[n]);
    }
}

// eg 3: 用引用（reference）的设备函数
void __device__ add3_device(const double x, const double y, double &z){
    z = x + y;
}

void __global__ add3(const double *x, const double *y, double *z, const int N){
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N){
        add3_device(x[n], y[n], z[n]);
    }
}
```