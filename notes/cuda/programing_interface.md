# 编程接口

nvcc 编译流程：
1. nvcc分离全部源码：host code(c/c++语言), device code(c/c++扩展语言)
2. nvcc 将device code编译为 PTX（Parallel Thread Execution）为汇编代码，使用 `-arch=compute_XY`指定虚拟架构的计算能力
3. 将PTX编译为二进制的cubin目标代码，使用 `-code=sm_ZW`指定真实架构的计算能力（指定真实架构能力时必须指定且大于虚拟架构能力）


## 1. CUDA 兼容性问题
eg1: nvcc helloworld.cu -o helloworld -arch=compute_61
可以在计算能力 >=6.1 的 GPU 上执行
eg2: nvcc helloworld.cu -o helloworld -arch=compute_61 -code=sm_61






## 5. 加法运算
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