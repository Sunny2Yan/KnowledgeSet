# CUDA 编程
一般 CPU 称为 host，GPU 称为 device，host 与 device之间的内存访问一般通过 PCIE 总线链接（传输速度慢）。
注：GPU 不能单独计算，需要 CPU 做调度。

CUDA 支持 C, C++, Python 编程，提供两层 API 接口，cuda driver api 和 cuda runtime api（用户更友好）：

```
                     |-> CUDA Lib ---|                    |
Application Program -|------> CUDA Runtime API -|         |
                     |----------------> CUDA Driver API ->| GPU
                       cpu                                |
```

安装：`CUDA Toolkit`， nvidia-smi中显示的是 CUDA Driver版本

## 1. C++ 简介
C++编译过程：编写源代码 -> 编译器对源码进行预处理、编译、链接，生成可执行文件 -> 执行可执行文件

```c++
// hello.cpp
#include <stdio.h>

int main(void)
{
    printf("hello world!\n");

    return 0;
}

// g++ hello.cpp -o hello 编译源码
// ./hello  执行文件
```

## 2. 核函数
CUDA 程序开发既需要编写在 CPU 上面的 host 代码，也要编写运行在 GPU 上面的 device 代码，host 对 device 的调用是通过 kernel function 进行的。
即，CUDA 源码中含有 C++ 部分，也含有不属于 C++ 的部分，在编译时 C++ 部分由 C++ 编译器编译，其他部分由 CUDA 编译。

核函数是在GPU上并行执行，且由限定词 `__global__`修饰，返回必须是 `void`。

```
__global__ void hello_from_gpu()  // 核函数就是在c++函数前加 __global__ 限定词
{
    printf("hello world from the gpu\n")
}

void __global__ hello_from_gpu()  // 放到 void 后面也可以
{
    printf("hello world from the gpu\n")
}
```

注：
- 核函数只能访问GPU 内存（通过PCIE） 
- 核函数不能使用变长参数（需要明确参数的个数）
- 核函数不能使用静态变量；
- 核函数不能使用函数指针；
- 核函数具有异步性。

cuda 程序编写流程：
```
int main(void)
{
    host code
    kernel function call  // 核函数不支持c++ iostream，即cout
    host code
    return0;
}
```

举例：
```cuda
// test.cu
#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("hello world from the gpu\n");
}

int main(void)
{
    hello_from_gpu<<<2, 3>>>();  // <>中分别指定线程块数量，每个块的线程数
    cudaDeviceSynchronize();  // 可能GPU还没执行完，但CPU执行完了，需要同步host与device，让CPU等待GPU执行完

    return 0;
}

// nvcc test.cu -o test  # 编译
// ./test  运行
```

## 3. 线程模型
启动一个核函数包含一个 grid，grid 中包含若干个 thread block, 即 <<<grid_size, block_size>>>。
线程是 GPU 编程中的最小单位，线程分块是逻辑上的划分，物理上不分块。
最大线程块大小：1024；最大网格大小：2^31-1（一维网格）

每个线程唯一标识符，由保存在内建变量（build-in variable）中的grid_size, block_size确定的：
`gridDim.x` = grid_size；`blockDim.x` = block_size

线程索引保存在内建变量中：
`blockIdx.x`, `threadIdx.x`

唯一标识符：Idx = threadIdx.x + blockIdx.x * blockDim.x

推广到多（如，3）维线程：blockIdx 和 threadIdx 是类型为 uint3 的变量存储在结构体中：
`blockIdx.x, blockIdx.y, blockIdx.z`, `threadIdx.x, threadIdx.y, threadIdx.z`

定义网格与线程块（缺省的默认为1）：
dim3 grid_size(2, 2)  <==> (2, 2, 1)
dim3 block_size(2, 2, 2)

多维线程块中线程索引：
int tid = threadIdx.z + blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x
多维网格中线程块索引：
int bid = blockIdx.z + gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x

grid size 限制：
gridDim.x <= 2^31-1; gridDim.y <= 2^16-1; gridDim.z <= 2^16-1;
block size 限制：
blockDim.x <= 1024; blockDim.y <= 1024; blockDim.z <= 64;
注：线程块总个数不能超过1024，即 x * y * z <= 1024

```c++
// test.cu
#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = threadIdx.x + blockIdx.x * blockDim.x
    printf("hello world from block %d and thread %d, global id %d\n", bid, tid, id);
}

int main(void)
{
    hello_from_gpu<<<2, 4>>>();  
    cudaDeviceSynchronize();

    return 0;
}
```

### 3.1 线程全局索引计算方式 （变成一维，从0编号）
1. 一维网格，一维线程块：
   dim3 grid_size(4);
   dim3 block_size(8);
   kernel_func<<<grid_size, block_size>>>(...)

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
2. 二维网格，二维线程块
   dim3 grid_size(2, 2);
   dim3 block_size(4, 4);
   kernel_func<<<grid_size, block_size>>>(...)
   
   int blockId = blockIdx.x + blockIdx.y * gridDim.x
   int threadId = threadIdx.x + threadIdx.y * blockDim.x
   int idx = blockId * (blockDim.x * blockDim.y) + threadId;
3. 三维网格，三维线程块
   dim3 grid_size(2, 2, 2);
   dim3 block_size(4, 4, 2);
   kernel_func<<<grid_size, block_size>>>(...)
   
   int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * gridDim.z
   int threadId = threadIdx.x + threadIdx.y * blockDim.x
   int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + threadId;

## 4. CUDA 兼容性问题
nvcc 编译流程：
1. nvcc分离全部源码：host code(c/c++语言), device code(c/c++扩展语言)
2. nvcc 将device code编译为 PTX（Parallel Thread Execution）为汇编代码，使用 `-arch=compute_XY`指定虚拟架构的计算能力
3. 将PTX编译为二进制的cubin目标代码，使用 `-code=sm_ZW`指定真实架构的计算能力（指定真实架构能力时必须指定且大于虚拟架构能力）

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