# 编程模型

## 1. 核函数
CUDA 程序开发既需要编写在 CPU 上面的 host 代码，也要编写运行在 GPU 上面的 device 代码，host 对 device 的调用是通过 kernel function 进行的。
即，CUDA 源码中含有 C++ 部分，也含有不属于 C++ 的部分，在编译时 C++ 部分由 C++ 编译器编译，其他部分由 CUDA 编译。

当调用内核时，由 N 个不同的 CUDA 线程并行执行 N 次，而不是像常规 C++ 函数那样只执行一次.

核函数是在GPU上并行执行，且由限定词 `__global__`修饰，返回必须是 `void`。

```
__global__ void hello_from_gpu()  // 核函数就是在c++函数前加 __global__ 限定词
{
    printf("hello world from the gpu\n");
}

void __global__ hello_from_gpu()  // 放到 void 后面也可以
{
    printf("hello world from the gpu\n");
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
```C++
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

## 2. 线程层次结构
启动一个核函数包含一个 grid，grid 中包含若干个 thread block, 即 <<<grid_size, block_size>>>。
线程是 GPU 编程中的最小单位，线程分块是逻辑上的划分，物理上不分块。
最大线程块大小：1024；最大网格大小：2^31-1（一维网格）

每个线程都有唯一标识符threadIdx， 它是一个类型为 `dim3` 的3分量向量，因此可以使用一个一维、二维或三维的线程索引(thread index)来识别线程。
线程块索引 `blockIdx` 与线程索引 `threadIdx` 都保存在内建变量（build-in variable）中，且由保存在内建变量中的 grid_size, block_size 确定。

线程全局索引计算方式 （变成一维，从0编号）：
1. 一维网格，一维线程块：
   dim3 grid_size(4);  
   dim3 block_size(8);
   kernel_func<<<grid_size, block_size>>>(...)

   int idx = threadIdx.x + blockIdx.x * blockDim.x;
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

注：定义网格与线程块时，缺省的默认为 1，如：grid_size(2, 2)  <==> (2, 2, 1)。

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

## 3. 内存层次结构
1. 每个线程都有私有的本地内存 (local memory)。 
2. 每个线程块都具有共享内存 (shared memory)，该共享内存内存对该块中的所有线程可见，并且具有与该块相同的生命周期。 
3. 所有线程都可以访问相同的全局内存 (global memory)。