# CUDA (Compute Unified Device Architecture)编程
一般 CPU 称为 host，GPU 称为 device，host 与 device之间的内存访问一般通过 PCIE (peripheral component interconnect express bus)总线链接（传输速度慢）。
注：GPU 计算不是指单独的 GPU 计算，而是指 CPU + GPU 的异构（heterogeneous）计算。GPU 不能单独计算，需要 CPU 做调度。

GPU的计算能力（compute capability）表示版本号，可以写为 X.Y 的形式。其中，X 表示主版本号，Y 表示次版本号。
注：计算能力并不等价于计算性能。

常用的GPU编程软件：
- CUDA：Nvidia GPU；
- OpenCL：更为通用的为各种异构平台编写并行程序的框架，也用于 AMD （Advanced Micro Devices）GPU；
- OpenACC：一个由多个公司共同开发的异构并行编程标准。

CUDA 支持 C, C++, Python 编程，提供两层 API 接口，cuda driver api (更底层) 和 cuda runtime api（用户更友好）：

```
                     |-> CUDA Lib -> CUDA Runtime API -> CUDA Driver API ->|
Application Program -|-------------> CUDA Runtime API -> CUDA Driver API ->|
                     |---------------------------------> CUDA Driver API ->| GPU
                     |                           cpu                       |
```

安装：在 `https://developer.nvidia.com/cuda-downloads` 下载安装 `CUDA Toolkit`， nvidia-smi中显示的是 CUDA Driver版本
切换到指定GPU：`export CUDA_VISIBLE_DEVICES=0,1`

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

// linux: g++ hello.cpp -o hello 编译源码 (hello)
// windows: cl hello.cpp -o hello (hello.exe)
// ./hello  执行文件
```