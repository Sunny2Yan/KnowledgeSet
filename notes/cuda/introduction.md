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