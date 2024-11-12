// 定义cuda错误检查宏文件

#pragma once  // 确保当前文件在一个编译单元中不被重复包含
#include <stdio.h>

#define CHECK(call) \
do { \
    const cudaError_t error_code = call; \
    if (error_code != cudaSuccess){ \
        printf("CUDA Error:\n"); \
        printf(" File: %s\n", __FILE__); \
        printf(" Line: %d\n", __LINE__); \
        printf(" Error code: %d\n", error_code); \
        printf(" Error text: %s\n", cudaGetErrorString(error_code)); \
        exit(1); \
    } \
} while (0);

// 定义宏时，如果一行写不下，需要在行末写 \，表示续行
// call 表示调用CHECK所处理的函数
