#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// 向量加法
const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void add(const double *x, const double *y, double *z, const int N);
void check(const double *z, const int N);

int main(void){
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *x = (double*) malloc(M);  // 指针变量指向一个长为M的数组
    double *y = (double*) malloc(M);
    double *z = (double*) malloc(M);

    for (int n = 0; n < N; ++n){  // 初始化数组元素
        x[n] = a;
        y[n] = b;
    }

    add(x, y, z, N);
    check(z, N);

    free(x);  // 释放分配的内存
    free(y);
    free(z);

    return 0;
}

void add(const double *x, const double *y, double *z, const int N){
    for (int n = 0; n < N; n++){
        z[n] = x[n] + y[n];
    }
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