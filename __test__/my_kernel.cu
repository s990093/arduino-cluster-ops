/*
 * CUDA Kernel 模板
 * 
 * 复制这个文件开始编写你的 kernel
 */

#include "../micro_cuda_compiler/mcuda.h"

/**
 * 模板 Kernel 1: 简单处理
 * 
 * 每个 lane 处理一个元素
 */
__global__ void simple_process(int* input, int* output) {
    // 获取当前 lane 的 ID (0-7)
    int idx = laneId();
    
    // 从输入读取数据
    int value = input[idx];
    
    // 处理数据（在这里添加你的逻辑）
    int result = value * 2;  // 示例：简单倍数
    
    // 写回结果
    output[idx] = result;
}


/**
 * 模板 Kernel 2: 使用累加器
 * 
 * 适合需要多次计算的场景
 */
__global__ void accumulator_pattern(int* input, int* output) {
    int idx = laneId();
    
    // 累加器
    int sum = 0;
    
    // 多次处理（重用变量）
    int val;
    
    val = input[idx];
    sum = sum + val;
    
    val = input[idx + 8];
    sum = sum + val;
    
    val = input[idx + 16];
    sum = sum + val;
    
    // 写回结果
    output[idx] = sum;
}


/**
 * 模板 Kernel 3: 多参数处理
 */
__global__ void multi_param(int* A, int* B, int* C, int* D) {
    int idx = laneId();
    
    // 读取多个输入
    int a = A[idx];
    int b = B[idx];
    int c = C[idx];
    
    // 计算
    int result = (a + b) * c;
    
    // 写回
    D[idx] = result;
}


/**
 * 模板 Kernel 4: 邻域处理（类似卷积）
 */
__global__ void neighborhood(int* input, int* output, int width) {
    int idx = laneId();
    
    // 计算当前位置
    int center = idx * width;
    
    // 累加器
    int sum = 0;
    int val;
    
    // 处理邻域（例如：1D 3-point）
    val = input[center - 1];
    sum = sum + val;
    
    val = input[center];
    sum = sum + val;
    
    val = input[center + 1];
    sum = sum + val;
    
    output[idx] = sum;
}


/**
 * 你的 Kernel - 在这里开始编写！
 */
__global__ void your_kernel_here(int* input, int* output) {
    int idx = laneId();
    
    // TODO: 添加你的代码
    
    output[idx] = input[idx];
}
