/*
 * Convolution Kernel - 1D Convolution (size 3)
 * 
 * Computes: Output[i] = Input[i]*K[0] + Input[i+1]*K[1] + Input[i+2]*K[2]
 * 
 * Memory Layout:
 * - Input I:  VRAM[0x00-0x2F]  (12 elements: 1..12)
 * - Kernel K: VRAM[0x40-0x4B]  (3 elements: 2,3,4 + padding)
 * - Output O: VRAM[0x80-0x9F]  (8 elements, computed)
 * 
 * Each lane computes one output element.
 */

#include "../mcuda.h"

/**
 * Kernel: 1D Convolution (窗口大小 3)
 * 
 * @param input  Input array (至少 lane_count + 2 個元素)
 * @param kernel Convolution kernel (3 個元素)
 * @param output Output array (lane_count 個元素)
 * 
 * 每個 lane 計算:
 * output[i] = input[i]*kernel[0] + input[i+1]*kernel[1] + input[i+2]*kernel[2]
 */
__global__ void conv1d_kernel3(int* input, int* kernel, int* output) {
    int lane = laneId();
    
    // 讀取輸入數據 (sliding window)
    int i0 = input[lane];      // Input[i]
    int i1 = input[lane + 1];  // Input[i+1]
    int i2 = input[lane + 2];  // Input[i+2]
    
    // 讀取 kernel 權重
    int k0 = kernel[0];
    int k1 = kernel[1];
    int k2 = kernel[2];
    
    // 計算卷積 (MAC - Multiply-Accumulate)
    int mac0 = i0 * k0;
    int mac1 = i1 * k1;
    int mac2 = i2 * k2;
    
    int result = mac0 + mac1 + mac2;
    
    // 寫回結果
    output[lane] = result;
}

/**
 * Kernel: 1D Convolution with Manual Memory Access
 * 
 * 這個版本展示如何手動控制記憶體存取
 * (當編譯器尚未完全實作時的替代方案)
 */
__global__ void conv1d_manual(unsigned int input_base,
                               unsigned int kernel_base, 
                               unsigned int output_base) {
    int lane = laneId();
    
    // 計算輸入位址 (4 bytes per int)
    unsigned int i0_addr = input_base + lane * 4;
    unsigned int i1_addr = input_base + (lane + 1) * 4;
    unsigned int i2_addr = input_base + (lane + 2) * 4;
    
    // 使用 intrinsic 讀取
    int i0 = __mcuda_vram_read_int(i0_addr);
    int i1 = __mcuda_vram_read_int(i1_addr);
    int i2 = __mcuda_vram_read_int(i2_addr);
    
    // 讀取 kernel
    int k0 = __mcuda_vram_read_int(kernel_base + 0);
    int k1 = __mcuda_vram_read_int(kernel_base + 4);
    int k2 = __mcuda_vram_read_int(kernel_base + 8);
    
    // MAC
    int result = (i0 * k0) + (i1 * k1) + (i2 * k2);
    
    // 寫回
    unsigned int out_addr = output_base + lane * 4;
    __mcuda_vram_write_int(out_addr, result);
}
