#include "../micro_cuda_compiler/mcuda.h"

// 1D Convolution Kernel
// Arguments stored in VRAM:
// - input: Input array
// - kernel_weights: Convolution kernel (3 elements)
// - output: Output array
// - params: [start_offset]

__global__ void conv1d(int* input, int* kernel_weights, int* output, int* params) {
    int lane = laneId();
    int start_offset = params[0]; 
    int idx = start_offset + lane;
    
    // Read kernel weights (1D, size 3)
    int k0 = kernel_weights[0];
    int k1 = kernel_weights[1];
    int k2 = kernel_weights[2];
    
    // Read input (centered window)
    // Neighborhood: idx-1, idx, idx+1
    int val_left = input[idx - 1];
    int val_center = input[idx];
    int val_right = input[idx + 1];
    
    // Convolve
    int result = (val_left * k0) + (val_center * k1) + (val_right * k2);
    
    // Write result
    output[idx] = result;
}
