/*
 * Image Identity Kernel (with scaling)
 * 
 * Verifies compiler correctness by performing:
 * output[lane] = input[lane] * kernel[4]
 */

#include "../micro_cuda_compiler/mcuda.h"

__global__ void image_conv_test(int* input, int* kernel, int* output, int width) {
    // Get lane ID (0-7)
    int lane = laneId();
    
    // Read center weight from kernel (index 4)
    // kernel pointer is param 1
    int weight = kernel[4];
    
    // Read pixel for this lane
    // input pointer assumes batch data loaded at 0x0000
    int pixel = input[lane];
    
    // Compute
    int result = pixel * weight;
    
    // Write result
    output[lane] = result;
}
