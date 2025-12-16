/*
 * Conv1D Kernel - Manual Implementation
 * 
 * This is a hand-coded version showing the exact assembly we want.
 * Used for testing until the compiler can generate equivalent code.
 */

#include "../mcuda.h"

/**
 * Manual Conv1D implementation using explicit addressing
 * 
 * Memory Layout:
 * - Input:  VRAM[0x00] (0)
 * - Kernel: VRAM[0x40] (64)
 * - Output: VRAM[0x80] (128)
 */
__global__ void conv1d_manual(unsigned int input_base,
                               unsigned int kernel_base,
                               unsigned int output_base) {
    int lane = laneId();
    
    // Calculate addresses using lane ID
    // Each int is 4 bytes, so offset = lane * 4
    unsigned int lane_offset = lane * 4;
    
    // Output address for this lane
    unsigned int out_addr = output_base + lane_offset;
    
    // Read input window: I[lane], I[lane+1], I[lane+2]
    int i0 = __mcuda_vram_read_int(input_base + lane_offset);
    int i1 = __mcuda_vram_read_int(input_base + lane_offset + 4);
    int i2 = __mcuda_vram_read_int(input_base + lane_offset + 8);
    
    // Read kernel weights: K[0], K[1], K[2]
    int k0 = __mcuda_vram_read_int(kernel_base + 0);
    int k1 = __mcuda_vram_read_int(kernel_base + 4);
    int k2 = __mcuda_vram_read_int(kernel_base + 8);
    
    // Compute MAC: result = i0*k0 + i1*k1 + i2*k2
    int mac0 = i0 * k0;
    int mac1 = i1 * k1;
    int mac2 = i2 * k2;
    
    int sum01 = mac0 + mac1;
    int result = sum01 + mac2;
    
    // Write result
    __mcuda_vram_write_int(out_addr, result);
}
