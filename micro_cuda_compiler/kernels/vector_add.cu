/*
 * Example Kernel: Vector Addition
 * 
 * This is the reference kernel for the Micro-CUDA compiler project.
 * It demonstrates basic SIMT parallel programming.
 * 
 * Computation: C[i] = A[i] + B[i] for i in 0..7 (8 lanes)
 */

#include "../mcuda.h"

/**
 * Kernel: Vector Addition
 * 
 * @param A Input vector A (must have at least 8 elements)
 * @param B Input vector B (must have at least 8 elements)
 * @param C Output vector C (must have space for at least 8 elements)
 * 
 * Each of the 8 SIMD lanes processes one element in parallel.
 * 
 * Memory Layout (expected by run_kernel.py):
 * - A: VRAM[0x0000-0x001F] (8 elements * 4 bytes)
 * - B: VRAM[0x0020-0x003F] (8 elements * 4 bytes)
 * - C: VRAM[0x0040-0x005F] (8 elements * 4 bytes)
 */
__global__ void vectorAdd(int* A, int* B, int* C) {
    // Get current lane ID (0-7)
    int idx = laneId();
    
    // Each lane loads its element using SIMT addressing
    // Hardware automatically computes: addr = base + laneId * 4
    int a = A[idx];
    int b = B[idx];
    
    // Perform addition (all 8 lanes compute in parallel)
    int c = a + b;
    
    // Each lane stores its result
    C[idx] = c;
}

/**
 * Kernel: Vector Addition with Manual Addressing
 * 
 * This version explicitly uses SIMT intrinsics to demonstrate
 * the low-level memory operations.
 */
__global__ void vectorAddManual(int* A, int* B, int* C) {
    int lane = __mcuda_lane_id();
    
    // Calculate element-specific addresses
    // (In reality, the compiler should optimize A[lane] to this)
    unsigned int addr_a = (unsigned int)A + lane * 4;
    unsigned int addr_b = (unsigned int)B + lane * 4;
    unsigned int addr_c = (unsigned int)C + lane * 4;
    
    // Load using lane-based addressing
    int a = __mcuda_load_lane_int((unsigned int)A);
    int b = __mcuda_load_lane_int((unsigned int)B);
    
    // Compute
    int c = a + b;
    
    // Store using lane-based addressing
    __mcuda_store_lane_int((unsigned int)C, c);
}
