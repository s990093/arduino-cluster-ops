/*
 * Micro-CUDA Runtime Header
 * 
 * Purpose: Provide CUDA-like programming interface for Micro-CUDA ISA v1.5
 * Target: ESP32 CUDA VM with 8-lane SIMD architecture
 * 
 * This header allows writing CUDA-style C++ code that will be compiled
 * to Micro-CUDA ISA via LLVM IR transformation.
 */

#ifndef MCUDA_H
#define MCUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// ===== CUDA Kernel Qualifiers =====
#define __global__
#define __device__
#define __host__
#define __shared__

// ===== Built-in Types =====
struct Dim3 {
    unsigned int x, y, z;
};

// ===== Built-in Variables (Thread Indexing) =====
// These will be provided by the runtime
extern struct Dim3 threadIdx;
extern struct Dim3 blockIdx;
extern struct Dim3 blockDim;
extern struct Dim3 gridDim;

// ===== Intrinsic Functions =====
// These are recognized by the Micro-CUDA compiler backend
// and translated to corresponding ISA instructions

/**
 * Get current lane ID (0-7)
 * Maps to: S2R Rd, SR_LANEID
 */
int __mcuda_lane_id(void);

/**
 * Get warp size (always 8 for Micro-CUDA)
 * Maps to: S2R Rd, SR_WARPSIZE
 */
int __mcuda_warp_size(void);

/**
 * Get thread ID
 * Maps to: S2R Rd, SR_TID
 */
int __mcuda_thread_id(void);

/**
 * Get block/CTA ID
 * Maps to: S2R Rd, SR_CTAID
 */
int __mcuda_block_id(void);

/**
 * Synchronize all lanes in the warp
 * Maps to: BAR.SYNC
 */
void __mcuda_syncthreads(void);

/**
 * Write to VRAM (Global Memory)
 * Maps to: STG [addr], val
 */
void __mcuda_vram_write_int(unsigned int addr, int val);
void __mcuda_vram_write_float(unsigned int addr, float val);

/**
 * Read from VRAM (Global Memory)
 * Maps to: LDG Rd, [addr]
 */
int __mcuda_vram_read_int(unsigned int addr);
float __mcuda_vram_read_float(unsigned int addr);

/**
 * Lane-based SIMT load
 * Maps to: LDL Rd, [base]
 * Each lane loads from: base + laneId * 4
 */
int __mcuda_load_lane_int(unsigned int base);
float __mcuda_load_lane_float(unsigned int base);

/**
 * Lane-based SIMT store
 * Maps to: STL [base], val
 * Each lane stores to: base + laneId * 4
 */
void __mcuda_store_lane_int(unsigned int base, int val);
void __mcuda_store_lane_float(unsigned int base, float val);

/**
 * Emit trace marker for debugging
 * Maps to: TRACE imm
 */
void __mcuda_trace(int marker);

// ===== Convenience Macros =====
#define laneId() __mcuda_lane_id()
#define warpSize() __mcuda_warp_size()
#define threadId() __mcuda_thread_id()
#define blockId() __mcuda_block_id()

// Synchronization
#define __syncthreads() __mcuda_syncthreads()

// Memory Aliases (for readability)
#define vram_write_int(addr, val) __mcuda_vram_write_int(addr, val)
#define vram_write_float(addr, val) __mcuda_vram_write_float(addr, val)
#define vram_read_int(addr) __mcuda_vram_read_int(addr)
#define vram_read_float(addr) __mcuda_vram_read_float(addr)

// ===== Math Functions (SFU - Special Function Unit) =====
// These will be mapped to SFU instructions

/**
 * Reciprocal: 1.0 / x
 * Maps to: SFU.RCP
 */
float __mcuda_rcp(float x);

/**
 * Square root
 * Maps to: SFU.SQRT
 */
float __mcuda_sqrt(float x);

/**
 * Exponential: e^x
 * Maps to: SFU.EXP
 */
float __mcuda_exp(float x);

/**
 * GELU activation function
 * Maps to: SFU.GELU
 */
float __mcuda_gelu(float x);

/**
 * ReLU activation: max(0, x)
 * Maps to: SFU.RELU
 */
float __mcuda_relu(float x);

// Math aliases
#define mcuda_rcp(x) __mcuda_rcp(x)
#define mcuda_sqrt(x) __mcuda_sqrt(x)
#define mcuda_exp(x) __mcuda_exp(x)
#define mcuda_gelu(x) __mcuda_gelu(x)
#define mcuda_relu(x) __mcuda_relu(x)

#ifdef __cplusplus
}
#endif

#endif // MCUDA_H
