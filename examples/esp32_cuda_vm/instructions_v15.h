/*
 * Micro-CUDA ISA v1.5 Instruction Set
 * 
 * Architecture: True SIMT (Single Instruction, Multiple Threads)
 * Core Feature: Lane-Awareness with SR_LANEID
 * 
 * Encoding: 32-bit Fixed
 * [31:24 OPCODE] [23:16 DEST] [15:8 SRC1] [7:0 SRC2/IMM]
 */

#ifndef INSTRUCTIONS_V15_H
#define INSTRUCTIONS_V15_H

// ===== Group 1: System Control (0x00-0x0F) =====
#define OP_NOP          0x00  // No operation
#define OP_EXIT         0x01  // Terminate kernel
#define OP_BRA          0x02  // Unconditional branch
#define OP_BRZ          0x03  // Branch if zero (conditional)
#define OP_BAR_SYNC     0x05  // Warp barrier synchronization
#define OP_YIELD        0x07  // Yield time slice

// ===== Group 2: Integer ALU (0x10-0x2F) =====
#define OP_MOV          0x10  // Move immediate
#define OP_IADD         0x11  // Integer add
#define OP_ISUB         0x12  // Integer subtract
#define OP_IMUL         0x13  // Integer multiply
#define OP_IDIV         0x14  // Integer divide
#define OP_AND          0x17  // Bitwise AND
#define OP_OR           0x18  // Bitwise OR
#define OP_XOR          0x19  // Bitwise XOR
#define OP_ISETP_EQ     0x1A  // Set predicate if equal
#define OP_ISETP_GT     0x1C  // Set predicate if greater than
#define OP_SHL          0x1D  // Shift left
#define OP_SHR          0x1E  // Shift right

// ===== Group 3: Floating Point & AI (0x30-0x5F) =====
#define OP_FADD         0x30  // FP32 add
#define OP_FSUB         0x31  // FP32 subtract
#define OP_FMUL         0x32  // FP32 multiply
#define OP_FDIV         0x33  // FP32 divide
#define OP_FFMA         0x34  // FP32 fused multiply-add
#define OP_HMMA_I8      0x40  // 4-way SIMD INT8 dot product
#define OP_SFU_RCP      0x50  // Reciprocal (1/x)
#define OP_SFU_SQRT     0x51  // Square root
#define OP_SFU_EXP      0x52  // Exponential
#define OP_SFU_GELU     0x53  // GELU activation
#define OP_SFU_RELU     0x54  // ReLU activation

// ===== Group 4: Memory & SIMT Addressing (0x60-0x7F) =====
// Uniform (Broadcast) Operations
#define OP_LDG          0x60  // Load Global (Uniform/Broadcast)
#define OP_STG          0x61  // Store Global (Uniform)
#define OP_LDS          0x62  // Load Shared (Local RAM)
#define OP_STS          0x63  // Store Shared

// SIMT Operations (NEW in v1.5)
#define OP_LDX          0x64  // Indexed SIMT Load: [Ra + Rb]
#define OP_LDL          0x65  // Lane-Based Load: [Ra + SR_LANEID]
#define OP_STX          0x66  // Indexed SIMT Store: [Ra + Rb]
#define OP_STL          0x67  // Lane-Based Store: [Ra + SR_LANEID]

// Atomic Operations
#define OP_ATOM_ADD     0x70  // Atomic add
#define OP_ATOM_CAS     0x71  // Atomic compare-and-swap

// ===== Group 5: System Registers (0xF0-0xFF) =====
#define OP_S2R          0xF0  // System to Register
#define OP_R2S          0xF1  // Register to System
#define OP_TRACE        0xF2  // Trace marker

// ===== System Register Indices =====
#define SR_TID          0     // Thread ID (Physical Core ID)
#define SR_CTAID        1     // Block ID
#define SR_LANEID       2     // Lane ID (NEW in v1.5)
#define SR_WARPSIZE     3     // Warp Size
#define SR_GPU_UTIL     6     // GPU Utilization
#define SR_WARP_ID      8     // Warp ID
#define SR_SM_ID        9     // SM ID

// ===== Predicate Register Count =====
#define NUM_PREDICATES  8

// ===== Memory Size Definitions =====
// Moved to vm_config.h
// #define VRAM_SIZE       4096
// #define SHARED_SIZE     256

#endif // INSTRUCTIONS_V15_H
