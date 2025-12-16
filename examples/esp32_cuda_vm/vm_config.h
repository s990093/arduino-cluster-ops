#ifndef VM_CONFIG_H
#define VM_CONFIG_H

// ===== Memory Configuration =====

// Program Memory (Number of Instructions)
// Each instruction is 32-bit (Instruction struct is larger in RAM, approx 4 bytes raw + decoding overhead)
// 1024 instructions
#define VM_PROGRAM_SIZE 1024 * 2

// Video RAM / Global Memory (Bytes)
// 32KB for standard ESP32 (without PSRAM)
// For ESP32 with PSRAM, you can increase to 102400 (100KB) or more
#define VM_VRAM_SIZE 65536

// Shared Memory Size (Per Lane or Per Warp?)
// N-ISA v1.5 defines Shared Memory Region, usually per-warp or per-SM
#define VM_SHARED_SIZE 256 // Per-lane addressable range in current SIMD implementation

// ===== System Configuration =====

// FreeRTOS Task Stack Size (Bytes)
#define VM_STACK_SIZE 20480  // 20KB for LZ4 2KB chunks (~4.8KB buffers)

// Instruction Queue Size (Number of Batches)
#define VM_QUEUE_SIZE  32

// Batch Size (Instructions per Queue Item)
#define VM_BATCH_SIZE 32

// Serial Baud Rate
// Serial Baud PORT = "/dev/cu.usbserial-589A0095521"
#define VM_BAUD_RATE 460800 // High Speed (921600 unstable) 

// Serial RX Buffer Size (Bytes) - Increased for Turbo Mode
#define VM_SERIAL_RX_SIZE 32768

// Serial Read Chunk Size (Bytes) - Tuned for 921600 baud stability
#define VM_SERIAL_BLOCK_READ_SIZE 1024

// CPU Frequency (MHz) - Locked for performance
#define VM_CPU_FREQ      240

// Debug Trace (Comment out for max performance)
// #define DEBUG_TRACE

#endif // VM_CONFIG_H
