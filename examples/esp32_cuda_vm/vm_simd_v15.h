/*
 * VM SIMD Engine v1.5 - True SIMT Architecture
 * 
 * Optimized for Speed:
 * - SoA Memory Layout (WarpState)
 * - 32 Lanes Support
 * - Fast Approximations
 */

#ifndef VM_SIMD_V15_H
#define VM_SIMD_V15_H

#include <Arduino.h>
#include "vm_config.h"

// Forward declaration
struct Instruction;

// // ===== Optimized SoA Layout (Level 2) =====
// [ Host / Monitor ]
//         |
//         v
// [ Global VRAM ]  <--- mem addr val（你這段）
//         |
//         v
// [ Warp Execution ]
//    ├─ Registers (R/F)
//    ├─ Predicates
//    └─ Shared Memory 



// 0x00000000 - 0x0FFFFFFF : Global Memory (VRAM)
// 0x10000000 - 0x10000FFF : Shared Memory (warp-local)
// 0x20000000 - 0x2000FFFF : Local Memory (lane-local)


struct WarpState {
    // Registers: R[RegIndex][LaneIndex]
    // 32 Registers, 8 Lanes
    uint32_t R[32][8]; 
    
    // Float Registers: F[RegIndex][LaneIndex]
    float F[32][8];
    
    // Predicates: P[LaneIndex]
    uint8_t P[8]; 
    
    // Shared Memory: [Lane][Offset]
    uint8_t shared_mem[8][256];
    
     // System Registers State
    struct {
        uint32_t tid[8];
        uint32_t laneid[8];
        uint32_t warpsize;
    } SR;

    void reset() {
        memset(R, 0, sizeof(R));
        memset(F, 0, sizeof(F));
        memset(P, 0, sizeof(P));
        memset(shared_mem, 0, sizeof(shared_mem));
        for(int i=0; i<8; i++) {
             SR.laneid[i] = i;
             SR.tid[i] = i; // Default TID = LaneID
        }
        SR.warpsize = 8; // Default to 8
    }
};

// ===== Memory Access Record =====
struct MemoryAccess {
    uint8_t lane;
    const char* type;      // "read" or "write"
    uint32_t addr;
    uint32_t val;
    
    MemoryAccess() : lane(0), type("none"), addr(0), val(0) {}
    
    MemoryAccess(uint8_t l, const char* t, uint32_t a, uint32_t v) 
        : lane(l), type(t), addr(a), val(v) {}
};

// ===== SIMD Engine with SIMT Support =====
class SIMDEngineV15 {
public:
    // Optimized Warp State (SoA)
    WarpState warp_state;
    
    // Global VRAM (shared across all lanes)
    // For large VRAM (>64KB), we allocate in PSRAM if available
    uint8_t* vram;
    
    // Shared state
    bool halted;
    uint32_t warp_size;
    
    

    // Memory Tracking for Trace
    MemoryAccess memory_accesses[16];
    uint8_t memory_access_count;
    
    SIMDEngineV15() {
        // Allocate VRAM in PSRAM if available, otherwise heap
        vram = (uint8_t*)heap_caps_malloc(VM_VRAM_SIZE, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!vram) {
            // Fallback to regular heap if PSRAM not available
            vram = (uint8_t*)malloc(VM_VRAM_SIZE);
        }
        
        // Critical: Ensure allocation succeeded
        if (!vram) {
            Serial.println("FATAL: VRAM allocation failed!");
            Serial.printf("  Requested: %d bytes\n", VM_VRAM_SIZE);
            Serial.println("  Reduce VM_VRAM_SIZE in vm_config.h");
            while(1) { delay(1000); } // Halt
        }
        
        reset();
    }
    
    ~SIMDEngineV15() {
        if (vram) {
            free(vram);
        }
    }
    
    void softReset() {
        warp_state.reset();
        halted = false;
        warp_size = 32; // Upgrade to 32
        memory_access_count = 0;
    }

    void reset() {
        Serial.printf("VM_VRAM_SIZE: %d\n", VM_VRAM_SIZE);
        softReset();
        memset(vram, 0, VM_VRAM_SIZE);
    }
    
    // Memory Access Tracking (for Trace)
    void clearMemoryAccesses() {
        memory_access_count = 0;
    }

    // Level 1: Wrapped logging
    void addMemoryAccess(uint8_t lane, const char* type, uint32_t addr, uint32_t val) {
        #ifdef DEBUG_TRACE
        if (memory_access_count < 16) {
            memory_accesses[memory_access_count++] = MemoryAccess(lane, type, addr, val);
        }
        #endif
    }
    
    // ===== SIMD Execution Interface =====
    void execute(const Instruction& inst);
    
    // Execution by category
    void executeInteger(const Instruction& inst);
    void executeFloat(const Instruction& inst);
    void executeMemory(const Instruction& inst);    // NEW: SIMT memory ops
    void executeControl(const Instruction& inst);
    void executeSFU(const Instruction& inst);
    void executeSystem(const Instruction& inst);     // NEW: S2R/R2S
    
    // Helper: Read system register
    uint32_t readSystemReg(uint8_t lane_id, uint8_t sr_index);
    
    // Wrapper to get register value for external debugger (slow)
    uint32_t getReg(int lane_id, int reg_id) {
        if (lane_id < 32 && reg_id < 32) return warp_state.R[reg_id][lane_id];
        return 0;
    }
};

#endif // VM_SIMD_V15_H
