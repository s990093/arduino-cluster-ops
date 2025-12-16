/*
 * VM Core v1.5 - Simplified for SIMT Architecture
 * 
 * Core responsibilities:
 * - Instruction fetch and decode
 * - PC management  
 * - Program loading
 * - Coordination with SIMD engine
 */

#ifndef VM_CORE_V15_H
#define VM_CORE_V15_H

#include <Arduino.h>
#include "vm_config.h"




// Forward declarations
class SIMDEngineV15;
class TraceUnit;

// ===== Instruction Structure =====
struct Instruction {
    uint8_t opcode;
    uint8_t dest;
    uint8_t src1;
    uint8_t src2_imm;
    
    void decode(uint32_t word) {
        opcode = (word >> 24) & 0xFF;
        dest = (word >> 16) & 0xFF;
        src1 = (word >> 8) & 0xFF;
        src2_imm = word & 0xFF;
    }
    
    uint32_t encode() const {
        return ((uint32_t)opcode << 24) | 
               ((uint32_t)dest << 16) | 
               ((uint32_t)src1 << 8) | 
               (uint32_t)src2_imm;
    }
};

// ===== VM Core (Front-End) =====
class VMCore {
private:
    uint32_t program[VM_PROGRAM_SIZE];
    size_t program_length;
    
    uint32_t PC;
    bool running;
    bool halted;
    uint64_t cycle_count;
    
public:
    VMCore() {
        init();
    }
    
    void init() {
        memset(program, 0, VM_PROGRAM_SIZE * sizeof(uint32_t));
        resetVM();
        program_length = 0; // This line was moved from the original init()
    }
    
    void resetVM() {
        PC = 0;
        running = false;
        halted = false;
        cycle_count = 0;
    }
    
    // Load single instruction
    bool loadInstruction(uint32_t inst_word) {
        if (program_length >= VM_PROGRAM_SIZE) {
            return false;
        }
        program[program_length++] = inst_word;
        return true;
    }
    
    // Fetch instruction at PC
    uint32_t fetch() {
        if (PC >= program_length) {
            return 0x01000000;  // EXIT
        }
        return program[PC];
    }
    
    // Execute program
    void run(SIMDEngineV15& simd, TraceUnit& trace);
    
    // Step execution
    void step(SIMDEngineV15& simd, TraceUnit& trace, int count = 1);
    
    // Getters
    uint32_t getPC() const { return PC; }
    void setPC(uint32_t pc) { PC = pc; }
    void incPC() { PC++; }
    
    uint32_t getInstruction(uint32_t addr) const {
        if (addr >= program_length) return 0x01000000; // EXIT
        return program[addr];
    }
    size_t getProgramLength() const { return program_length; }

    bool isHalted() const { return halted; }
    void setHalted(bool h) { halted = h; }
    uint64_t getCycleCount() const { return cycle_count; }
    void incCycleCount() { cycle_count++; }

    // High-Speed Loader / Flash Kernel Support
    uint32_t* getProgramMemoryPtr() { return program; }
    void setProgramLength(size_t len) { 
        if (len <= VM_PROGRAM_SIZE) program_length = len;
        else program_length = VM_PROGRAM_SIZE;
    }
};

#endif // VM_CORE_V15_H
