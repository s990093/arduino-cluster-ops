/*
 * VM Core v1.5 Implementation
 */

#include "vm_core.h"
#include "vm_simd_v15.h"
#include "vm_trace.h"
#include "instructions_v15.h"

void VMCore::run(SIMDEngineV15& simd, TraceUnit& trace) {
    resetVM();
    running = true;
    
    // Start trace capture
    if (trace.isStreamMode()) {
        trace.startProgram();
    }
    
    // Run until completion (-1)
    step(simd, trace, -1);
    
    // End trace capture
    if (trace.isStreamMode()) {
        trace.endProgram();
    }
}

void VMCore::step(SIMDEngineV15& simd, TraceUnit& trace, int count) {
    if (!running && PC == 0) running = true; // Ensure running if just started
    
    int executed = 0;
    
    while (running && !halted && PC < program_length) {
        // Check step limit
        if (count > 0 && executed >= count) {
            break;
        }
        
        // Fetch
        uint32_t inst_word = fetch();
        
        // Decode
        Instruction inst;
        inst.decode(inst_word);
        
        // Trace before execution (Registers) - PRE-EXECUTION STATE
        if (trace.isStreamMode()) {
            trace.beginInstructionRecord(cycle_count, PC, inst_word, simd, inst);
        }
        
        // Execute on SIMD engine
        simd.execute(inst);

        // Trace after execution (Memory Access) - POST-EXECUTION EVENTS
        if (trace.isStreamMode()) {
            trace.endInstructionRecord(simd);
        }
        
        // Check for halt
        if (inst.opcode == OP_EXIT) {
            halted = true;
            running = false;
        }
        
        // Branch Logic
        if (inst.opcode == OP_BRA) {
            // Unconditional Branch (Absolute in DEST field)
            PC = inst.dest;
        }
        else if (inst.opcode == OP_BRZ) {
            // Conditional Branch (Branch if P0 is Zero/False)
            // Using Lane 0 P0 for control flow (Scalar approximation)
            if (simd.warp_state.P[0] == 0) {
                PC = inst.dest;
            } else {
                PC++;
            }
        }
        else {
            // Advance PC for non-branch instructions
            PC++;
        }
        
        cycle_count++;
        executed++;
        
        // Safety limit (only for run mode)
        if (count < 0 && cycle_count > 100000) {
            Serial.println("⚠️  Cycle limit reached");
            break;
        }
    }
}
