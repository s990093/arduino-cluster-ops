/*
 * Trace Unit v1.5 - Enhanced Performance Trace Implementation
 */

#include "vm_trace.h"
#include "vm_simd_v15.h"
#include "vm_core.h"
#include "instructions_v15.h"

void TraceUnit::startProgram() {
    if (!stream_mode) return;
    
    program_start_time = micros();
    
    Serial.println("{");
    Serial.println("  \"trace_version\": \"2.1\",");
    Serial.println("  \"architecture\": \"SIMT\",");
    Serial.println("  \"program\": \"GPU-Like Kernel\",");
    Serial.println("  \"warp_size\": 8,"); // Reverted to 8
    Serial.println("  \"records\": [");
    
    trace_count = 0;
}

void TraceUnit::endProgram() {
    if (!stream_mode) return;
    
    Serial.println();
    Serial.println("  ],");
    Serial.print("  \"total_instructions\": ");
    Serial.println(trace_count);
    Serial.println("}");
}

const char* TraceUnit::getOpcodeName(uint8_t opcode) {
    switch (opcode) {
        case 0x00: return "NOP";
        case 0x01: return "EXIT";
        case 0x10: return "MOV";
        case 0x11: return "IADD";
        case 0x12: return "ISUB";
        case 0x13: return "IMUL";
        case 0x18: return "AND"; // Added missing opcodes names if helpful
        case 0x19: return "OR";
        case 0x1A: return "XOR";
        case 0x30: return "FADD";
        case 0x31: return "FSUB";
        case 0x32: return "FMUL";
        case 0x60: return "LDG";
        case 0x61: return "STG";
        case 0x64: return "LDX";
        case 0x65: return "LDL";
        case 0x66: return "STX";
        case 0x67: return "STL";
        case 0xF0: return "S2R";
        case 0xF2: return "TRACE";
        default: return "UNKNOWN";
    }
}

void TraceUnit::printLaneData(SIMDEngineV15& simd, int lane_id) {
    // SoA Access
    
    Serial.print("        {");
    Serial.print("\"lane_id\": ");
    Serial.print(lane_id);
    Serial.print(", \"sr_laneid\": ");
    Serial.print(simd.warp_state.SR.laneid[lane_id]);
    
    // Output all 32 R registers
    Serial.print(", \"R\": [");
    for (int i = 0; i < 32; i++) {
        if (i > 0) Serial.print(", ");
        Serial.print(simd.warp_state.R[i][lane_id]);
    }
    Serial.print("]");
    
    // Output all 32 F registers
    Serial.print(", \"F\": [");
    for (int i = 0; i < 32; i++) {
        if (i > 0) Serial.print(", ");
        Serial.print(simd.warp_state.F[i][lane_id], 6);
    }
    Serial.print("]");
    
    // Output P (Predicate) - Assuming P[lane] holds the predicate
    // Only outputting single value as P is now simplified per lane? 
    // Or if we want to show it as array to match JSON format: [P]
    Serial.print(", \"P\": [");
    Serial.print(simd.warp_state.P[lane_id]); 
    Serial.print("]");
    
    Serial.print("}");
}

void TraceUnit::beginInstructionRecord(uint64_t cycle, uint32_t pc, uint32_t inst_word,
                                  SIMDEngineV15& simd, const Instruction& inst) {
    if (!stream_mode) return;
    
    // Print comma except for first record
    if (trace_count > 0) {
        Serial.println(",");
    }
    
    unsigned long exec_time = micros() - program_start_time;
    
    // Start record
    Serial.print("    {");
    
    // Basic info
    Serial.print("\"cycle\": ");
    Serial.print(cycle);
    Serial.print(", \"pc\": ");
    Serial.print(pc);
    Serial.print(", \"instruction\": \"0x");
    Serial.print(inst_word, HEX);
    Serial.print("\"");
    
    // ASM representation
    Serial.print(", \"asm\": \"");
    Serial.print(getOpcodeName(inst.opcode));
    Serial.print(" dest=");
    Serial.print(inst.dest);
    Serial.print(" src1=");
    Serial.print(inst.src1);
    Serial.print(" src2=");
    Serial.print(inst.src2_imm);
    Serial.print("\"");
    
    // Execution time
    Serial.print(", \"exec_time_us\": ");
    Serial.print(exec_time);
    
    // Hardware Context
    Serial.print(", \"hw_ctx\": {");
    Serial.print("\"sm_id\": 0, ");
    Serial.print("\"warp_id\": 0, ");
    Serial.print("\"active_mask\": \"0x000000FF\""); // 8 lanes active
    Serial.print("}");
    
    // Performance Info
    Serial.print(", \"perf\": {");
    Serial.print("\"latency\": 1, ");
    Serial.print("\"stall_cycles\": 0, ");
    Serial.print("\"stall_reason\": \"NONE\", ");
    Serial.print("\"pipe_stage\": \"EXEC\", ");
    Serial.print("\"core_id\": 1, ");
    Serial.print("\"predicate_masked\": false, ");
    Serial.print("\"sync_barrier\": ");
    Serial.print(inst.opcode == 0x05 ? "true" : "false");
    Serial.print(", \"simd_width\": 8");
    Serial.print("}");
    
    // Lane data (Pre-Execution State)
    Serial.print(", \"lanes\": [");
    Serial.println();
    
    for (int lane = 0; lane < 8; lane++) { // Dump 8 lanes
        if (lane > 0) {
            Serial.println(",");
        }
        printLaneData(simd, lane);
        Serial.flush(); 
        delay(10); // Throttle
        // Note: Dumping 32 lanes via Serial is VERY slow. 
        // Recommend trace:off during performance testing.
    }
    
    Serial.println();
    Serial.print("      ]");
}

void TraceUnit::endInstructionRecord(SIMDEngineV15& simd) {
    if (!stream_mode) return;

    // Memory Access Info
    if (simd.memory_access_count > 0) {
        Serial.print(", \"memory_access\": [");
        for (int i = 0; i < simd.memory_access_count; i++) {
            if (i > 0) Serial.print(", ");
            Serial.print("{ \"lane\": ");
            Serial.print(simd.memory_accesses[i].lane);
            Serial.print(", \"type\": \"");
            Serial.print(simd.memory_accesses[i].type);
            Serial.print("\", \"addr\": ");
            Serial.print(simd.memory_accesses[i].addr);
            Serial.print(", \"val\": ");
            Serial.print(simd.memory_accesses[i].val);
            Serial.print(" }");
        }
        Serial.print("]");
    } else {
         Serial.print(", \"memory_access\": []");
    }

    Serial.print("}");
    Serial.flush();
    delay(10);
    
    trace_count++;
}
