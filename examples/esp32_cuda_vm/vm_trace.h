/*
 * Trace Unit v1.5 - Enhanced Performance Trace
 * 
 * Outputs detailed JSON trace with hardware context and performance metrics
 */

#ifndef VM_TRACE_V15_H
#define VM_TRACE_V15_H

#include <Arduino.h>

// Forward declaration
class SIMDEngineV15;
struct Instruction;

class TraceUnit {
private:
    bool enabled;
    bool stream_mode;
    uint32_t trace_count;
    uint32_t total_instructions;
    unsigned long program_start_time;
    
public:
    TraceUnit() {
        reset();
    }
    
    void reset() {
        enabled = false;
        stream_mode = false;
        trace_count = 0;
        total_instructions = 0;
        program_start_time = 0;
    }
    
    void setEnabled(bool enable) { enabled = enable; }
    void setStreamMode(bool enable) { stream_mode = enable; }
    
    bool isEnabled() const { return enabled; }
    bool isStreamMode() const { return stream_mode; }
    
    void startProgram();
    void endProgram();
    
    // Split trace recording
    void beginInstructionRecord(uint64_t cycle, uint32_t pc, uint32_t inst_word, 
                                SIMDEngineV15& simd, const Instruction& inst);
    void endInstructionRecord(SIMDEngineV15& simd);
    
private:
    const char* getOpcodeName(uint8_t opcode);
    void printLaneData(SIMDEngineV15& simd, int lane_id);
};

#endif // VM_TRACE_V15_H
