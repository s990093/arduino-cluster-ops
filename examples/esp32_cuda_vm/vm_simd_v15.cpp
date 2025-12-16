/*
 * VM SIMD Engine v1.5 Implementation
 * 
 * Implements True SIMT with SoA Layout and ASM Optimization (Integer only)
 */

#include "vm_simd_v15.h"
#include "vm_core.h"
#include "instructions_v15.h"
#include <math.h>
#include <Arduino.h>

// Helper Macro for Force Unrolling
#define UNROLL_8(OP) { OP(0); OP(1); OP(2); OP(3); OP(4); OP(5); OP(6); OP(7); }

// ===== Fast Approximations (Level 3) =====

static inline float fast_rsqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5F;
    x2 = number * 0.5F;
    y  = number;
    i  = * ( long * ) &y;
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );
    return y;
}

// Sigmoid Approximation for GELU
static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-1.702f * x));
}

// ===== ASM Accelerators (Integer Only) =====
// Level 6: Full Unroll (8 lanes) + Predicate Packing

// Integer Add: dest[] = src1[] + src2[] (8 lanes)
static inline void asm_warp_add(
    uint32_t* __restrict__ dest, 
    const uint32_t* __restrict__ src1, 
    const uint32_t* __restrict__ src2,
    uint8_t* __restrict__ P
) {
    uint32_t t0, t1, t2, t3, t4, t5, t6, t7;
    uint32_t p_accum;
    uint32_t zero = 0;

    __asm__ volatile (
        "movi %13, 0\n\t"

        // ================= BLOCK 1: Lanes 0-3 =================
        // Load Lanes 0, 1, 2, 3
        "l32i.n %4, %0, 0\n\t"   // t0 = src1[0]
        "l32i.n %5, %1, 0\n\t"   // t1 = src2[0]
        "l32i.n %6, %0, 4\n\t"   // t2 = src1[1]
        "l32i.n %7, %1, 4\n\t"   // t3 = src2[1]
        
        "add %4, %4, %5\n\t"     // t0 = res[0]
        "add %6, %6, %7\n\t"     // t2 = res[1]

        "l32i.n %8, %0, 8\n\t"   // t4 = src1[2]
        "l32i.n %9, %1, 8\n\t"   // t5 = src2[2]
        "l32i.n %10, %0, 12\n\t" // t6 = src1[3]
        "l32i.n %11, %1, 12\n\t" // t7 = src2[3]

        "add %8, %8, %9\n\t"     // t4 = res[2]
        "add %10, %10, %11\n\t"  // t6 = res[3]

        // Store Lanes 0-3
        "s32i.n %4, %2, 0\n\t"
        "s32i.n %6, %2, 4\n\t"
        "s32i.n %8, %2, 8\n\t"
        "s32i.n %10, %2, 12\n\t"

        // --- Predicate Packing (Lanes 0-3) ---
        "movi %12, 0\n\t"         // p_accum = 0
        "movnez %12, %4, %4\n\t"  // P[0]
        "movi %4, 1\n\t"          // t0 = 1
        "movnez %12, %4, %12\n\t"

        "movi %5, 0\n\t"
        "movnez %5, %4, %6\n\t"   // P[1]
        "slli %5, %5, 8\n\t"
        "or %12, %12, %5\n\t"

        "movi %5, 0\n\t"
        "movnez %5, %4, %8\n\t"   // P[2]
        "slli %5, %5, 16\n\t"
        "or %12, %12, %5\n\t"

        "movi %5, 0\n\t"
        "movnez %5, %4, %10\n\t"  // P[3]
        "slli %5, %5, 24\n\t"
        "or %12, %12, %5\n\t"

        "s32i.n %12, %3, 0\n\t"

        // ================= BLOCK 2: Lanes 4-7 =================
        // Load Lanes 4-7
        "l32i.n %4, %0, 16\n\t"  // t0 = src1[4]
        "l32i.n %5, %1, 16\n\t"  // t1 = src2[4]
        "l32i.n %6, %0, 20\n\t"  // t2 = src1[5]
        "l32i.n %7, %1, 20\n\t"  // t3 = src2[5]

        "add %4, %4, %5\n\t"
        "add %6, %6, %7\n\t"

        "l32i.n %8, %0, 24\n\t"  // t4 = src1[6]
        "l32i.n %9, %1, 24\n\t"  // t5 = src2[6]
        "l32i.n %10, %0, 28\n\t" // t6 = src1[7]
        "l32i.n %11, %1, 28\n\t" // t7 = src2[7]

        "add %8, %8, %9\n\t"
        "add %10, %10, %11\n\t"

        // Store Lanes 4-7
        "s32i.n %4, %2, 16\n\t"
        "s32i.n %6, %2, 20\n\t"
        "s32i.n %8, %2, 24\n\t"
        "s32i.n %10, %2, 28\n\t"

        // --- Predicate Packing (Lanes 4-7) ---
        "movi %12, 0\n\t"
        "movnez %12, %4, %4\n\t"
        "movi %4, 1\n\t"
        "movnez %12, %4, %12\n\t"

        "movi %5, 0\n\t"
        "movnez %5, %4, %6\n\t"
        "slli %5, %5, 8\n\t"
        "or %12, %12, %5\n\t"

        "movi %5, 0\n\t"
        "movnez %5, %4, %8\n\t"
        "slli %5, %5, 16\n\t"
        "or %12, %12, %5\n\t"

        "movi %5, 0\n\t"
        "movnez %5, %4, %10\n\t"
        "slli %5, %5, 24\n\t"
        "or %12, %12, %5\n\t"

        "s32i.n %12, %3, 4\n\t"

        : "+r"(src1), "+r"(src2), "+r"(dest), "+r"(P), 
          "=&r"(t0), "=&r"(t1), "=&r"(t2), "=&r"(t3), 
          "=&r"(t4), "=&r"(t5), "=&r"(t6), "=&r"(t7),
          "=&r"(p_accum), "=&r"(zero)
        : 
        : "memory"
    );
}

// Integer Sub: dest[] = src1[] - src2[]
static inline void asm_warp_sub(
    uint32_t* __restrict__ dest, 
    const uint32_t* __restrict__ src1, 
    const uint32_t* __restrict__ src2,
    uint8_t* __restrict__ P
) {
    uint32_t t0, t1, t2, t3, t4, t5, t6, t7;
    uint32_t p_accum;
    uint32_t zero = 0;

    __asm__ volatile (
        "movi %13, 0\n\t"
        // ================= BLOCK 1 =================
        "l32i.n %4, %0, 0\n\t"
        "l32i.n %5, %1, 0\n\t"
        "l32i.n %6, %0, 4\n\t"
        "l32i.n %7, %1, 4\n\t"
        "sub %4, %4, %5\n\t"
        "sub %6, %6, %7\n\t"
        "l32i.n %8, %0, 8\n\t"
        "l32i.n %9, %1, 8\n\t"
        "l32i.n %10, %0, 12\n\t"
        "l32i.n %11, %1, 12\n\t"
        "sub %8, %8, %9\n\t"
        "sub %10, %10, %11\n\t"
        "s32i.n %4, %2, 0\n\t"
        "s32i.n %6, %2, 4\n\t"
        "s32i.n %8, %2, 8\n\t"
        "s32i.n %10, %2, 12\n\t"
        
        // Packing P[0..3]
        "movi %12, 0\n\t"
        "movnez %12, %4, %4\n\t"
        "movi %4, 1\n\t"
        "movnez %12, %4, %12\n\t"
        "movi %5, 0\n\t"
        "movnez %5, %4, %6\n\t"
        "slli %5, %5, 8\n\t"
        "or %12, %12, %5\n\t"
        "movi %5, 0\n\t"
        "movnez %5, %4, %8\n\t"
        "slli %5, %5, 16\n\t"
        "or %12, %12, %5\n\t"
        "movi %5, 0\n\t"
        "movnez %5, %4, %10\n\t"
        "slli %5, %5, 24\n\t"
        "or %12, %12, %5\n\t"
        "s32i.n %12, %3, 0\n\t"

        // ================= BLOCK 2 =================
        "l32i.n %4, %0, 16\n\t"
        "l32i.n %5, %1, 16\n\t"
        "l32i.n %6, %0, 20\n\t"
        "l32i.n %7, %1, 20\n\t"
        "sub %4, %4, %5\n\t"
        "sub %6, %6, %7\n\t"
        "l32i.n %8, %0, 24\n\t"
        "l32i.n %9, %1, 24\n\t"
        "l32i.n %10, %0, 28\n\t"
        "l32i.n %11, %1, 28\n\t"
        "sub %8, %8, %9\n\t"
        "sub %10, %10, %11\n\t"
        "s32i.n %4, %2, 16\n\t"
        "s32i.n %6, %2, 20\n\t"
        "s32i.n %8, %2, 24\n\t"
        "s32i.n %10, %2, 28\n\t"

        // Packing P[4..7]
        "movi %12, 0\n\t"
        "movnez %12, %4, %4\n\t"
        "movi %4, 1\n\t"
        "movnez %12, %4, %12\n\t"
        "movi %5, 0\n\t"
        "movnez %5, %4, %6\n\t"
        "slli %5, %5, 8\n\t"
        "or %12, %12, %5\n\t"
        "movi %5, 0\n\t"
        "movnez %5, %4, %8\n\t"
        "slli %5, %5, 16\n\t"
        "or %12, %12, %5\n\t"
        "movi %5, 0\n\t"
        "movnez %5, %4, %10\n\t"
        "slli %5, %5, 24\n\t"
        "or %12, %12, %5\n\t"
        "s32i.n %12, %3, 4\n\t"

        : "+r"(src1), "+r"(src2), "+r"(dest), "+r"(P), 
          "=&r"(t0), "=&r"(t1), "=&r"(t2), "=&r"(t3), 
          "=&r"(t4), "=&r"(t5), "=&r"(t6), "=&r"(t7),
          "=&r"(p_accum), "=&r"(zero)
        : 
        : "memory"
    );
}

// ===== Main Execution Dispatcher (Computed Goto + Flattened) =====
IRAM_ATTR void SIMDEngineV15::execute(const Instruction& inst) {
    // Clear log
    clearMemoryAccesses();
    
    // -------------------------------------------------------------------------
    // COMPUTED GOTO DISPATCH TABLE
    // -------------------------------------------------------------------------
    static void* dispatch_table[256];
    static bool table_initialized = false;
    
    if (!table_initialized) {
        // Default all to UNKNOWN
        for(int i=0; i<256; i++) dispatch_table[i] = &&LABEL_UNKNOWN;
        
        dispatch_table[OP_NOP] = &&LABEL_OP_NOP;
        dispatch_table[OP_EXIT] = &&LABEL_OP_EXIT;
        dispatch_table[OP_BAR_SYNC] = &&LABEL_OP_BAR_SYNC;
        dispatch_table[OP_BRA] = &&LABEL_OP_BRA;
        dispatch_table[OP_BRZ] = &&LABEL_OP_BRZ;
        dispatch_table[OP_YIELD] = &&LABEL_OP_YIELD;
        
        dispatch_table[OP_MOV] = &&LABEL_OP_MOV;
        dispatch_table[OP_IADD] = &&LABEL_OP_IADD;
        dispatch_table[OP_ISUB] = &&LABEL_OP_ISUB;
        dispatch_table[OP_IMUL] = &&LABEL_OP_IMUL;
        dispatch_table[OP_IDIV] = &&LABEL_OP_IDIV;
        dispatch_table[OP_AND] = &&LABEL_OP_AND;
        dispatch_table[OP_OR] = &&LABEL_OP_OR;
        dispatch_table[OP_XOR] = &&LABEL_OP_XOR;
        dispatch_table[OP_SHL] = &&LABEL_OP_SHL;
        dispatch_table[OP_SHR] = &&LABEL_OP_SHR;
        dispatch_table[OP_ISETP_EQ] = &&LABEL_OP_ISETP_EQ;
        dispatch_table[OP_ISETP_GT] = &&LABEL_OP_ISETP_GT;
        
        dispatch_table[OP_FADD] = &&LABEL_OP_FADD;
        dispatch_table[OP_FSUB] = &&LABEL_OP_FSUB;
        dispatch_table[OP_FMUL] = &&LABEL_OP_FMUL;
        dispatch_table[OP_FDIV] = &&LABEL_OP_FDIV;
        dispatch_table[OP_FFMA] = &&LABEL_OP_FFMA;
        dispatch_table[OP_HMMA_I8] = &&LABEL_OP_HMMA_I8;
        
        dispatch_table[OP_SFU_RCP] = &&LABEL_OP_SFU_RCP;
        dispatch_table[OP_SFU_SQRT] = &&LABEL_OP_SFU_SQRT;
        dispatch_table[OP_SFU_EXP] = &&LABEL_OP_SFU_EXP;
        dispatch_table[OP_SFU_GELU] = &&LABEL_OP_SFU_GELU;
        dispatch_table[OP_SFU_RELU] = &&LABEL_OP_SFU_RELU;
        
        dispatch_table[OP_LDG] = &&LABEL_OP_LDG;
        dispatch_table[OP_STG] = &&LABEL_OP_STG;
        dispatch_table[OP_LDS] = &&LABEL_OP_LDS;
        dispatch_table[OP_STS] = &&LABEL_OP_STS;
        dispatch_table[OP_LDX] = &&LABEL_OP_LDX;
        dispatch_table[OP_STX] = &&LABEL_OP_STX;
        dispatch_table[OP_LDL] = &&LABEL_OP_LDL;
        dispatch_table[OP_STL] = &&LABEL_OP_STL;
        
        dispatch_table[OP_S2R] = &&LABEL_OP_S2R;
        dispatch_table[OP_R2S] = &&LABEL_OP_R2S;
        dispatch_table[OP_TRACE] = &&LABEL_OP_TRACE;
        
        table_initialized = true;
    }

    // DISPATCH
    goto *dispatch_table[inst.opcode];
    
    // -------------------------------------------------------------------------
    // LABELS & HANDLERS
    // -------------------------------------------------------------------------

    LABEL_UNKNOWN:
    LABEL_OP_NOP: 
        return;
    
    LABEL_OP_EXIT: 
        halted = true; 
        return;
    
    LABEL_OP_BAR_SYNC: 
        return; // Barrier placeholder
    
    LABEL_OP_BRA: 
    LABEL_OP_BRZ: 
    LABEL_OP_YIELD:
        return; // Control flow placeholder

    // --- INTEGER ALU ---
    {
        uint8_t dest, src1, src2_reg;
        uint32_t *dest_ptr, *src1_ptr, *src2_ptr;

        // Use a macro to fetch operands for Integer ops blocks to avoid redefinition errors
        #define SETUP_INT_OPS \
            dest = inst.dest; \
            src1 = inst.src1; \
            src2_reg = inst.src2_imm; \
            dest_ptr = &warp_state.R[dest][0]; \
            src1_ptr = &warp_state.R[src1][0]; \
            src2_ptr = &warp_state.R[src2_reg][0];

    LABEL_OP_MOV:
        SETUP_INT_OPS
        if (src1 == 0 && src2_reg != 0) { 
            uint32_t imm = inst.src2_imm; 
            #define MOV_OP_IMM(i) dest_ptr[i] = imm
            UNROLL_8(MOV_OP_IMM);
            #undef MOV_OP_IMM
        } else {
            #define MOV_OP_REG(i) dest_ptr[i] = src1_ptr[i]
            UNROLL_8(MOV_OP_REG);
            #undef MOV_OP_REG
        }
        return;

    LABEL_OP_IADD:
        SETUP_INT_OPS
        asm_warp_add(dest_ptr, src1_ptr, src2_ptr, warp_state.P);
        return;

    LABEL_OP_ISUB:
        SETUP_INT_OPS
        asm_warp_sub(dest_ptr, src1_ptr, src2_ptr, warp_state.P);
        return;

    LABEL_OP_IMUL:
        SETUP_INT_OPS
        #define IMUL_OP(i) dest_ptr[i] = src1_ptr[i] * src2_ptr[i]
        UNROLL_8(IMUL_OP);
        #undef IMUL_OP
        return;

    LABEL_OP_IDIV:
        SETUP_INT_OPS
        #define IDIV_OP(i) if (src2_ptr[i] != 0) dest_ptr[i] = src1_ptr[i] / src2_ptr[i]; else dest_ptr[i] = 0xFFFFFFFF;
        UNROLL_8(IDIV_OP);
        #undef IDIV_OP
        return;

    LABEL_OP_AND:
        SETUP_INT_OPS
        #define AND_OP(i) dest_ptr[i] = src1_ptr[i] & src2_ptr[i]
        UNROLL_8(AND_OP);
        #undef AND_OP
        return;

    LABEL_OP_OR:
        SETUP_INT_OPS
        #define OR_OP(i) dest_ptr[i] = src1_ptr[i] | src2_ptr[i]
        UNROLL_8(OR_OP);
        #undef OR_OP
        return;

    LABEL_OP_XOR:
        SETUP_INT_OPS
        #define XOR_OP(i) dest_ptr[i] = src1_ptr[i] ^ src2_ptr[i]
        UNROLL_8(XOR_OP);
        #undef XOR_OP
        return;

    LABEL_OP_SHL:
        SETUP_INT_OPS
        #define SHL_OP(i) dest_ptr[i] = src1_ptr[i] << src2_ptr[i]
        UNROLL_8(SHL_OP);
        #undef SHL_OP
        return;

    LABEL_OP_SHR:
        SETUP_INT_OPS
        #define SHR_OP(i) dest_ptr[i] = src1_ptr[i] >> src2_ptr[i]
        UNROLL_8(SHR_OP);
        #undef SHR_OP
        return;

    LABEL_OP_ISETP_EQ:
        SETUP_INT_OPS
        #define IEQ_OP(i) warp_state.P[i] = (src1_ptr[i] == src2_ptr[i])
        UNROLL_8(IEQ_OP);
        #undef IEQ_OP
        return;

    LABEL_OP_ISETP_GT:
        SETUP_INT_OPS
        #define IGT_OP(i) warp_state.P[i] = (src1_ptr[i] > src2_ptr[i])
        UNROLL_8(IGT_OP);
        #undef IGT_OP
        return;
    } // End Integer Block

    // --- FLOAT ALU ---
    {
        uint8_t dest, src1, src2;
        float *dest_ptr, *src1_ptr, *src2_ptr;
        
        #define SETUP_FLOAT_OPS \
            dest = inst.dest; \
            src1 = inst.src1; \
            src2 = inst.src2_imm; \
            dest_ptr = &warp_state.F[dest][0]; \
            src1_ptr = &warp_state.F[src1][0]; \
            src2_ptr = &warp_state.F[src2][0];

    LABEL_OP_FADD:
        SETUP_FLOAT_OPS
        #define FADD_OP(i) dest_ptr[i] = src1_ptr[i] + src2_ptr[i]
        UNROLL_8(FADD_OP);
        #undef FADD_OP
        return;

    LABEL_OP_FSUB:
        SETUP_FLOAT_OPS
        #define FSUB_OP(i) dest_ptr[i] = src1_ptr[i] - src2_ptr[i]
        UNROLL_8(FSUB_OP);
        #undef FSUB_OP
        return;

    LABEL_OP_FMUL:
        SETUP_FLOAT_OPS
        #define FMUL_OP(i) dest_ptr[i] = src1_ptr[i] * src2_ptr[i]
        UNROLL_8(FMUL_OP);
        #undef FMUL_OP
        return;

    LABEL_OP_FDIV:
        SETUP_FLOAT_OPS
        #define FDIV_OP(i) dest_ptr[i] = src1_ptr[i] / src2_ptr[i]
        UNROLL_8(FDIV_OP);
        #undef FDIV_OP
        return;

    LABEL_OP_FFMA:
        SETUP_FLOAT_OPS
        #define FFMA_OP(i) dest_ptr[i] = src1_ptr[i] * src2_ptr[i] + dest_ptr[i]
        UNROLL_8(FFMA_OP);
        #undef FFMA_OP
        return;

    LABEL_OP_HMMA_I8:
        return; // Placeholder
    } // End Float Block

    // --- SFU ---
    {
        uint8_t dest, src1;
        float *F_dest, *F_src1;

        #define SETUP_SFU_OPS \
            dest = inst.dest; \
            src1 = inst.src1; \
            F_dest = &warp_state.F[dest][0]; \
            F_src1 = &warp_state.F[src1][0];

    LABEL_OP_SFU_RCP:
        SETUP_SFU_OPS
        #define RCP_OP(i) F_dest[i] = 1.0f / F_src1[i]
        UNROLL_8(RCP_OP);
        #undef RCP_OP
        return;

    LABEL_OP_SFU_SQRT:
        SETUP_SFU_OPS
        #define SQRT_OP(i) F_dest[i] = 1.0f / fast_rsqrt(F_src1[i])
        UNROLL_8(SQRT_OP);
        #undef SQRT_OP
        return;

    LABEL_OP_SFU_GELU:
        SETUP_SFU_OPS
        #define GELU_OP(i) F_dest[i] = F_src1[i] * fast_sigmoid(F_src1[i])
        UNROLL_8(GELU_OP);
        #undef GELU_OP
        return;

    LABEL_OP_SFU_RELU:
        SETUP_SFU_OPS
        #define RELU_OP(i) F_dest[i] = (F_src1[i] > 0.0f) ? F_src1[i] : 0.0f
        UNROLL_8(RELU_OP);
        #undef RELU_OP
        return;
        
    LABEL_OP_SFU_EXP: return; // Placeholder
    } // End SFU Block

    // --- MEMORY ---
    {
        uint8_t dest, src1, src2;
        uint32_t *R_dest, *R_src1, *R_src2;

        #define SETUP_MEM_OPS \
            dest = inst.dest; \
            src1 = inst.src1; \
            src2 = inst.src2_imm; \
            R_dest = &warp_state.R[dest][0]; \
            R_src1 = &warp_state.R[src1][0]; \
            R_src2 = &warp_state.R[src2][0];

    LABEL_OP_LDG: {
        SETUP_MEM_OPS
        uint32_t addr = R_src1[0];
        if (addr < VM_VRAM_SIZE) {
            uint32_t val = *(uint32_t*)&vram[addr];
            #define LDG_OP(i) R_dest[i] = val
            UNROLL_8(LDG_OP);
            #undef LDG_OP
            #ifdef DEBUG_TRACE
            addMemoryAccess(0, "read", addr, val);
            #endif
        }
        return;
    }

    LABEL_OP_STG: {
        SETUP_MEM_OPS
        uint32_t addr = R_src1[0];
        uint32_t val = R_dest[0];
        if (addr < VM_VRAM_SIZE) {
            *(uint32_t*)&vram[addr] = val; // Keep write non-volatile usually logic but volatile often used for IO/DMA buffers
            #ifdef DEBUG_TRACE
            addMemoryAccess(0, "write", addr, val);
            #endif
        }
        return;
    }

    LABEL_OP_LDL: {
        SETUP_MEM_OPS
        #define LDL_OP(i) \
        { \
            uint32_t addr = R_src1[i] + i * 4; \
            if(addr < VM_VRAM_SIZE) R_dest[i] = *(uint32_t*)&vram[addr]; \
        }
        UNROLL_8(LDL_OP);
        #undef LDL_OP
        return;
    }

    LABEL_OP_STL: {
        SETUP_MEM_OPS
        #define STL_OP(i) \
        { \
            uint32_t addr = R_src1[i] + i * 4; \
            if(addr < VM_VRAM_SIZE) *(uint32_t*)&vram[addr] = R_dest[i]; \
        }
        UNROLL_8(STL_OP);
        #undef STL_OP
        return;
    }

    LABEL_OP_LDX: {
        SETUP_MEM_OPS
        #define LDX_OP(i) \
        { \
            uint32_t addr = R_src1[i] + R_src2[i]; \
            if(addr < VM_VRAM_SIZE) R_dest[i] = *(uint32_t*)&vram[addr]; \
        }
        UNROLL_8(LDX_OP);
        #undef LDX_OP
        return;
    }

    LABEL_OP_STX: {
        SETUP_MEM_OPS
        #define STX_OP(i) \
        { \
            uint32_t addr = R_src1[i] + R_src2[i]; \
            /* if(addr < VM_VRAM_SIZE) */ *(uint32_t*)&vram[addr] = R_dest[i]; \
        }
        UNROLL_8(STX_OP);
        #undef STX_OP
        return;
    }

    LABEL_OP_LDS: {
        SETUP_MEM_OPS
        #define LDS_OP(i) \
        { \
            uint32_t addr = R_src1[i]; \
            if (addr < 256) R_dest[i] = warp_state.shared_mem[i][addr]; \
        }
        UNROLL_8(LDS_OP);
        #undef LDS_OP
        return;
    }

    LABEL_OP_STS: {
        SETUP_MEM_OPS
        #define STS_OP(i) \
        { \
            uint32_t addr = R_src1[i]; \
            if (addr < 256) warp_state.shared_mem[i][addr] = (uint8_t)R_dest[i]; \
        }
        UNROLL_8(STS_OP);
        #undef STS_OP
        return;
    }
    } // End Memory block

    // --- SYSTEM ---
    {
        uint8_t dest, sr_index;
        uint32_t* R_dest;

    LABEL_OP_S2R:
        dest = inst.dest;
        sr_index = inst.src1;
        R_dest = &warp_state.R[dest][0];
        #define S2R_OP(i) R_dest[i] = readSystemReg(i, sr_index)
        UNROLL_8(S2R_OP);
        #undef S2R_OP
        return;

    LABEL_OP_R2S: 
        return;

    LABEL_OP_TRACE:
        Serial.print("TRACE: "); Serial.println(warp_state.R[inst.src1][0], HEX);
        return;
    }
}

// Stubs for Interface Compatibility (Empty as they are flattened)
IRAM_ATTR void SIMDEngineV15::executeInteger(const Instruction& inst) {}
IRAM_ATTR void SIMDEngineV15::executeFloat(const Instruction& inst) {}
IRAM_ATTR void SIMDEngineV15::executeMemory(const Instruction& inst) {}
IRAM_ATTR void SIMDEngineV15::executeSFU(const Instruction& inst) {}
IRAM_ATTR void SIMDEngineV15::executeSystem(const Instruction& inst) {}

IRAM_ATTR uint32_t SIMDEngineV15::readSystemReg(uint8_t lane_id, uint8_t sr_index) {
     switch (sr_index) {
        case SR_TID:      return warp_state.SR.laneid[lane_id];
        case SR_LANEID:   return warp_state.SR.laneid[lane_id];
        case SR_WARPSIZE: return warp_state.SR.warpsize;
        default: return 0;
    }
}

IRAM_ATTR void SIMDEngineV15::executeControl(const Instruction& inst) {
    if(inst.opcode == OP_EXIT) halted = true;
}
