/*
 * ESP32 CUDA VM - Dual Core Implementation
 * architecture: Front-End (Core 0) / Back-End (Core 1)
 *
 * Core 0: Fetch, Decode, PC Control, Serial CLI, Branch Prediction (Sync)
 * Core 1: SIMD Execution, Trace Generation
 */

#include "vm_core.h"
#include "vm_simd_v15.h"
#include "vm_trace.h"
#include "instructions_v15.h"
// Force Recompile 2025-12-14
#include "vm_config.h"
#include "lz4.h"

// ===== Configuration =====
// (See vm_config.h)

// ===== Global Objects =====
VMCore vm;
SIMDEngineV15 simd_engine;
TraceUnit trace_unit;

// ===== Inter-Core Communication =====
// Batch Structure for High Throughput
struct InstrBatch
{
    uint8_t count;                    // Number of valid instructions (0-16)
    Instruction insts[VM_BATCH_SIZE]; // Instruction Array

    // Debug/Trace Metadata (First instruction in batch)
    uint32_t start_pc;
    uint64_t start_cycle;

    // Control Signal (handled as special batch)
    bool is_sync_req; // If true, this is a sync request (count should be 0)
    bool is_exit;     // If true, this is exit signal
};

QueueHandle_t instrQueue;      // FrontEnd -> BackEnd (InstrBatch)
QueueHandle_t feedbackQueue;   // BackEnd -> FrontEnd (Predicate Result)
SemaphoreHandle_t serialMutex; // Protect Serial Output

// ===== Task Handles =====
TaskHandle_t frontEndTaskHandle = NULL;
TaskHandle_t backEndTaskHandle = NULL;

// ===== Helper Functions =====
void safePrint(const String &str)
{
    if (xSemaphoreTake(serialMutex, portMAX_DELAY))
    {
        Serial.print(str);
        xSemaphoreGive(serialMutex);
    }
}

void safePrintln(const String &str)
{
    if (xSemaphoreTake(serialMutex, portMAX_DELAY))
    {
        Serial.println(str);
        xSemaphoreGive(serialMutex);
    }
}

// LZ4 Buffers
#define LZ4_CHUNK_SIZE 2048  // 2KB chunks for better performance
#define LZ4_COMPRESSED_BUF_SIZE LZ4_COMPRESSBOUND(LZ4_CHUNK_SIZE)
uint8_t rxBuffer[LZ4_COMPRESSED_BUF_SIZE]; 
uint8_t decompressedChunk[LZ4_CHUNK_SIZE];

void handle_compressed_data(uint8_t* dest_ptr, int max_dest_size, int total_original_size, bool is_imem) {
    int bytes_processed = 0;
    
    safePrintln("ACK_LZ4_GO");
    Serial.flush();
    vTaskDelay(1);

    while (bytes_processed < total_original_size) {
        // 1. Read Block Header (2 Bytes: Compressed Length)
        uint16_t compressed_len = 0;
        if (Serial.readBytes((uint8_t*)&compressed_len, 2) != 2) {
             safePrintln("ERR_LZ4_HEAD_TIMEOUT");
             break;
        }

        // 2. Read Compressed Data
        if (Serial.readBytes(rxBuffer, compressed_len) != compressed_len) {
            safePrintln("ERR_LZ4_DATA_TIMEOUT"); 
            break; 
        }

        // 3. Decompress
        int decompressed_len = LZ4_decompress_safe(
            (const char*)rxBuffer, 
            (char*)decompressedChunk, 
            compressed_len, 
            LZ4_CHUNK_SIZE
        );

        if (decompressed_len < 0) {
            safePrintln("ERR_LZ4_CORRUPT"); 
            break;
        }

        // 4. Write to Destination
        if (bytes_processed + decompressed_len > max_dest_size) {
             safePrintln("ERR_OVERFLOW");
             break;
        }
        
        memcpy(dest_ptr + bytes_processed, decompressedChunk, decompressed_len);
        bytes_processed += decompressed_len;
    }
    
    if (bytes_processed == total_original_size) {
        if (is_imem) {
            vm.setProgramLength(total_original_size / 4);
        }
        safePrintln("LZ4_LOAD_OK");
        Serial.flush();
        vTaskDelay(1);
    }
}

void printLaneRegisters(int lane_id); // Forward declaration

// ===== Back-End Task (Core 1) =====
// Responsible for: Batch Execution & SIMD
// Optimized for throughput: Hot Path Splitting

// Force O3 optimization for this critical hot path
#pragma GCC optimize ("O3")

// Place in IRAM for fastest execution (no Flash access latency)
void IRAM_ATTR backEndTask(void *pvParameters)
{
    InstrBatch batch;
    
    // Cache pointers to reduce indirect access overhead
    SIMDEngineV15* pEngine = &simd_engine;
    TraceUnit* pTrace = &trace_unit;

    safePrintln("✅ Back-End (SIMD) Task Started on Core 1 [TURBO]");

    while (true)
    {
        // Wait for Batch (Blocking)
        if (xQueueReceive(instrQueue, &batch, portMAX_DELAY) == pdTRUE)
        {
            // --- 1. Control Signals (Low Frequency) ---
            if (batch.is_sync_req || batch.is_exit)
            {
                if (batch.is_sync_req) 
                {
                    // Sync Request: Return Lane 0 Predicate
                    uint32_t predicate = pEngine->warp_state.P[0];
                    xQueueSend(feedbackQueue, &predicate, portMAX_DELAY);
                } 
                else 
                { 
                    // Exit Signal
                    uint32_t doneSignal = 0xFFFFFFFF;
                    xQueueSend(feedbackQueue, &doneSignal, portMAX_DELAY);
                }
                continue;
            }

            // --- 2. Execution Path (High Frequency Hot Spot) ---
            
            // Optimization Key: Move trace check outside loop
            // Enables更激进的 Loop Unrolling for "Fast Path"
            
#ifdef DEBUG_TRACE
            if (pTrace->isStreamMode()) 
            {
                // [Slow Path] Debug Mode: Full Logging
                for (int i = 0; i < batch.count; i++)
                {
                    pTrace->beginInstructionRecord(batch.start_cycle + i, batch.start_pc + i, 0, *pEngine, batch.insts[i]);
                    pEngine->execute(batch.insts[i]);
                    pTrace->finalizeInstructionRecord(*pEngine);
                }
            } 
            else
#endif // DEBUG_TRACE
            {
                // [Fast Path] Turbo Mode: No branches, pure computation
                // Use local pointer to accelerate array access
                Instruction* instructions = batch.insts;
                int count = batch.count;
                
                for (int i = 0; i < count; i++)
                {
                    pEngine->execute(instructions[i]);
                }
            }
        }
    }
}

// ===== Front-End Task (Core 0) =====
void frontEndTask(void *pvParameters)
{
    String inputBuffer = "";
    bool running = false;

    // Batch Buffer
    InstrBatch currentBatch;
    currentBatch.count = 0;
    currentBatch.is_sync_req = false;
    currentBatch.is_exit = false;

    safePrintln("✅ Front-End (Control) Task Started on Core 0");
    safePrintln("\nReady. Type 'help' for commands.");

    while (true)
    {
        if (!running)
        {
            int serialQuota = 64;
            while (Serial.available() && serialQuota-- > 0)
            {
                char c = Serial.read();
                if (c == '\n')
                {
                    String cmd = inputBuffer;
                    inputBuffer = "";
                    cmd.trim();

                    if (cmd.length() > 0)
                    {
                        // 🚀 LZ4 Compressed Kernel Load
                        // 格式: load_imem_lz4 <uncompressed_size>
                        if (cmd.startsWith("load_imem_lz4 "))
                        {
                            int firstSpace = cmd.indexOf(' ');
                            if (firstSpace != -1)
                            {
                                int uncompressed_size = cmd.substring(firstSpace + 1).toInt();
                                if (uncompressed_size > 0 && uncompressed_size <= VM_PROGRAM_SIZE * 4)
                                {
                                    handle_compressed_data((uint8_t *)vm.getProgramMemoryPtr(), VM_PROGRAM_SIZE * 4, uncompressed_size, true);
                                }
                                else
                                {
                                    safePrintln("ERR_INVALID_SIZE");
                                }
                            }
                        }
                        // 🚀 Standard Kernel Load
                        // 格式: load_imem <byte_count>
                        else if (cmd.startsWith("load_imem "))
                        {
                            int firstSpace = cmd.indexOf(' ');
                            if (firstSpace != -1)
                            {
                                int byte_count = cmd.substring(firstSpace + 1).toInt();

                                if (byte_count > 0 && byte_count <= VM_PROGRAM_SIZE * 4)
                                {
                                    safePrintln("ACK_KERN_GO:" + String(byte_count));
                                    Serial.flush(); // Ensure ACK is sent
                                    vTaskDelay(1); // Yield to let TX finish

                                    uint8_t *imem_ptr = (uint8_t *)vm.getProgramMemoryPtr();
                                    int total_received = 0;
                                    unsigned long last_activity = millis();

                                    while (total_received < byte_count)
                                    {
                                        int to_read = min((int)VM_SERIAL_BLOCK_READ_SIZE, byte_count - total_received);
                                        int got = Serial.readBytes(imem_ptr + total_received, to_read);
                                        
                                        if (got > 0)
                                        {
                                            total_received += got;
                                            last_activity = millis();
                                        }
                                        else
                                        {
                                            if (millis() - last_activity > 3000)
                                            {
                                                safePrintln("KERN_TIMEOUT");
                                                break;
                                            }
                                        }
                                    }

                                    if (total_received == byte_count)
                                    {
                                        vm.setProgramLength(byte_count / 4);
                                        safePrintln("KERN_OK");
                                    }
                                }
                                else
                                {
                                    safePrintln("ERR_SIZE");
                                }
                            }
                        }
                        else if (cmd == "kernel_launch")
                        {
                            vm.resetVM();
                            simd_engine.softReset(); // Important: Don't clear VRAM here!

                            safePrintln("Running...");

                            while (vm.getPC() < vm.getProgramLength())
                            {
                                uint32_t raw_inst = vm.fetch();
                                Instruction inst;
                                inst.decode(raw_inst);

                                InstrBatch &batch = currentBatch;
                                bool is_exit = (inst.opcode == OP_EXIT);
                                bool is_sync = (inst.opcode == OP_BAR_SYNC);
                                bool is_branch = (inst.opcode == OP_BRA || inst.opcode == OP_BRZ);

                                vm.incPC();
                                vm.incCycleCount();

                                if (is_exit || is_sync || is_branch)
                                {
                                    if (batch.count > 0)
                                    {
                                        batch.is_exit = false;
                                        batch.is_sync_req = false;
                                        xQueueSend(instrQueue, &batch, portMAX_DELAY);
                                        batch.count = 0;
                                    }
                                    
                                    // Handle Sync/Exit/Conditional Branch (Requests needing feedback)
                                    if (is_exit || is_sync || (is_branch && inst.opcode == OP_BRZ))
                                    {
                                        InstrBatch ctrlBatch;
                                        ctrlBatch.count = 0;
                                        ctrlBatch.is_sync_req = (!is_exit); // Sync for BRZ or BAR_SYNC
                                        ctrlBatch.is_exit = is_exit;
                                        
                                        xQueueSend(instrQueue, &ctrlBatch, portMAX_DELAY);
                                        
                                        uint32_t feedback;
                                        xQueueReceive(feedbackQueue, &feedback, portMAX_DELAY);
                                        
                                        if (inst.opcode == OP_BRZ) {
                                            // Branch if Predicate is Zero (False)
                                            // P0 is returned in feedback
                                            if (feedback == 0) vm.setPC(inst.dest);
                                        }
                                    }
                                    
                                    if (inst.opcode == OP_BRA)
                                        vm.setPC(inst.dest);
                                    if (is_exit)
                                        break;
                                }
                                else
                                {
                                    batch.insts[batch.count++] = inst;
                                    if (batch.count >= VM_BATCH_SIZE)
                                    {
                                        batch.is_exit = false;
                                        batch.is_sync_req = false;
                                        xQueueSend(instrQueue, &batch, portMAX_DELAY);
                                        batch.count = 0;
                                    }
                                }
                            }
                            safePrintln("Program Finished (EXIT)");
                            if (trace_unit.isStreamMode())
                                trace_unit.endProgram();
                        }
                        else if (cmd == "gpu_reset")
                        {
                            vm.init();
                            simd_engine.reset();
                            safePrintln("GPU Reset Complete");
                        }
                        else if (cmd.startsWith("dma_h2d "))
                        {
                            int firstSpace = cmd.indexOf(' ');
                            int secondSpace = cmd.indexOf(' ', firstSpace + 1);

                            if (firstSpace != -1 && secondSpace != -1)
                            {
                                uint32_t addr = strtoul(cmd.substring(firstSpace + 1, secondSpace).c_str(), NULL, 16);
                                int count = cmd.substring(secondSpace + 1).toInt();

                                if (addr + count <= VM_VRAM_SIZE)
                                {
                                    safePrintln("ACK_DMA_GO:" + String(count));
                                    Serial.flush();
                                    vTaskDelay(1);

                                    // Use chunk read for speed
                                    uint8_t *device_ptr = (uint8_t *)&simd_engine.vram[addr];
                                    int total_received = 0;
                                    unsigned long last_activity = millis();

                                    while (total_received < count)
                                    {
                                        int to_read = min((int)VM_SERIAL_BLOCK_READ_SIZE, count - total_received);
                                        int got = Serial.readBytes(device_ptr + total_received, to_read);

                                        if (got > 0)
                                        {
                                            total_received += got;
                                            last_activity = millis();
                                        }
                                        else
                                        {
                                            if (millis() - last_activity > 2000)
                                            {
                                                safePrintln("DMA_TIMEOUT_ERR");
                                                break;
                                            }
                                        }
                                    } // Close While Loop

                                    if (total_received == count)
                                        safePrintln("DMA_OK");
                                }
                                else
                                {
                                    safePrintln("ERR_SEGFAULT");
                                }
                            }
                        }
                        else if (cmd.startsWith("dma_d2h "))
                        {
                            int firstSpace = cmd.indexOf(' ');
                            int secondSpace = cmd.indexOf(' ', firstSpace + 1);
                            if (firstSpace != -1)
                            {
                                uint32_t addr = 0;
                                int count = 0;
                                if (secondSpace != -1)
                                {
                                    addr = strtoul(cmd.substring(firstSpace + 1, secondSpace).c_str(), NULL, 16);
                                    count = cmd.substring(secondSpace + 1).toInt();
                                }
                                else
                                {
                                    addr = strtoul(cmd.substring(firstSpace + 1).c_str(), NULL, 16);
                                    count = 1;
                                }
                                for (int i = 0; i < count; i++)
                                {
                                    int curr = addr + i * 4;
                                    if (curr < VM_VRAM_SIZE)
                                    {
                                        uint32_t val = *((volatile uint32_t *)&simd_engine.vram[curr]);
                                        safePrintln(String(curr, HEX) + ": " + String(val, HEX));
                                    }
                                }
                            }
                        }
                        else if (cmd.startsWith("dma_d2h_binary "))
                        {
                            int firstSpace = cmd.indexOf(' ');
                            int secondSpace = cmd.indexOf(' ', firstSpace + 1);

                            if (firstSpace != -1)
                            {
                                uint32_t addr = 0;
                                int count = 0; // Word count
                                
                                if (secondSpace != -1)
                                {
                                    addr = strtoul(cmd.substring(firstSpace + 1, secondSpace).c_str(), NULL, 16);
                                    count = cmd.substring(secondSpace + 1).toInt();
                                }
                                else
                                {
                                    addr = strtoul(cmd.substring(firstSpace + 1).c_str(), NULL, 16);
                                    count = 1;
                                }

                                int total_bytes = count * 4;

                                if (addr + total_bytes <= VM_VRAM_SIZE)
                                {
                                    safePrintln("ACK_D2H_BIN:" + String(total_bytes));
                                    Serial.flush();
                                    vTaskDelay(1);
                                    
                                    // Critical Section for Serial Write
                                    if (xSemaphoreTake(serialMutex, portMAX_DELAY))
                                    {
                                        Serial.write((uint8_t*)&simd_engine.vram[addr], total_bytes);
                                        xSemaphoreGive(serialMutex);
                                    }
                                    
                                    safePrintln("D2H_OK");
                                }
                                else
                                {
                                    safePrintln("ERR_SEGFAULT");
                                }
                            }
                        }
                        // 🚀 LZ4 Image Data Load (H2D)
                        // 格式: dma_h2d_lz4 <hex_addr> <uncompressed_size>
                        else if (cmd.startsWith("dma_h2d_lz4 "))
                        {
                            int firstSpace = cmd.indexOf(' ');
                            int secondSpace = cmd.indexOf(' ', firstSpace + 1);
                            
                            if (firstSpace != -1 && secondSpace != -1)
                            {
                                uint32_t addr = strtoul(cmd.substring(firstSpace + 1, secondSpace).c_str(), NULL, 16);
                                int uncompressed_size = cmd.substring(secondSpace + 1).toInt();

                                if (addr + uncompressed_size <= VM_VRAM_SIZE)
                                {
                                    handle_compressed_data((uint8_t *)&simd_engine.vram[addr], VM_VRAM_SIZE - addr, uncompressed_size, false);
                                }
                                else
                                {
                                    safePrintln("ERR_SEGFAULT");
                                }
                            }
                        }
                        else if (cmd.startsWith("reg"))
                        {
                            int lane = 0;
                            if (cmd.length() > 3)
                                lane = cmd.substring(4).toInt();
                            if (lane >= 0 && lane < 8)
                                printLaneRegisters(lane);
                        }
                        else if (cmd == "stats")
                        {
                            safePrintln("\n=== VM Statistics ===");
                            safePrintln("Instructions Loaded : " + String(vm.getProgramLength()));
                            safePrintln("VRAM Size           : " + String(VM_VRAM_SIZE) + " bytes");
                            safePrintln("=====================");
                        }
                    }
                }
                else if (c != '\r')
                {
                    inputBuffer += c;
                }
            }
            vTaskDelay(1);
        }
        else
        {
            vTaskDelay(10);
        }
    }
}

// ===== Initial Setup =====
void setup()
{
    setCpuFrequencyMhz(VM_CPU_FREQ);       // 240MHz
    Serial.setRxBufferSize(VM_SERIAL_RX_SIZE); // Turbo Buffer
    Serial.begin(VM_BAUD_RATE);          // 4x Standard Speed
    delay(500);

    Serial.println("\n=== ESP32 CUDA VM [TURBO MODE] ===");

    instrQueue = xQueueCreate(VM_QUEUE_SIZE, sizeof(InstrBatch));
    feedbackQueue = xQueueCreate(1, sizeof(uint32_t));
    serialMutex = xSemaphoreCreateMutex();

    if (!instrQueue || !feedbackQueue)
        while (1)
            ;

    vm.init();
    simd_engine.reset();

    xTaskCreatePinnedToCore(frontEndTask, "FrontEnd", VM_STACK_SIZE, NULL, 2, &frontEndTaskHandle, 0);
    xTaskCreatePinnedToCore(backEndTask, "BackEnd", VM_STACK_SIZE, NULL, 1, &backEndTaskHandle, 1);
}

// ===== Helper: Print Lane Registers =====
void printLaneRegisters(int lane_id)
{
    if (xSemaphoreTake(serialMutex, portMAX_DELAY))
    {
        Serial.println();
        Serial.println("========================================");
        Serial.print(" Lane ");
        Serial.print(lane_id);
        Serial.println(" Registers");
        Serial.println("========================================");

        Serial.println("System Registers:");
        Serial.print("  SR_TID      = ");
        Serial.println(simd_engine.warp_state.SR.tid[lane_id]);
        Serial.print("  SR_LANEID   = ");
        Serial.println(simd_engine.warp_state.SR.laneid[lane_id]);

        Serial.println("General Purpose Registers (non-zero):");
        bool has_nonzero = false;
        for (int i = 0; i < 32; i++)
        {
            uint32_t val = simd_engine.warp_state.R[i][lane_id];
            if (val != 0)
            {
                Serial.print("  R");
                if (i < 10)
                    Serial.print(" ");
                Serial.print(i);
                Serial.print(" = ");
                Serial.println(val);
                has_nonzero = true;
            }
        }
        if (!has_nonzero)
            Serial.println("  (all zero)");
        Serial.println("========================================");
        Serial.println();

        xSemaphoreGive(serialMutex);
    }
}
// ===== Unused Loop =====
void loop()
{
    vTaskDelete(NULL); // Delete the default Arduino loop task to save RAM
}
