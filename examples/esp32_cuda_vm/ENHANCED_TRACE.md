# ESP32 增強 Trace 格式更新

## ✅ 已增強的功能

### 新增 JSON 字段

每個 trace record 現在包含：

```json
{
  "cycle": 1,
  "pc": 1,
  "instruction": "0x10000002",
  "asm": "0x10 dest=0 src1=0 src2=2",      // ⭐ 新增：指令反匯編
  "exec_time_us": 125,                     // ⭐ 新增：實際執行時間
  "hw_ctx": {                             // ⭐ 新增：硬件上下文
    "sm_id": 0,
    "warp_id": 0,
    "lane_id": 0,
    "active_mask": "0xFF"
  },
  "perf": {                               // ⭐ 增強：性能指標
    "latency": 1,
    "stall_cycles": 0,
    "stall_reason": "NONE",
    "pipe_stage": "WRITEBACK",
    "core_id": 0,
    "predicate_masked": false,
    "sync_barrier": false,
    "simd_width": 8
  },
  "lanes": [                              // ⭐ 擴展：R0-R23
    {
      "lane_id": 0,
      "R": [2, 0, 0, ...]  // 24 個寄存器
    },
    // ... lanes 1-7
  ]
}
```

### 性能特性

1. **exec_time_us**: 使用 `micros()` 精確測量每條指令的執行時間
2. **hw_ctx**: 模擬 GPU 硬件上下文（SM ID, Warp ID, Active Mask）
3. **asm**: 自動反匯編顯示 opcode、dest、src1、src2
4. **perf.sync_barrier**: 自動檢測 BAR.SYNC 指令（opcode 0x05）
5. **完整寄存器**: 每個 lane 顯示 R0-R23

## 🔧 手動上傳固件

由於自動上傳失敗，請按以下步驟手動上傳：

### 方法 1: Arduino IDE

1. 打開 Arduino IDE
2. File → Open → 選擇 `examples/esp32_cuda_vm/esp32_cuda_vm.ino`
3. Tools → Board → ESP32 Arduino → ESP32 Dev Module
4. Tools → Port → `/dev/cu.usbserial-589A0095521`
5. 點擊 Upload 按鈕 (→)

### 方法 2: 重試 arduino-cli（按下 boot 按鈕）

```bash
# 步驟：
# 1. 按住 ESP32 的 BOOT 按鈕
# 2. 運行以下命令
# 3. 看到 "Connecting..." 時保持按住 BOOT
# 4. 上傳開始後釋放按鈕

arduino-cli upload --fqbn esp32:esp32:esp32 -p /dev/cu.usbserial-589A0095521 examples/esp32_cuda_vm
```

## 📊 測試增強格式

上傳後運行：

```bash
python example_usage.py /dev/cu.usbserial-589A0095521
```

檢查 `transformer_complete_trace.json`:

```bash
# 查看第一條 instruction 的完整格式
jq '.records[0]' transformer_complete_trace.json

# 查看所有執行時間
jq '.records[].exec_time_us' transformer_complete_trace.json

# 查看 assembly
jq '.records[].asm' transformer_complete_trace.json
```

## 📈 預期輸出範例

```json
{
  "trace_version": "2.1",
  "program": "Transformer Program",
  "total_instructions": 10,
  "records": [
    {
      "cycle": 1,
      "pc": 1,
      "instruction": "0x10000002",
      "asm": "0x10 dest=0 src1=0 src2=2",
      "exec_time_us": 125,
      "hw_ctx": {
        "sm_id": 0,
        "warp_id": 0,
        "lane_id": 0,
        "active_mask": "0xFF"
      },
      "perf": {
        "latency": 1,
        "stall_cycles": 0,
        "stall_reason": "NONE",
        "pipe_stage": "WRITEBACK",
        "core_id": 0,
        "predicate_masked": false,
        "sync_barrier": false,
        "simd_width": 8
      },
      "lanes": [
        {
          "lane_id": 0,
          "R": [
            2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0
          ]
        }
        // ... 8 lanes total
      ]
    }
    // ... 更多 records
  ]
}
```

## 🎯 改進摘要

| 功能         | 舊版  | 新版                                |
| ------------ | ----- | ----------------------------------- |
| 寄存器範圍   | R0-R7 | R0-R23 ✅                           |
| 執行時間     | ❌    | exec_time_us ✅                     |
| 硬件上下文   | ❌    | hw_ctx ✅                           |
| 反匯編       | ❌    | asm ✅                              |
| 性能指標     | 基本  | 完整 (latency, stalls, pipeline) ✅ |
| Barrier 檢測 | ❌    | sync_barrier ✅                     |
