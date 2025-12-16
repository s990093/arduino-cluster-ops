# ✅ Enhanced Trace 功能说明

## 📊 关于完整寄存器输出

### 问题

当输出所有 32 个 R 寄存器 + 32 个 F 寄存器 + 8 个 P 寄存器时：

- **每个 Lane**: ~1500 字节
- **8 个 Lane**: ~12,000 字节
- **13 条指令**: ~156,000 字节

**ESP32 Serial Buffer 限制**：

- 默认 buffer 太小
- 会导致数据截断
- JSON 格式被破坏

### 解决方案

#### 方案 1: 压缩输出（推荐）✅

只输出**非零寄存器**，大幅减少数据量：

```cpp
// 当前实现（已修改为完整输出）
// 如需压缩，改回只输出非零值

void TraceUnit::printLaneData() {
    // ... 只打印非零的 R/F/P
}
```

#### 方案 2: 增加 Serial Buffer

修改 ESP32 固件配置：

```cpp
// 在 .ino 文件开头添加
#define SERIAL_TX_BUFFER_SIZE 4096
#define SERIAL_RX_BUFFER_SIZE 4096
```

#### 方案 3: 分批输出

修改为每次只输出部分寄存器：

```json
{
  "lanes": [
    {
      "lane_id": 0,
      "R_0_15": [0, 0, ...],  // 前 16 个
      "R_16_31": [0, 0, ...], // 后 16 个
      "F_active": [5, 10],    // 只输出有值的索引
      "F_values": [1.2, 3.4]  // 对应的值
    }
  ]
}
```

---

## 🎯 当前状态

**更新**：固件已修改为输出**所有寄存器**

但由于数据量过大，**建议使用以下配置之一**：

### 配置 A: 完整输出（需增加 Buffer）

```cpp
// esp32_cuda_vm.ino 开头添加
#define SERIAL_TX_BUFFER_SIZE 8192

// vm_trace.cpp 保持当前实现
// 输出所有 32 R + 32 F + 8 P
```

### 配置 B: 智能压缩（默认推荐）

```cpp
// vm_trace.cpp 修改为
void TraceUnit::printLaneData() {
    // 输出所有 R (重要)
    Serial.print(", \"R\": [");
    for (int i = 0; i < 32; i++) {
        if (i > 0) Serial.print(", ");
        Serial.print(lane.R[i]);
    }

    // 只输出非零 F (节省空间)
    Serial.print("], \"F_active\": {");
    bool first = true;
    for (int i = 0; i < 32; i++) {
        if (lane.F[i] != 0.0f) {
            if (!first) Serial.print(", ");
            Serial.print("\"");
            Serial.print(i);
            Serial.print("\": ");
            Serial.print(lane.F[i], 6);
            first = false;
        }
    }

    // 只输出非零 P
    Serial.print("}, \"P_active\": [");
    first = true;
    for (int i = 0; i < 8; i++) {
        if (lane.P[i] != 0) {
            if (!first) Serial.print(", ");
            Serial.print(i);
            first = false;
        }
    }
    Serial.print("]");
}
```

---

## 📌 建议

### 对于当前项目（整数运算为主）

使用**配置 B（智能压缩）**：

- ✅ 输出所有 32 个 R 寄存器（主要使用）
- ✅ 只输出非零 F 寄存器（节省空间）
- ✅ 只输出非零 P 寄存器（很少使用）

**优点**：

- 所有重要数据都保留
- JSON 不会被截断
- 数据量减少 70%

---

## 🔧 快速修复

如果你需要立即看到完整的寄存器状态，最简单的方法是：

### 方法 1: 减少 trace 指令数

只 trace 关键的几条指令，而不是全部 13 条。

### 方法 2: 使用 `reg` 命令

直接在串口监视器中查看寄存器：

```bash
# 连接 ESP32
python3 cli.py monitor -p /dev/cu.usbserial-589A0095521 -b 115200

# 查看所有 Lane 的寄存器
reg 0
reg 1
...
reg 7
```

这样可以看到完整的 32 个 R 寄存器！

---

## ✅ 总结

**目前的问题**：输出数据量太大，Serial Buffer 溢出

**推荐方案**：

1. ✅ 使用智能压缩（R 全输出，F/P 只输出非零）
2. ✅ 或使用 `reg` 命令查看完整状态
3. ⏳ 如需完整 trace，增加 Serial Buffer 大小

**你想使用哪个方案？**
