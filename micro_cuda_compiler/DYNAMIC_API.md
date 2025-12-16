# 動態 Kernel 編譯 API

## 🎯 功能

像 `test_enhanced_trace.py` 一樣，直接在 Python 測試腳本中：

1. ✅ **內聯寫 CUDA kernel 代碼**
2. ✅ **動態編譯到 .asm**
3. ✅ **指定輸出路徑**
4. ✅ **指定 target 配置**
5. 🚧 **生成二進制** (開發中)

## 🚀 快速使用

### 方式 1: 內聯 Kernel 代碼

```python
from micro_cuda_compiler.dynamic_compile import compile_kernel

# 在 Python 中直接寫 kernel！
kernel_code = """
#include "mcuda.h"

__global__ void vectorAdd(int* A, int* B, int* C) {
    int idx = laneId();
    C[idx] = A[idx] + B[idx];
}
"""

# 編譯並指定輸出
program, asm_path = compile_kernel(
    kernel_code,
    output_asm="my_kernels/vector_add.asm",  # 指定 .asm 輸出路徑
    target="esp32s3",                         # 指定硬體配置
    verbose=True
)

# asm_path = "my_kernels/vector_add.asm"
# 會自動生成包含硬體配置的 .asm 文件！
```

### 方式 2: 從文件編譯

```python
from micro_cuda_compiler.dynamic_compile import compile_kernel_file

program, asm_path = compile_kernel_file(
    "kernels/conv1d.cu",
    output_asm="output/conv1d_compiled.asm",
    target="esp32-psram"
)
```

### 方式 3: 使用 KernelCompiler 類別

```python
from micro_cuda_compiler.dynamic_compile import KernelCompiler

compiler = KernelCompiler()

# 編譯內聯代碼
kernel = """
#include "mcuda.h"
__global__ void myKernel(int* data) {
    data[laneId()] *= 2;
}
"""

program, asm = compiler.compile_from_string(
    kernel,
    output_asm="kernels/my_kernel.asm",
    target="esp32s3",
    verbose=True
)

# 編譯文件
program2, asm2 = compiler.compile_from_file(
    "kernels/another.cu",
    output_asm="output/another.asm"
)

# 清理臨時文件
compiler.cleanup()
```

## 📝 完整範例：`conv_dynamic.py`

```python
#!/usr/bin/env python3
from pathlib import Path
from micro_cuda_compiler.dynamic_compile import compile_kernel
from esp32_tools import ESP32Connection
from esp32_tools.program_loader_v15 import InstructionV15

# ===== 在測試中直接寫 Kernel =====
KERNEL = """
#include "mcuda.h"

__global__ void conv1d(int* input, int* kernel, int* output) {
    int lane = laneId();

    // 讀取滑動窗口
    int i0 = input[lane];
    int i1 = input[lane + 1];
    int i2 = input[lane + 2];

    // 讀取 kernel 權重
    int k0 = kernel[0];
    int k1 = kernel[1];
    int k2 = kernel[2];

    // MAC
    int result = i0*k0 + i1*k1 + i2*k2;

    // 寫回
    output[lane] = result;
}
"""

def main():
    # Step 1: 動態編譯 (會生成 .asm!)
    _, asm_path = compile_kernel(
        KERNEL,
        output_asm="__test__/conv1d_dynamic.asm",  # 輸出路徑
        target="esp32s3",                          # Target 配置
        verbose=True
    )

    print(f"✅ Assembly saved to: {asm_path}")
    print(f"   Contains full hardware configuration!")

    # Step 2: 手動提供程式 (直到編譯器完成)
    program = [...]  # 手寫的 assembly

    # Step 3: 連接 ESP32 並執行
    conn = ESP32Connection("/dev/cu.usbserial-XXX")

    # ... 設置 VRAM、載入、執行 ...

if __name__ == "__main__":
    main()
```

## 🎯 優點

### 像真正的測試腳本

```python
# 就像 test_enhanced_trace.py 一樣！
def test_my_kernel():
    # 直接在這裡寫 kernel
    kernel = """
    #include "mcuda.h"
    __global__ void test(int* data) {
        data[laneId()] = laneId() * 2;
    }
    """

    # 編譯
    _, asm = compile_kernel(kernel, output_asm="test.asm")

    # 執行
    # ... 連接 ESP32，載入，執行 ...
```

### 自動生成文檔化的 .asm

生成的 `.asm` 文件包含：

```assembly
; ====================================================================
; Micro-CUDA Kernel - Compiled Assembly
; ====================================================================
;
; Target Configuration:
;   Device:        ESP32-S3 with 8MB PSRAM
;   ISA Version:   v1.5
;   Architecture:  Dual-Core SIMT
;
; SIMD Configuration:
;   Lanes:         8
;   Warp Size:     8
;
; Memory Configuration:
;   VRAM Size:     1048576 bytes (1024 KB)
;   ...
;
; Source File: tmp285uef6w.cu
; Kernel Functions: conv1d
; Total Instructions: XX
; Registers Used: YY
;
; ===== CODE SECTION =====
...
```

### 靈活的輸出路徑

```python
# 可以將不同 kernel 的 .asm 放在不同目錄
compile_kernel(kernel1, output_asm="kernels/module_a/kernel1.asm")
compile_kernel(kernel2, output_asm="kernels/module_b/kernel2.asm")
compile_kernel(kernel3, output_asm="output/debug/kernel3.asm")
```

### 多 Target 支援

```python
# 為不同硬體生成
compile_kernel(kernel, output_asm="esp32_standard.asm", target="esp32")
compile_kernel(kernel, output_asm="esp32_psram.asm", target="esp32-psram")
compile_kernel(kernel, output_asm="esp32s3.asm", target="esp32s3")
```

## 📊 API 參考

### `compile_kernel()`

```python
def compile_kernel(
    kernel_code: str,              # CUDA kernel 源碼
    output_asm: Optional[str] = None,   # .asm 輸出路徑
    output_binary: Optional[str] = None, # 二進制輸出 (TODO)
    target: str = "default",       # Target 配置
    verbose: bool = True           # 顯示編譯訊息
) -> Tuple[Optional[List[InstructionV15]], str]
```

**Returns**: `(program, asm_path)`

- `program`: InstructionV15 列表 (當前為 None，未來會實作)
- `asm_path`: 生成的 .asm 文件路徑

### `compile_kernel_file()`

```python
def compile_kernel_file(
    kernel_file: str,              # .cu 文件路徑
    output_asm: Optional[str] = None,
    output_binary: Optional[str] = None,
    target: str = "default"
) -> Tuple[Optional[List[InstructionV15]], str]
```

### `KernelCompiler` 類別

```python
class KernelCompiler:
    def compile_from_string(...)  # 從字符串編譯
    def compile_from_file(...)    # 從文件編譯
    def compile_and_load(...)     # 編譯並載入
    def cleanup()                 # 清理臨時文件
```

## 🔧 當前狀態

- ✅ 內聯 kernel 代碼支援
- ✅ 動態編譯到 .asm
- ✅ 自定義輸出路徑
- ✅ Target 配置記錄
- ✅ 硬體參數 header 生成
- 🚧 LLVM IR → ISA 編譯器 (部分完成)
- 🚧 二進制輸出 (開發中)
- 🚧 從 .asm 解析回 InstructionV15 (開發中)

## 📖 使用場景

1. **快速原型開發**

   ```python
   # 快速測試不同 kernel 實作
   kernel_v1 = "..."
   kernel_v2 = "..."
   compile_kernel(kernel_v1, output_asm="test_v1.asm")
   compile_kernel(kernel_v2, output_asm="test_v2.asm")
   ```

2. **單元測試**

   ```python
   def test_vector_add():
       kernel = """..."""
       compile_kernel(kernel, output_asm="tests/vector_add.asm")
       # ... 執行並驗證 ...
   ```

3. **CI/CD Pipeline**
   ```python
   # 自動編譯所有 kernels 並保存 .asm
   for kernel_file in kernel_files:
       compile_kernel_file(
           kernel_file,
           output_asm=f"build/{kernel_file.stem}.asm"
       )
   ```

---

**版本**: 0.1.0 Alpha  
**狀態**: 動態編譯 ✅ | 二進制輸出 🚧  
**更新**: 2025-12-13
