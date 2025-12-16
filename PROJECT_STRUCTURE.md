# Arduino Cluster Ops - 项目结构

## 目录说明

### 📁 根目录
- `cli.py` - 主命令行工具
- `upload_esp32.sh` - ESP32 上传脚本
- `LICENSE` - 许可证文件

### 📁 apps/
应用程序和工具脚本
- `app_edge_conv.py` - 边缘检测应用（主程序）
- `app_attention_esp32.py` - Transformer Attention ESP32实现
- `app_attention_map.py` - Attention可视化
- `app_debug_trace.py` - 调试追踪工具
- `build_attention_kernels.py` - Attention内核编译器
- `mcc_run.py` - MCC运行工具

#### 📁 apps/tests/
测试脚本

### 📁 docs/
项目文档

#### 📁 docs/markdown/
Markdown 文档
- `PROJECT_SUMMARY.md` - 项目总结
- `FINAL_SUMMARY.md` - 最终总结
- `MCC_RUN_GUIDE.md` - MCC运行指南
- 其他文档...

### 📁 examples/
示例代码（ESP32固件等）

### 📁 arduino_tools/
Arduino 相关工具

### 📁 esp32_tools/
ESP32 专用工具

### 📁 micro_cuda_compiler/
Micro-CUDA 编译器

### 📁 simulator/
模拟器相关代码

### 📁 build/
编译输出目录

### 📁 image/
图像资源

### 📁 venv/
Python 虚拟环境

### 📁 __test__/
临时测试文件

### 📁 __pycache__/
Python 缓存文件
