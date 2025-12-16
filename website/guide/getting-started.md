# Getting Started with Micro-CUDA

Micro-CUDA provides a Python-first development experience. You don't need to write C++ firmware; just flash the provided firmware once, and then write kernels in Python.

## 1. Flash Firmware (One-time)

```bash
# Upload the pre-compiled VM to your ESP32
./upload_esp32.sh
```

## 2. Write a Kernel (Python)

Create `hello_cuda.py`:

```python
from esp32_tools import quick_run, Instruction

# Simple Kernel: R2 = R0 * R1
program = [
    Instruction.mov(0, 10),       # R0 = 10
    Instruction.mov(1, 5),        # R1 = 5
    Instruction.imul(2, 0, 1),    # R2 = R0 * R1 = 50
    Instruction.exit_inst()
]

# Run on device (Change PORT to your actual USB port)
quick_run(
    port="/dev/cu.usbserial-0001",
    program=program,
    expected={'R2': 50}
)
```

## 3. Run It

```bash
python hello_cuda.py
```
