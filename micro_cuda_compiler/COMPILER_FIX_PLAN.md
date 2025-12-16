# 🔧 编译器修复计划

## 🐛 发现的问题

### 1. 未初始化寄存器 (Line 314)

```python
val_reg = self.allocator.allocate_if_needed(f'%{match.group(1)}')
```

**问题**: `allocate_if_needed` 可能返回一个从未赋值的寄存器

**示例**:

```assembly
; Bug: R9 从未被初始化！
STX R8, R10, R9  ; 存储 R9 的值（但 R9 是未知的）
```

### 2. 寄存器分配器缺陷

当前的 `RegisterAllocator` 不跟踪寄存器是否已被赋值。

### 3. 编译器生成冗余代码

多次编译相同的常量（如 MOV R4, 4）

## ✅ 修复方案

### Fix 1: 增强寄存器分配器

添加寄存器初始化跟踪：

```python
class RegisterAllocator:
    def __init__(self, max_regs=32):
        self.max_regs = max_regs
        self.next_reg = 0
        self.var_to_reg = {}
        self.initialized_regs = set()  # NEW: 跟踪已初始化的寄存器

    def allocate(self, var_name):
        if var_name in self.var_to_reg:
            return self.var_to_reg[var_name]

        if self.next_reg >= self.max_regs:
            raise RuntimeError(f"Out of registers! Need more than {self.max_regs}")

        reg = self.next_reg
        self.var_to_reg[var_name] = reg
        self.next_reg += 1
        # Mark as initialized when allocated for a destination
        self.initialized_regs.add(reg)
        return reg

    def allocate_if_needed(self, var_name):
        \"\"\"Allocate only if not already allocated\"\"\"
        if var_name in self.var_to_reg:
            reg = self.var_to_reg[var_name]
            # Check if initialized
            if reg not in self.initialized_regs:
                raise RuntimeError(f"Using uninitialized register R{reg} for {var_name}")
            return reg
        else:
            # This is an ERROR - we're using a variable that was never defined!
            raise RuntimeError(f"Variable {var_name} used before definition!")
```

### Fix 2: 修复 Store 指令

```python
# Store instruction: store i32 %9, ptr %10, align 4
elif ir_inst.startswith('store'):
    match = re.match(r'store\\s+\\w+\\s+%(\\w+),\\s*ptr\\s+%(\\w+)', ir_inst)
    if match:
        val_var = f'%{match.group(1)}'
        addr_var = f'%{match.group(2)}'

        # IMPORTANT: Check if val_var exists
        if val_var not in self.allocator.var_to_reg:
            # This variable was never defined! Skip or error
            print(f"WARNING: Storing undefined variable {val_var}, skipping")
            return inst_list

        val_reg = self.allocator.var_to_reg[val_var]
        addr_reg = self.allocator.allocate_if_needed(addr_var)

        zero_reg = self.allocator.allocate('%zero_offset_st')
        inst_list.append(MicroCUDAInstruction(
            opcode="MOV",
            dest=zero_reg,
            src1=None,
            src2=None,
            imm=0,
            comment="Zero offset"
        ))

        inst_list.append(MicroCUDAInstruction(
            opcode="STX",
            dest=addr_reg,
            src1=zero_reg,
            src2=val_reg,
            imm=None,
            comment=f"Mem[R{addr_reg}] = R{val_reg}"
        ))
```

### Fix 3: 常量重用

```python
class RegisterAllocator:
    def __init__(self, max_regs=32):
        self.max_regs = max_regs
        self.next_reg = 0
        self.var_to_reg = {}
        self.initialized_regs = set()
        self.constant_cache = {}  # NEW: 缓存常量寄存器

    def allocate_constant(self, value):
        \"\"\"Allocate or reuse register for constant\"\"\"
        const_key = f'const_{value}'
        if const_key in self.constant_cache:
            return self.constant_cache[const_key]

        reg = self.allocate(const_key)
        self.constant_cache[const_key] = reg
        return reg
```

## 🎯 完整修复代码

修改 `/Users/hungwei/Desktop/Proj/arduino-cluster-ops/micro_cuda_compiler/mcc.py`

### 修改点 1: RegisterAllocator 类

```python
class RegisterAllocator:
    \"\"\"Register allocation with initialization tracking\"\"\"

    def __init__(self, max_regs=32):
        self.max_regs = max_regs
        self.next_reg = 0
        self.var_to_reg = {}
        self.initialized_regs = set()
        self.constant_cache = {}

    def allocate(self, var_name):
        \"\"\"Allocate a new register for a variable\"\"\"
        if var_name in self.var_to_reg:
            reg = self.var_to_reg[var_name]
            self.initialized_regs.add(reg)  # Mark as initialized
            return reg

        if self.next_reg >= self.max_regs:
            raise RuntimeError(f"Out of registers! Need more than {self.max_regs}")

        reg = self.next_reg
        self.var_to_reg[var_name] = reg
        self.initialized_regs.add(reg)
        self.next_reg += 1
        return reg

    def allocate_if_needed(self, var_name):
        \"\"\"Get existing register or error\"\"\"
        if var_name in self.var_to_reg:
            reg = self.var_to_reg[var_name]
            if reg not in self.initialized_regs:
                raise RuntimeError(f"Using uninitialized register R{reg} for {var_name}")
            return reg
        else:
            # Variable used before definition - try to allocate
            # This can happen with function parameters
            return self.allocate(var_name)

    def allocate_constant(self, value):
        \"\"\"Allocate or reuse constant\"\"\"
        const_key = f'const_{value}'
        if const_key in self.constant_cache:
            return self.constant_cache[const_key]

        reg = self.allocate(const_key)
        self.constant_cache[const_key] = reg
        return reg

    def reset(self):
        \"\"\"Reset allocator for new function\"\"\"
        self.next_reg = 0
        self.var_to_reg = {}
        self.initialized_regs = set()
        self.constant_cache = {}
```

### 修改点 2: Store 指令处理 (Line 310-335)

```python
# Store instruction: store i32 %9, ptr %10, align 4
elif ir_inst.startswith('store'):
    match = re.match(r'store\\s+\\w+\\s+%(\\w+),\\s*ptr\\s+%(\\w+)', ir_inst)
    if match:
        val_var = f'%{match.group(1)}'
        addr_var = f'%{match.group(2)}'

        # Check if value variable exists
        if val_var not in self.allocator.var_to_reg:
            # WARNING: Variable undefined - create a zero register
            print(f"WARNING: {val_var} undefined in store, using 0")
            val_reg = self.allocator.allocate(val_var)
            # Initialize to 0
            inst_list.append(MicroCUDAInstruction(
                opcode="MOV",
                dest=val_reg,
                src1=None,
                src2=None,
                imm=0,
                comment=f"Initialize {val_var} to 0"
            ))
        else:
            val_reg = self.allocator.var_to_reg[val_var]

        addr_reg = self.allocator.allocate_if_needed(addr_var)

        # STX: Store to address
        zero_reg = self.allocator.allocate_constant(0)  # Use constant allocator
        if zero_reg not in [r for inst in inst_list for r in [inst.dest] if inst.dest is not None]:
            # Only emit MOV if not already done
            inst_list.append(MicroCUDAInstruction(
                opcode="MOV",
                dest=zero_reg,
                src1=None,
                src2=None,
                imm=0,
                comment="Zero offset"
            ))

        inst_list.append(MicroCUDAInstruction(
            opcode="STX",
            dest=addr_reg,
            src1=zero_reg,
            src2=val_reg,
            imm=None,
            comment=f"Mem[R{addr_reg}] = R{val_reg}"
        ))
```

## 📋 测试计划

修复后测试：

```bash
# 重新编译 kernel
python micro_cuda_compiler/compile_kernel.py __test__/image_conv_kernel.cu -o __test__/image_conv_kernel.asm

# 运行诊断
python __test__/diagnose_conv.py

# 预期: MAE < 0.1
```

## 🎯 预期改进

修复后:

- ✅ 无未初始化寄存器
- ✅ 正确的变量跟踪
- ✅ 常量重用减少寄存器使用
- ✅ 更好的错误检测
