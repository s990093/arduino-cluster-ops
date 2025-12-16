# Python SDK Reference

The `esp32_tools` package provides the bridge between your host PC and the ESP32.

## Key Functions

### `quick_run()`

Compiles, uploads, and runs a kernel in one step.

```python
def quick_run(port: str, program: List[Instruction], expected: Dict[str, Any] = None):
    """
    Args:
        port: Serial port (e.g., /dev/ttyUSB0).
        program: List of Instructions.
        expected: Optional dictionary of expected register values for verification.
    """
```

### `Instruction` Builder

Helper class to generate 32-bit opcodes.

- `Instruction.mov(dest, src)`
- `Instruction.iadd(dest, src1, src2)`
- `Instruction.exit_inst()`
