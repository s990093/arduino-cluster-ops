# SIMD Lane Architecture Guide

## Lane Execution Model

The Micro-CUDA engine emulates 8 lanes (Thread IDs 0-7).
When a warp is scheduled, all 8 lanes execute the instruction in a loop (unrolled for performance).

## Lane State Variables

Accessing lane state in kernels:

- `SR.LANEMASK`: Bitmask of active lanes (usually `0xFF`).
- `SR.LANEID`: Current lane ID (0-7).

## Predicated Execution

Instructions can be conditional based on the `P0-P7` predicate registers.
Example:

```python
# R1 = (R0 > 10) ? 1 : 0
Instruction.isetp_gt(4, 0, 10),  # P4 = (R0 > 10)
Instruction.mov(1, 1, predicate=4), # R1 = 1 if P4 is True
Instruction.mov(1, 0, predicate=4, not_p=True) # R1 = 0 if P4 is False
```
