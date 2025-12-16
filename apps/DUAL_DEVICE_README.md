# Dual-Device Edge Convolution Implementation

## Overview

Created an accelerated version of the edge convolution application using **two ESP32 devices in parallel** for enhanced performance with **16 total SIMD lanes** (8 lanes per device).

## Files Created

### `app_edge_conv_dual.py`

Main application implementing dual-device parallel processing.

**Key Features:**

- **Dual-device architecture**: Uses two ESP32s simultaneously
- **Thread-based parallelism**: Worker threads for each device
- **Task queue system**: Efficient work distribution
- **Load balancing**: Round-robin task assignment
- **Result merging**: Synchronized collection of outputs
- **Compression support**: LZ4 compression for faster data transfer

**Architecture:**

```
Host (Python)
├── DualDeviceManager
│   ├── ESP32TurboClient (Device 0: /dev/cu.usbserial-589A0095521)
│   └── ESP32TurboClient (Device 1: /dev/cu.usbserial-2130)
├── Task Queue (shared)
└── Result Queue (shared)
```

### `test_dual_device.py`

Quick connectivity test script.

**Purpose:**

- Verify both devices are connected
- Test single tile processing on each device
- Check kernel execution and data transfer
- Validate basic functionality before full run

## Configuration

### Device Ports

- **Device 0**: `/dev/cu.usbserial-589A0095521`
- **Device 1**: `/dev/cu.usbserial-2130`

### Image Parameters

- **Resolution**: 1024x1024 (configurable)
- **Tile Size**: 128x32 per tile
- **Channels**: 3 (RGB)
- **Total Tasks**: (1024/128) × (1024/32) × 3 = 768 tasks

### Performance Benefits

- **Parallel H2D/D2H transfers**: Devices can upload/download simultaneously
- **Concurrent kernel execution**: No waiting for single device
- **16 lanes total**: Double the SIMD parallelism
- **Expected speedup**: ~1.8-1.9x (considering overhead)

## Usage

### Quick Test (Connectivity Check)

```bash
./venv/bin/python apps/test_dual_device.py
```

### Full Image Processing

```bash
./venv/bin/python apps/app_edge_conv_dual.py
```

## Implementation Details

### Work Distribution

Tasks are distributed round-robin across devices to ensure load balancing:

- Task 0 → Device 0
- Task 1 → Device 1
- Task 2 → Device 0
- Task 3 → Device 1
- ...

### Threading Model

```python
# Main thread: Task distribution
for task in tasks:
    task_queue.put(task)

# Worker threads (2): Process tasks
def worker_thread(device_id):
    while True:
        task = task_queue.get()
        result = device.process(task)
        result_queue.put(result)
```

### Synchronization

- **Task Queue**: Thread-safe task distribution
- **Result Queue**: Thread-safe result collection
- **Progress Tracking**: tqdm progress bar with total task count

## Verification

The application includes built-in verification:

1. **PyTorch Reference**: Computes ground truth using PyTorch
2. **MAE Calculation**: Mean Absolute Error between HW and reference
3. **Visualization**: Side-by-side comparison with difference map

## Output Statistics

Each run provides:

- **Per-device task count**
- **Compression ratios** (if LZ4 enabled)
- **Average kernel execution time**
- **Total execution time**
- **Verification metrics** (MAE, Max Diff)

## Next Steps

To test the implementation:

1. Ensure both ESP32 devices are connected
2. Upload firmware to both devices (if not already done)
3. Run `test_dual_device.py` for quick verification
4. Run `app_edge_conv_dual.py` for full image processing
