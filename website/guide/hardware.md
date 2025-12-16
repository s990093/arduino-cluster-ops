# Hardware & Bus Protocol

## Hardware Topology

The architecture avoids a shared bus in favor of a **Dual-Port Split-Bus** design to enable a true pipeline.

1.  **Global G-BUS (Upstream)**: AMB82-Mini $\to$ ESP32-S3 (50 MB/s).
2.  **Local G-BUS (Downstream)**: ESP32-S3 $\to$ RP2040 Array (Broadcast).

![High Level Pipeline](/images/pipeline_diagram_placeholder.png)
_(Note: Diagram placeholders refer to TikZ figures in the paper)_

## Physical Bus Interface

We use a custom **8-bit Parallel Low-Latency Bus** (Intel 8080-style).

- **Bandwidth**: ~50 MB/s (20ns cycle time)
- **Voltage**: 3.3V CMOS
- **Transmission**: Big-Endian, Burst Mode

### Pin Definition

| Signal   | Type    | Description                                 |
| :------- | :------ | :------------------------------------------ |
| `D[0:7]` | Data    | 8-bit bidirectional data bus                |
| `CS#`    | Control | Chip Select (Active Low)                    |
| `DC`     | Control | **Low**: Command / **High**: Data           |
| `WR#`    | Clock   | Write Strobe (Slave latches on Rising Edge) |
| `SYNC`   | Global  | **Warp Trigger** (Global Barrier Release)   |

## ESP32-S3 Pin Mapping

The ESP32-S3 acts as a router/scheduler, managing simultaneous RX (from Host) and TX (to SMSP Cores).

### Input Interface (Slave)

| Signal          | Pin      | Function                     |
| :-------------- | :------- | :--------------------------- |
| `G_DATA_[0..7]` | GPIO 1-9 | Data Input (Skipping GPIO 3) |
| `G_WR`          | GPIO 10  | Write Strobe                 |
| `G_DC`          | GPIO 11  | Data/Command                 |

### Output Interface (Master)

| Signal          | Pin        | Function           |
| :-------------- | :--------- | :----------------- |
| `L_DATA_[0..3]` | GPIO 15-18 | Low Nibble         |
| `L_DATA_[4..7]` | GPIO 39-42 | High Nibble        |
| `L_WR`          | GPIO 48    | Write Strobe       |
| `SYNC_TRIG`     | GPIO 46    | **Global Barrier** |

## Timing & Integrity

### Cycle Timing

The effective write cycle is **20 ns** (50 MHz).

- **Setup Time**: Data must be stable 5ns before `WR#` rising edge.
- **Hold Time**: Data must be held 3ns after `WR#` rising edge.

### Implementation Notes

1.  **Length Matching**: `D0-D7` traces should be matched to within ±1mm.
2.  **Drive Strength**: Configure GPIOs for **20mA** drive strength.
3.  **Grounding**: A solid common ground plane is required between all 3 layers (AMB82, ESP32, RP2040s).
