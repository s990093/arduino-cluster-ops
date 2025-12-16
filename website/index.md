---
layout: home

hero:
  name: "Micro-CUDA"
  text: "SIMT Architecture on ESP32"
  tagline: "Turn your ESP32-S3 into a CUDA-compatible Streaming Multiprocessor."
  image:
    src: /images/micro_arch_diag.png
    alt: Micro-CUDA Architecture
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View Source
      link: https://github.com/s990093/arduino-cluster-ops

features:
  - title: 8-Lane SIMT Engine
    details: True lockstep execution across 8 lanes, each with independent registers.
  - title: BFloat16 Tensor Cores
    details: Hardware accelerated matrix multiplication for deep learning inference.
  - title: Python Kernel Support
    details: Write kernels in Python and deploy instantly without recompiling firmware.
---

# 🏗️ Architecture: The Micro-SM

The system implements a classic **Front-End / Back-End** GPU architecture. The ESP32's dual cores are utilized to decouple instruction scheduling from parallel execution, emulating the internal structure of a discrete GPU Streaming Multiprocessor.

![ESP32 Micro-Architecture](/images/micro_arch_diag.png)

## SIMT Lane Architecture

Each SIMD lane operates in lockstep, possessing its own register context and execution units.

![SIMT Lane Architecture](/images/arch_diag.png)

## Micro-CUDA ISA v2.0 Specification

The ISA is a 32-bit RISC-style instruction set optimized for deep learning and tensor math.

![Instruction Encoding](/images/enc_diag.png)
