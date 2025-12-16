"""
Target Configuration for Micro-CUDA Compiler

This module defines the target hardware configuration for compilation.
It records ESP32 CUDA VM parameters like VRAM size, lane count, etc.
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TargetConfig:
    """Target hardware configuration"""
    
    # Device Information
    device_name: str = "ESP32 CUDA VM"
    isa_version: str = "v1.5"
    architecture: str = "Dual-Core SIMT"
    
    # SIMD Configuration
    num_lanes: int = 8              # Number of SIMD lanes
    warp_size: int = 8              # Warp size (always equals num_lanes)
    
    # Memory Configuration (in bytes)
    vram_size: int = 40960          # 40 KB (default)
    vram_min: int = 4096            # 4 KB (minimum)
    vram_max: int = 1048576         # 1 MB (with PSRAM)
    
    program_size: int = 1024        # Max number of instructions
    stack_size: int = 8192          # FreeRTOS stack size
    queue_size: int = 16            # Instruction queue depth
    
    # Register Configuration (per lane)
    num_gpr: int = 32               # General purpose registers (R0-R31)
    num_fpr: int = 32               # Floating point registers (F0-F31)
    num_pred: int = 8               # Predicate registers (P0-P7)
    num_sr: int = 10                # System registers (SR_0 - SR_9)
    
    # Communication Configuration
    baud_rate: int = 115200         # Serial baud rate
    
    # CPU Configuration
    cpu_freq_mhz: int = 240         # ESP32 CPU frequency
    
    # Performance Characteristics
    typical_inst_per_sec: int = 30000   # Typical instructions per second
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "device_name": self.device_name,
            "isa_version": self.isa_version,
            "architecture": self.architecture,
            "num_lanes": self.num_lanes,
            "warp_size": self.warp_size,
            "vram_size": self.vram_size,
            "program_size": self.program_size,
            "num_gpr": self.num_gpr,
            "num_fpr": self.num_fpr,
            "baud_rate": self.baud_rate,
            "cpu_freq_mhz": self.cpu_freq_mhz,
        }
    
    def format_header(self) -> str:
        """Format configuration as assembly comment header"""
        header = []
        header.append("; " + "=" * 68)
        header.append("; Micro-CUDA Kernel - Compiled Assembly")
        header.append("; " + "=" * 68)
        header.append(";")
        header.append("; Target Configuration:")
        header.append(f";   Device:        {self.device_name}")
        header.append(f";   ISA Version:   {self.isa_version}")
        header.append(f";   Architecture:  {self.architecture}")
        header.append(";")
        header.append("; SIMD Configuration:")
        header.append(f";   Lanes:         {self.num_lanes}")
        header.append(f";   Warp Size:     {self.warp_size}")
        header.append(";")
        header.append("; Memory Configuration:")
        header.append(f";   VRAM Size:     {self.vram_size} bytes ({self.vram_size // 1024} KB)")
        header.append(f";   Program Size:  {self.program_size} instructions")
        header.append(f";   Stack Size:    {self.stack_size} bytes")
        header.append(";")
        header.append("; Register Configuration (per lane):")
        header.append(f";   GP Registers:  R0-R{self.num_gpr-1} ({self.num_gpr} × 32-bit)")
        header.append(f";   FP Registers:  F0-F{self.num_fpr-1} ({self.num_fpr} × 32-bit)")
        header.append(f";   Predicates:    P0-P{self.num_pred-1} ({self.num_pred} × 1-bit)")
        header.append(f";   System Regs:   SR_0 - SR_{self.num_sr-1}")
        header.append(";")
        header.append("; Communication:")
        header.append(f";   Serial Baud:   {self.baud_rate}")
        header.append(f";   CPU Freq:      {self.cpu_freq_mhz} MHz")
        header.append(";")
        header.append("; Performance:")
        header.append(f";   Typical Speed: ~{self.typical_inst_per_sec:,} inst/sec")
        header.append(";")
        header.append("; " + "=" * 68)
        header.append("")
        
        return "\n".join(header)
    
    def validate_memory_layout(self, region_sizes: Dict[str, int]) -> bool:
        """
        Validate that memory layout fits in VRAM
        
        Args:
            region_sizes: Dictionary of region name -> size in bytes
        
        Returns:
            True if layout is valid
        """
        total_size = sum(region_sizes.values())
        
        if total_size > self.vram_size:
            print(f"[ERROR] Memory layout exceeds VRAM size!")
            print(f"  Total required: {total_size} bytes")
            print(f"  VRAM available: {self.vram_size} bytes")
            print(f"  Overflow: {total_size - self.vram_size} bytes")
            return False
        
        return True

# Default target configuration
DEFAULT_TARGET = TargetConfig()

# Alternative configurations for different ESP32 variants

ESP32_STANDARD = TargetConfig(
    device_name="ESP32 (Standard)",
    vram_size=32768,  # 32 KB (conservative)
)

ESP32_WITH_PSRAM = TargetConfig(
    device_name="ESP32 with 2MB PSRAM",
    vram_size=102400,  # 100 KB
)

ESP32_S3_8MB_PSRAM = TargetConfig(
    device_name="ESP32-S3 with 8MB PSRAM",
    vram_size=1048576,  # 1 MB
    cpu_freq_mhz=240,
)

# Available target configurations
AVAILABLE_TARGETS = {
    "default": DEFAULT_TARGET,
    "esp32": ESP32_STANDARD,
    "esp32-psram": ESP32_WITH_PSRAM,
    "esp32s3": ESP32_S3_8MB_PSRAM,
}

def get_target(name: str = "default") -> TargetConfig:
    """
    Get target configuration by name
    
    Args:
        name: Target name (default, esp32, esp32-psram, esp32s3)
    
    Returns:
        TargetConfig instance
    """
    if name not in AVAILABLE_TARGETS:
        print(f"[WARNING] Unknown target '{name}', using default")
        return DEFAULT_TARGET
    
    return AVAILABLE_TARGETS[name]

def list_targets():
    """Print available target configurations"""
    print("Available Target Configurations:")
    print()
    for name, config in AVAILABLE_TARGETS.items():
        print(f"  {name:15} - {config.device_name}")
        print(f"                   VRAM: {config.vram_size // 1024} KB, "
              f"Lanes: {config.num_lanes}, "
              f"CPU: {config.cpu_freq_mhz} MHz")
    print()
    print("Usage: --target <name>")
