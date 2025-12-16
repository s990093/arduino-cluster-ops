"""
ESP32 CUDA Tools Package
提供 ESP32 CUDA VM 的測試和控制工具
"""

from .connection import ESP32Connection
from .program_loader import Instruction, ProgramLoader
from .trace import TraceCollector
from .analyzer import ResultAnalyzer
from .tester import TransformerTester
from .simd_initializer import SIMDInitializer
from .runner import CUDARunner, quick_run
from .trace_parser import (
    parse_enhanced_trace,
    verify_trace_memory_values,
    save_trace_json
)

__all__ = [
    'ESP32Connection',
    'Instruction',
    'ProgramLoader',
    'TraceCollector',
    'ResultAnalyzer',
    'TransformerTester',
    'SIMDInitializer',
    'CUDARunner',
    'quick_run',
    'parse_enhanced_trace',
    'verify_trace_memory_values',
    'save_trace_json'
]

__version__ = '2.1.0'
