#!/usr/bin/env python3
"""
ESP32 CUDA VM - Enhanced Trace Parser

Provides robust parsing utilities for JSON trace output from the ESP32 CUDA VM.
Includes error recovery and salvage logic for corrupted or incomplete traces.
"""

import json
import re
from typing import Dict, List, Optional, Any


def parse_enhanced_trace(trace_output: str, initial_mem: Optional[List[Dict]] = None) -> Optional[Dict[str, Any]]:
    """
    Parses the enhanced trace output with robust error handling and salvaging.
    
    Args:
        trace_output: Raw text output containing JSON trace data
        initial_mem: Optional list of initial memory state dicts with 'addr' and 'val' keys
        
    Returns:
        Parsed trace dictionary with structure:
        {
            "trace_version": str,
            "architecture": str,
            "program": str,
            "warp_size": int,
            "total_instructions": int,
            "records": List[Dict],
            "initial_memory": List[Dict]
        }
        Returns None if parsing completely fails.
        
    Example:
        >>> trace_data = parse_enhanced_trace(raw_output, initial_mem=[{"addr": 0, "val": 123}])
        >>> print(f"Parsed {trace_data['total_instructions']} instructions")
    """
    print("=" * 70)
    print("🔍 Parsing Enhanced Trace")
    print("=" * 70)

    # 1. Extract JSON part
    json_start = trace_output.find('{')
    json_end = trace_output.rfind('}')
    
    if json_start == -1 or json_end == -1:
        print("❌ Error: No JSON content found in output")
        return None

    json_str = trace_output[json_start : json_end + 1]
    
    # 2. Try Global Parse
    try:
        data = json.loads(json_str)
        # Ensure metadata exists
        if "total_instructions" not in data:
             data["total_instructions"] = len(data.get("records", []))
        if initial_mem:
             data["initial_memory"] = initial_mem
        print("✅ Global JSON Parse Successful")
        return data
    except json.JSONDecodeError as e:
        print(f"⚠️  Global JSON Parse Failed: {e}")
        print("   Attempting to parse individual records...")

    # 3. Robust Salvage Logic (Cycle-Key Splitter)
    return _salvage_trace_records(json_str, initial_mem)


def _salvage_trace_records(json_str: str, initial_mem: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """
    Internal helper to salvage individual records from corrupted JSON trace.
    
    Uses regex pattern matching to extract valid JSON objects even when
    the overall structure is malformed.
    """
    trace_version = "2.1"
    architecture = "SIMT" 
    program = "GPU-Like Kernel"
    warp_size = 8
    
    # Extract header info
    v_match = re.search(r'"trace_version":\s*"([^"]+)"', json_str)
    if v_match: 
        trace_version = v_match.group(1)
    
    # Isolate records array
    recs_start = json_str.find('"records": [')
    recs_str = json_str[recs_start + 12:] if recs_start != -1 else json_str

    # Split by record start pattern: {"cycle":
    record_pattern = re.compile(r'\{\s*"cycle":')
    matches = list(record_pattern.finditer(recs_str))
    
    parsed_records = []
    
    if not matches:
        print("❌ No records found to salvage")
        return {
            "trace_version": trace_version,
            "records": [], 
            "total_instructions": 0,
            "initial_memory": initial_mem or []
        }

    print(f"   ℹ️  Split into {len(matches)} potential segments (Cycle-Key Strategy)")
    
    for i in range(len(matches)):
        start_idx = matches[i].start()
        end_idx = matches[i+1].start() if i < len(matches) - 1 else len(recs_str)
        
        # Extract chunk and clean it
        chunk = recs_str[start_idx:end_idx].strip()
        if chunk.endswith(','): 
            chunk = chunk[:-1]
        if chunk.endswith(']'): 
            chunk = chunk[:-1].rstrip()
        
        # Try parse individual
        try:
            rec = json.loads(chunk)
            parsed_records.append(rec)
        except json.JSONDecodeError:
            # Local Salvage
            salvaged = _salvage_single_record(chunk, i)
            if salvaged:
                parsed_records.append(salvaged)
    
    print(f"   ℹ️  Recovered {len(parsed_records)} valid/salvaged records")
    
    return {
        "trace_version": trace_version,
        "architecture": architecture,
        "program": program,
        "warp_size": warp_size,
        "total_instructions": len(parsed_records),
        "records": parsed_records,
        "initial_memory": initial_mem or []
    }


def _salvage_single_record(chunk: str, record_index: int) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract usable data from a corrupted single JSON record.
    
    Args:
        chunk: Raw JSON string of a single record
        record_index: Index of this record in the overall trace
        
    Returns:
        Salvaged record dict or None if extraction fails
    """
    # Extract basic fields
    cycle_m = re.search(r'"cycle":\s*(\d+)', chunk)
    pc_m = re.search(r'"pc":\s*(\d+)', chunk)
    asm_m = re.search(r'"asm":\s*"([^"]+)"', chunk)
    
    # Extract memory_access
    mem_data = []
    mem_m = re.search(r'"memory_access":\s*(\[\s*\{.*?\}\s*\])', chunk, re.DOTALL)
    if mem_m:
        try:
            mem_data = json.loads(mem_m.group(1))
            print(f"      🚑 Salvaged memory_access for cycle {cycle_m.group(1) if cycle_m else '?'}")
        except:
            pass
    
    if cycle_m and pc_m:
        rec = {
            "cycle": int(cycle_m.group(1)),
            "pc": int(pc_m.group(1)),
            "instruction": "0x00000000",
            "asm": asm_m.group(1) if asm_m else "SALVAGED_INSTRUCTION",
            "exec_time_us": 0,
            "hw_ctx": {"sm_id": 0, "warp_id": 0, "active_mask": "0xFF"},
            "perf": {"latency": 1, "stall_cycles": 0, "simd_width": 8},
            "lanes": [],  # Omitted to save space on salvage
            "memory_access": mem_data
        }
        print(f"      🔧 Reconstructed record #{record_index} (Cycle {rec['cycle']})")
        return rec
    
    return None


def verify_trace_memory_values(trace_data: Dict[str, Any], expected_nonzero: bool = True) -> Dict[str, Any]:
    """
    Verify memory access values in parsed trace data.
    
    Args:
        trace_data: Parsed trace dictionary from parse_enhanced_trace()
        expected_nonzero: If True, expect to find non-zero memory values
        
    Returns:
        Dictionary with verification results:
        {
            "total_records": int,
            "records_with_memory": int,
            "nonzero_values": int,
            "passed": bool
        }
    """
    mem_ops = 0
    non_zero_mem = 0
    
    for rec in trace_data.get("records", []):
        if rec.get("memory_access"):
            mem_ops += 1
            for ma in rec["memory_access"]:
                val = ma.get("val", 0)
                if val != 0:
                    non_zero_mem += 1
    
    passed = (non_zero_mem > 0) if expected_nonzero else True
    
    return {
        "total_records": len(trace_data.get("records", [])),
        "records_with_memory": mem_ops,
        "nonzero_values": non_zero_mem,
        "passed": passed
    }


def save_trace_json(trace_data: Dict[str, Any], filepath: str) -> None:
    """
    Save parsed trace data to JSON file.
    
    Args:
        trace_data: Parsed trace dictionary
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(trace_data, f, indent=2)
    print(f"✅ Trace saved to: {filepath}")
