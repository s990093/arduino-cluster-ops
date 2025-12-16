// TypeScript interfaces for GPU Trace data structure

export interface Lane {
  lane_id: number;
  R: number[]; // Register file (typically 24 registers)
  F?: number[]; // Optional float registers
  P?: number[]; // Optional predicate registers
}

export interface HardwareContext {
  sm_id: number;
  warp_id: number;
  lane_id: number;
  active_mask: string;
}

export interface PerformanceMetrics {
  latency: number;
  stall_cycles: number;
  stall_reason: string;
  pipe_stage: string;
  core_id: number;
  simd_width: number;
}

export interface MemoryAccess {
  lane: number;
  type: 'read' | 'write';
  addr: number;
  val: number;
}

export interface TraceRecord {
  cycle: number;
  pc: number;
  instruction: string;
  asm: string;
  exec_time_us: number;
  hw_ctx: HardwareContext;
  perf: PerformanceMetrics;
  lanes: Lane[];
  memory_access?: MemoryAccess[];
}

export interface TraceData {
  trace_version: string;
  program: string;
  total_instructions: number;
  records: TraceRecord[];
  initial_memory?: { addr: number; val: number }[];
}
