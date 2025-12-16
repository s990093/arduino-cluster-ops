"use client";

import React, { useState, useEffect, useMemo, useRef } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Activity,
  Layers,
  Cpu,
  Database,
  Clock,
  AlertTriangle,
  X,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";
import { TraceData } from "@/types/trace";

interface TraceAnalyzerProps {
  traceData: TraceData;
  onReset: () => void;
}

export default function TraceAnalyzer({
  traceData,
  onReset,
}: TraceAnalyzerProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeTab, setActiveTab] = useState("trace");
  const [selectedLane, setSelectedLane] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Playback control
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying) {
      interval = setInterval(() => {
        setCurrentIndex((prev) => {
          if (prev >= traceData.records.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 800);
    }
    return () => clearInterval(interval);
  }, [isPlaying, traceData.records.length]);

  // Auto scroll
  useEffect(() => {
    if (scrollRef.current && activeTab === "trace") {
      const activeRow = scrollRef.current.querySelector(
        `[data-index="${currentIndex}"]`
      );
      if (activeRow) {
        activeRow.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }
  }, [currentIndex, activeTab]);

  const currentRecord = traceData.records[currentIndex];

  // Calculate statistics
  const stats = useMemo(() => {
    const totalCycles = traceData.records[traceData.records.length - 1].cycle;
    const avgLatency =
      traceData.records.reduce((acc, r) => acc + r.perf.latency, 0) /
      traceData.total_instructions;
    return {
      totalInstructions: traceData.total_instructions,
      totalCycles,
      avgLatency: avgLatency.toFixed(2),
      program: traceData.program,
      simdWidth: currentRecord.perf.simd_width || 8,
    };
  }, [currentRecord, traceData]);

  // Stall data processing
  const stallData = useMemo(() => {
    const reasonCounts: { [key: string]: number } = {};
    traceData.records.forEach((r) => {
      const reason = r.perf.stall_reason;
      reasonCounts[reason] = (reasonCounts[reason] || 0) + 1;
    });
    return Object.keys(reasonCounts).map((reason) => ({
      name: reason,
      value: reasonCounts[reason],
    }));
  }, [traceData.records]);

  // Warp scheduling data
  const warpData = useMemo(() => {
    return traceData.records.map((r) => ({
      cycle: r.cycle,
      warpId: r.hw_ctx.warp_id,
      stallReason: r.perf.stall_reason,
    }));
  }, [traceData.records]);

  // Memory map processing
  const memoryMap = useMemo(() => {
    const map = new Map<
      number,
      { value: number; lastUpdateCycle: number; type: "read" | "write" }
    >();

    // Load initial memory if available
    if (traceData.initial_memory) {
      traceData.initial_memory.forEach((m: any) => {
        map.set(m.addr, {
          value: m.val,
          lastUpdateCycle: -1, // Initial state
          type: "write",
        });
      });
    }

    for (let i = 0; i <= currentIndex; i++) {
      const rec = traceData.records[i];
      if (rec.memory_access && rec.memory_access.length > 0) {
        rec.memory_access.forEach((m) => {
          map.set(m.addr, {
            value: m.val,
            lastUpdateCycle: rec.cycle,
            type: m.type,
          });
        });
      }
    }
    const grid = [];
    // Show 0-255 (256 bytes = 64 words)
    for (let i = 0; i < 256; i += 4) {
      grid.push({
        address: i,
        ...(map.get(i) || {
          value: 0,
          lastUpdateCycle: -1,
          type: "read" as const,
        }),
      });
    }
    return grid;
  }, [currentIndex, traceData.records, traceData.initial_memory]);

  // --- NEW: Advanced Visualizations Data Prep ---

  // 1. Memory Access Timing (Cycle vs Address)
  const memoryAccessPoints = useMemo(() => {
    const points: any[] = [];
    traceData.records.forEach((rec) => {
      if (rec.memory_access) {
        rec.memory_access.forEach((m) => {
          points.push({
            cycle: rec.cycle,
            address: m.addr,
            lane: m.lane,
            type: m.type,
            value: m.val,
          });
        });
      }
    });
    return points;
  }, [traceData.records]);

  // 2. Register Write Timing (Cycle vs RegIndex)
  const registerWritePoints = useMemo(() => {
    const points: any[] = [];
    traceData.records.forEach((rec) => {
      // Simple heuristic: Extract destination register from ASM
      // e.g. "MOV R15, 128" -> Dest R15
      // e.g. "IMUL R30, R31, R29" -> Dest R30
      // e.g. "LDX R0, ..." -> Dest R0

      const parts = rec.asm.split(" ");
      if (parts.length > 1) {
        const destPart = parts[1]; // Usually "Rxx," or "Rxx"
        if (destPart.startsWith("R")) {
          const regStr = destPart.replace(",", "").substring(1); // Remove 'R' and ','
          const regIdx = parseInt(regStr);
          if (!isNaN(regIdx)) {
            points.push({
              cycle: rec.cycle,
              reg: regIdx,
              asm: rec.asm,
            });
          }
        }
      }
    });
    return points;
  }, [traceData.records]);

  // 3. Memory Layout Detection
  const memorySegments = useMemo(() => {
    if (!traceData.initial_memory) return [];

    const sorted = [...traceData.initial_memory].sort(
      (a, b) => a.addr - b.addr
    );
    const segments: {
      start: number;
      count: number;
      values: number[];
      label: string;
    }[] = [];

    if (sorted.length === 0) return segments;

    let currentSegment = {
      start: sorted[0].addr,
      count: 1,
      values: [sorted[0].val],
    };

    for (let i = 1; i < sorted.length; i++) {
      const item = sorted[i];
      const prev = sorted[i - 1];

      // Check if contiguous (4 bytes apart)
      if (item.addr === prev.addr + 4) {
        currentSegment.count++;
        currentSegment.values.push(item.val);
      } else {
        // End current segment
        segments.push({
          ...currentSegment,
          label: getSegmentLabel(currentSegment.start),
        });
        // Start new
        currentSegment = {
          start: item.addr,
          count: 1,
          values: [item.val],
        };
      }
    }
    // Push last
    segments.push({
      ...currentSegment,
      label: getSegmentLabel(currentSegment.start),
    });

    return segments;
  }, [traceData.initial_memory]);

  function getSegmentLabel(startAddr: number): string {
    if (startAddr === 0) return "Input I";
    if (startAddr === 64) return "Kernel K"; // 0x40
    if (startAddr === 96) return "V_data / Padding"; // 0x60
    return "Data Block";
  }

  const COLORS = ["#10B981", "#F59E0B", "#EF4444", "#3B82F6", "#8B5CF6"];

  const TabButton = ({
    id,
    icon: Icon,
    label,
  }: {
    id: string;
    icon: React.ElementType;
    label: string;
  }) => (
    <button
      onClick={() => setActiveTab(id)}
      className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium transition-colors border-b-2 
        ${
          activeTab === id
            ? "border-emerald-500 text-emerald-400 bg-slate-800"
            : "border-transparent text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
        }`}
    >
      <Icon size={16} />
      <span>{label}</span>
    </button>
  );

  const currentLaneRegisters =
    currentRecord.lanes && currentRecord.lanes[selectedLane]
      ? currentRecord.lanes[selectedLane].R
      : [];

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-200 font-sans overflow-hidden">
      {/* Top header */}
      <header className="flex items-center justify-between px-6 py-4 bg-slate-900 border-b border-slate-800">
        <div className="flex items-center space-x-3">
          <Cpu className="text-emerald-500" size={24} />
          <div>
            <h1 className="text-lg font-bold text-white tracking-wide">
              GPU Trace Architect
            </h1>
            <p className="text-xs text-slate-400">
              {stats.program} | Ver {traceData.trace_version}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-6">
          <div className="flex space-x-6 text-sm">
            <div className="flex flex-col items-end">
              <span className="text-slate-500 uppercase text-[10px] tracking-wider">
                SIMD Width
              </span>
              <span className="font-mono text-purple-400">
                {stats.simdWidth} Lanes
              </span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-slate-500 uppercase text-[10px] tracking-wider">
                Instructions
              </span>
              <span className="font-mono text-emerald-400">
                {stats.totalInstructions}
              </span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-slate-500 uppercase text-[10px] tracking-wider">
                Current Cycle
              </span>
              <span className="font-mono text-blue-400">
                {currentRecord.cycle}
              </span>
            </div>
          </div>
          <button
            onClick={onReset}
            className="p-2 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
            title="Load new trace"
          >
            <X size={20} />
          </button>
        </div>
      </header>

      {/* Navigation tabs */}
      <div className="flex border-b border-slate-800 bg-slate-900/50 px-6">
        <TabButton id="trace" icon={Activity} label="Cycle-level Trace" />
        <TabButton id="stalls" icon={AlertTriangle} label="Stall Analysis" />
        <TabButton id="memory" icon={Database} label="Memory Heatmap" />
        <TabButton id="warp" icon={Layers} label="Warp Scheduling" />
        <TabButton id="timing" icon={Clock} label="Memory Timing" />
        <TabButton id="registers" icon={Cpu} label="Register Timing" />
      </div>

      {/* Main content */}
      <main className="flex-1 overflow-hidden flex relative">
        <div className="flex-1 p-6 bg-slate-950 overflow-hidden">
          {/* Trace View */}
          {activeTab === "trace" && (
            <div className="grid grid-cols-12 gap-6 h-full">
              {/* Instruction list */}
              <div className="col-span-8 bg-slate-900 rounded-lg border border-slate-800 flex flex-col h-full shadow-lg overflow-hidden">
                <div className="px-4 py-2 border-b border-slate-800 bg-slate-800/50 text-xs font-semibold text-slate-300 flex justify-between shrink-0">
                  <span>INSTRUCTION STREAM</span>
                  <span className="text-slate-500">
                    PC: 0x{currentRecord.pc.toString(16).padStart(4, "0")}
                  </span>
                </div>
                <div
                  className="flex-1 overflow-auto p-1 space-y-0.5 font-mono text-xs"
                  ref={scrollRef}
                >
                  {traceData.records.map((record, idx) => (
                    <div
                      key={idx}
                      data-index={idx}
                      className={`grid grid-cols-12 gap-1 p-1 rounded cursor-pointer transition-all border-l-2
                          ${
                            idx === currentIndex
                              ? "bg-emerald-500/10 border-emerald-500 text-white shadow-[0_0_15px_rgba(16,185,129,0.1)]"
                              : "border-transparent text-slate-500 hover:bg-slate-800"
                          }`}
                      onClick={() => {
                        setCurrentIndex(idx);
                        setIsPlaying(false);
                      }}
                    >
                      <div className="col-span-1 text-slate-600">
                        #{record.cycle}
                      </div>
                      <div className="col-span-2 text-blue-400">
                        0x{record.pc.toString(16)}
                      </div>
                      <div className="col-span-6">{record.asm}</div>
                      <div className="col-span-3 flex items-center justify-end space-x-2">
                        {record.perf.stall_reason !== "NONE" && (
                          <span className="px-1.5 py-0.5 text-[10px] bg-red-900/30 text-red-400 rounded border border-red-900/50">
                            {record.perf.stall_reason}
                          </span>
                        )}
                        <span className="text-xs text-slate-600">
                          W{record.hw_ctx.warp_id}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Detail panel */}
              <div className="col-span-4 space-y-6 flex flex-col h-full overflow-y-auto pr-2">
                {/* Register file */}
                <div className="bg-slate-900 rounded-lg border border-slate-800 shadow-lg shrink-0">
                  <div className="px-4 py-2 border-b border-slate-800 bg-slate-800/50 text-xs font-semibold text-slate-300 flex items-center justify-between">
                    <div className="flex items-center">
                      <Cpu size={14} className="mr-2" /> REGISTER FILE
                    </div>
                    <div className="flex space-x-1">
                      {Array.from({ length: 8 }).map((_, i) => (
                        <button
                          key={i}
                          onClick={() => setSelectedLane(i)}
                          className={`w-5 h-5 flex items-center justify-center rounded text-[9px] font-mono transition-colors
                            ${
                              selectedLane === i
                                ? "bg-emerald-500 text-black font-bold"
                                : "bg-slate-800 text-slate-500 hover:bg-slate-700"
                            }`}
                        >
                          {i}
                        </button>
                      ))}
                    </div>
                  </div>
                  <div className="p-2 text-center text-[10px] text-slate-500 border-b border-slate-800/30">
                    Showing Registers for <strong>Lane {selectedLane}</strong>{" "}
                    (SIMD Thread)
                  </div>
                  <div className="p-4 grid grid-cols-8 gap-1 font-mono text-[10px]">
                    {currentLaneRegisters.map((val, idx) => (
                      <div
                        key={idx}
                        className={`p-0.5 rounded flex flex-col items-center justify-center border border-slate-800/50 min-h-[32px]
                        ${
                          val !== 0
                            ? "bg-slate-800 text-emerald-400"
                            : "bg-slate-950 text-slate-600"
                        }`}
                      >
                        <span className="text-[8px] text-slate-500 leading-none mb-0.5">
                          R{idx}
                        </span>
                        <span className="font-bold leading-none">{val}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Pipeline stage */}
                <div className="bg-slate-900 rounded-lg border border-slate-800 p-4 shadow-lg shrink-0">
                  <h3 className="text-xs font-semibold text-slate-400 mb-3 uppercase tracking-wider">
                    Pipeline Stage
                  </h3>
                  <div className="flex items-center justify-between relative">
                    <div className="absolute top-1/2 left-0 w-full h-0.5 bg-slate-800 -z-10"></div>
                    {["FETCH", "DECODE", "EXEC", "MEM", "WB"].map((stage) => {
                      const isActive =
                        currentRecord.perf.pipe_stage ===
                        (stage === "WB" ? "WRITEBACK" : stage);
                      return (
                        <div
                          key={stage}
                          className="flex flex-col items-center space-y-2 z-10 bg-slate-900 px-1"
                        >
                          <div
                            className={`w-3 h-3 rounded-full border-2 ${
                              isActive
                                ? "bg-emerald-500 border-emerald-500 shadow-[0_0_10px_#10B981]"
                                : "bg-slate-800 border-slate-700"
                            }`}
                          ></div>
                          <span
                            className={`text-[10px] font-bold ${
                              isActive ? "text-emerald-400" : "text-slate-600"
                            }`}
                          >
                            {stage}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                  <div className="mt-6 pt-4 border-t border-slate-800 flex justify-between text-xs">
                    <div>
                      <span className="block text-slate-500">Latency</span>
                      <span className="text-slate-200">
                        {currentRecord.perf.latency} cycles
                      </span>
                    </div>
                    <div>
                      <span className="block text-slate-500">Exec Time</span>
                      <span className="text-slate-200">
                        {currentRecord.exec_time_us} μs
                      </span>
                    </div>
                    <div>
                      <span className="block text-slate-500">Mask</span>
                      <span className="font-mono text-slate-200">
                        {currentRecord.hw_ctx.active_mask}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          {/* Stall Analysis View */}
          {activeTab === "stalls" && (
            <div className="grid grid-cols-2 gap-6 h-full overflow-y-auto">
              <div className="bg-slate-900 rounded-lg border border-slate-800 p-6 flex flex-col items-center justify-center relative shadow-lg min-h-[300px]">
                <h3 className="absolute top-4 left-4 text-sm font-semibold text-slate-300 flex items-center">
                  <AlertTriangle size={16} className="mr-2 text-amber-500" />
                  STALL DISTRIBUTION
                </h3>
                {stallData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={stallData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        fill="#8884d8"
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {stallData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={COLORS[index % COLORS.length]}
                          />
                        ))}
                      </Pie>
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: "#0f172a",
                          borderColor: "#1e293b",
                          color: "#f1f5f9",
                        }}
                      />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="text-slate-500">
                    No stall data available in this trace segment.
                  </div>
                )}
              </div>

              <div className="bg-slate-900 rounded-lg border border-slate-800 p-6 relative shadow-lg min-h-[300px]">
                <h3 className="absolute top-4 left-4 text-sm font-semibold text-slate-300 flex items-center">
                  <Clock size={16} className="mr-2 text-blue-500" />
                  LATENCY OVER TIME
                </h3>
                <div className="mt-8 h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={traceData.records.slice(
                        0,
                        Math.max(currentIndex + 1, 10)
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="cycle" stroke="#64748b" fontSize={10} />
                      <YAxis stroke="#64748b" fontSize={10} />
                      <RechartsTooltip
                        contentStyle={{
                          backgroundColor: "#0f172a",
                          borderColor: "#1e293b",
                          color: "#f1f5f9",
                        }}
                      />
                      <Line
                        type="stepAfter"
                        dataKey="perf.latency"
                        stroke="#3B82F6"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
          {/* Memory Heatmap View */}

          {activeTab === "memory" && (
            <div className="flex h-full gap-6">
              {/* Left: Grid */}
              <div className="flex-1 bg-slate-900 rounded-lg border border-slate-800 p-6 shadow-lg overflow-y-auto">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-lg font-semibold text-slate-100 flex items-center">
                    <Database size={20} className="mr-2 text-emerald-500" />
                    MEMORY ADDRESS SPACE (0x00 - 0xFF)
                  </h3>
                  <div className="flex items-center space-x-4 text-xs">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-slate-700 mr-2 rounded"></div>
                      <span className="text-slate-400">Initialized</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-emerald-500/50 mr-2 rounded"></div>
                      <span className="text-slate-400">Written</span>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-8 gap-1">
                  {memoryMap.map((cell) => {
                    // Determine cell style
                    let bgColor = "bg-slate-800";
                    let textColor = "text-slate-600";
                    const isInitialized =
                      cell.lastUpdateCycle === -1 && cell.value !== 0; // Heuristic for init
                    const isWritten = cell.lastUpdateCycle > -1;

                    if (isWritten) {
                      bgColor = "bg-emerald-900/40 border-emerald-500/50";
                      textColor = "text-emerald-400 font-bold";
                    } else if (isInitialized) {
                      bgColor = "bg-slate-700/50 border-slate-600";
                      textColor = "text-slate-300";
                    }

                    return (
                      <div
                        key={cell.address}
                        className={`
                                    relative p-2 rounded border ${bgColor} 
                                    transition-all hover:scale-105 hover:z-10 h-16 flex flex-col justify-between
                                `}
                        title={`Addr: ${
                          cell.address
                        } (0x${cell.address.toString(16)})`}
                      >
                        <div className="flex justify-between items-start">
                          <span className="text-[10px] font-mono text-slate-500">
                            0x
                            {cell.address
                              .toString(16)
                              .toUpperCase()
                              .padStart(2, "0")}
                          </span>
                        </div>
                        <div
                          className={`text-right ${textColor} font-mono text-sm truncate`}
                        >
                          {cell.value}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Right: Detected Segments Legend */}
              <div className="w-80 bg-slate-900 rounded-lg border border-slate-800 flex flex-col shadow-lg">
                <div className="px-4 py-3 border-b border-slate-800 bg-slate-800/50">
                  <h4 className="text-sm font-semibold text-slate-200">
                    Memory Layout
                  </h4>
                  <p className="text-xs text-slate-500">
                    Auto-detected contiguous blocks
                  </p>
                </div>
                <div className="p-0 overflow-auto flex-1">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead className="bg-slate-800/80 sticky top-0 text-slate-400">
                      <tr>
                        <th className="px-3 py-2 font-medium border-b border-slate-700">
                          Addr
                        </th>
                        <th className="px-2 py-2 font-medium border-b border-slate-700">
                          Count
                        </th>
                        <th className="px-3 py-2 font-medium border-b border-slate-700">
                          Label/Content
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {memorySegments.length === 0 ? (
                        <tr>
                          <td
                            colSpan={3}
                            className="px-4 py-8 text-center text-slate-600 italic"
                          >
                            No initialized segments found.
                          </td>
                        </tr>
                      ) : (
                        memorySegments.map((seg, idx) => (
                          <tr
                            key={idx}
                            className="hover:bg-slate-800/50 transition-colors"
                          >
                            <td className="px-3 py-2 font-mono text-blue-400">
                              0x
                              {seg.start
                                .toString(16)
                                .toUpperCase()
                                .padStart(2, "0")}
                            </td>
                            <td className="px-2 py-2 text-slate-300">
                              {seg.count}
                            </td>
                            <td className="px-3 py-2">
                              <div className="font-semibold text-emerald-400 mb-0.5">
                                {seg.label}
                              </div>
                              <div
                                className="text-slate-500 truncate max-w-[120px]"
                                title={seg.values.join(", ")}
                              >
                                [{seg.values.slice(0, 3).join(", ")}
                                {seg.values.length > 3 ? "..." : ""}]
                              </div>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
                <div className="p-3 bg-slate-900 border-t border-slate-800 text-[10px] text-slate-500">
                  * Based on VRAM initialization snapshot
                </div>
              </div>
            </div>
          )}
          {/* Warp Scheduling View */}
          {activeTab === "warp" && (
            <div className="h-full flex flex-col space-y-6 overflow-y-auto">
              <div className="bg-slate-900 rounded-lg border border-slate-800 p-6 flex-1 shadow-lg min-h-[400px]">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-sm font-semibold text-slate-300 flex items-center">
                    <Layers size={16} className="mr-2 text-purple-500" />
                    WARP SCHEDULER TIMELINE
                  </h3>
                </div>

                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={warpData.slice(0, currentIndex + 10)}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="#1e293b"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="cycle"
                        stroke="#64748b"
                        label={{
                          value: "Cycle",
                          position: "insideBottomRight",
                          offset: -5,
                        }}
                      />
                      <YAxis
                        label={{
                          value: "Warp ID",
                          angle: -90,
                          position: "insideLeft",
                        }}
                        stroke="#64748b"
                        domain={[0, 4]}
                        tickCount={5}
                      />
                      <RechartsTooltip
                        cursor={{ fill: "#1e293b", opacity: 0.4 }}
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload as any;
                            return (
                              <div className="bg-slate-900 border border-slate-700 p-2 rounded shadow-xl text-xs">
                                <p className="text-slate-300 mb-1">
                                  Cycle: {data.cycle}
                                </p>
                                <p className="font-bold text-emerald-400">
                                  Warp ID: {data.warpId}
                                </p>
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar
                        dataKey="warpId"
                        fill="#8B5CF6"
                        barSize={20}
                        radius={[2, 2, 0, 0] as any}
                      >
                        {warpData
                          .slice(0, currentIndex + 10)
                          .map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={
                                entry.cycle === currentRecord.cycle
                                  ? "#F59E0B"
                                  : "#8B5CF6"
                              }
                            />
                          ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Bottom control bar */}
      <footer className="bg-slate-900 border-t border-slate-800 p-4 shadow-[0_-5px_20px_rgba(0,0,0,0.3)] z-20">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <button
              className="p-2 rounded-full hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
              onClick={() => {
                setCurrentIndex(0);
                setIsPlaying(false);
              }}
            >
              <SkipBack size={18} />
            </button>
            <button
              className="p-3 rounded-full bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg transition-all active:scale-95"
              onClick={() => setIsPlaying(!isPlaying)}
            >
              {isPlaying ? (
                <Pause size={20} fill="currentColor" />
              ) : (
                <Play size={20} fill="currentColor" />
              )}
            </button>
            <button
              className="p-2 rounded-full hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
              onClick={() => {
                setCurrentIndex((prev) =>
                  Math.min(prev + 1, traceData.records.length - 1)
                );
                setIsPlaying(false);
              }}
            >
              <SkipForward size={18} />
            </button>
          </div>

          <div className="flex-1 flex flex-col space-y-1">
            <div className="flex justify-between text-xs text-slate-400 font-mono">
              <span>CYCLE {currentRecord.cycle}</span>
              <span>
                {Math.round(
                  ((currentIndex + 1) / traceData.records.length) * 100
                )}
                %
              </span>
            </div>
            <input
              type="range"
              min="0"
              max={traceData.records.length - 1}
              value={currentIndex}
              onChange={(e) => {
                setCurrentIndex(parseInt(e.target.value));
                setIsPlaying(false);
              }}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
            />
          </div>

          <div className="hidden md:flex items-center space-x-4 border-l border-slate-700 pl-6">
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-500 uppercase">
                SM ID
              </span>
              <span className="text-sm font-mono text-slate-300">
                #{currentRecord.hw_ctx.sm_id}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-500 uppercase">Warp</span>
              <span className="text-sm font-mono text-slate-300">
                #{currentRecord.hw_ctx.warp_id}
              </span>
            </div>
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-500 uppercase">Core</span>
              <span className="text-sm font-mono text-slate-300">
                #{currentRecord.perf.core_id}
              </span>
            </div>
          </div>
        </div>
      </footer>
      {activeTab === "timing" && (
        <div className="bg-slate-800 p-6 rounded-lg border border-slate-700 shadow-xl">
          <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center">
            <Clock size={20} className="mr-2 text-emerald-500" />
            Memory Access Timing (Cycle vs Address)
          </h3>
          <div className="h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  type="number"
                  dataKey="cycle"
                  name="Cycle"
                  stroke="#94a3b8"
                  label={{
                    value: "Cycle",
                    position: "insideBottomRight",
                    offset: -10,
                    fill: "#94a3b8",
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="address"
                  name="Address"
                  stroke="#94a3b8"
                  label={{
                    value: "Address",
                    angle: -90,
                    position: "insideLeft",
                    fill: "#94a3b8",
                  }}
                />
                <ZAxis
                  type="number"
                  dataKey="lane"
                  range={[50, 50]}
                  name="Lane"
                />
                <RechartsTooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{
                    backgroundColor: "#1e293b",
                    borderColor: "#334155",
                    color: "#f1f5f9",
                  }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-slate-900 border border-slate-700 p-3 rounded shadow-lg text-xs">
                          <p className="text-emerald-400 font-bold mb-1">
                            Cycle {data.cycle}
                          </p>
                          <p className="text-slate-300">
                            Addr:{" "}
                            <span className="text-white">{data.address}</span>{" "}
                            (0x{data.address.toString(16).toUpperCase()})
                          </p>
                          <p className="text-slate-300">
                            Val:{" "}
                            <span className="text-white">{data.value}</span>
                          </p>
                          <p className="text-slate-300">
                            Lane:{" "}
                            <span className="text-white">{data.lane}</span>
                          </p>
                          <p
                            className={`font-bold ${
                              data.type === "write"
                                ? "text-red-400"
                                : "text-blue-400"
                            }`}
                          >
                            {data.type.toUpperCase()}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend />
                <Scatter
                  name="Reads"
                  data={memoryAccessPoints.filter((p) => p.type === "read")}
                  fill="#60A5FA"
                  shape="circle"
                />
                <Scatter
                  name="Writes"
                  data={memoryAccessPoints.filter((p) => p.type === "write")}
                  fill="#EF4444"
                  shape="square"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {activeTab === "registers" && (
        <div className="bg-slate-800 p-6 rounded-lg border border-slate-700 shadow-xl">
          <h3 className="text-lg font-semibold text-slate-100 mb-4 flex items-center">
            <Activity size={20} className="mr-2 text-emerald-500" />
            Register Write Timing (Cycle vs Register)
          </h3>
          <div className="h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  type="number"
                  dataKey="cycle"
                  name="Cycle"
                  stroke="#94a3b8"
                  label={{
                    value: "Cycle",
                    position: "insideBottomRight",
                    offset: -10,
                    fill: "#94a3b8",
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="reg"
                  name="Reg Index"
                  stroke="#94a3b8"
                  domain={[0, 31]}
                  label={{
                    value: "Register Index (R0-R31)",
                    angle: -90,
                    position: "insideLeft",
                    fill: "#94a3b8",
                  }}
                />
                <RechartsTooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-slate-900 border border-slate-700 p-3 rounded shadow-lg text-xs">
                          <p className="text-emerald-400 font-bold mb-1">
                            Cycle {data.cycle}
                          </p>
                          <p className="text-white">Write to R{data.reg}</p>
                          <p className="text-slate-400 italic">{data.asm}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Scatter
                  name="Register Writes"
                  data={registerWritePoints}
                  fill="#F59E0B"
                  shape="diamond"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 p-4 bg-slate-900 rounded border border-slate-700">
            <p className="text-xs text-slate-400">
              * Displays registers explicitly targeted by instructions (e.g.,
              'MOV R15, ...'). Vertical patterns indicate multiple writes or
              parallel lane activity (if differentiated).
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
