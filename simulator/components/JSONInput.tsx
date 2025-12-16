"use client";

import React, { useState, useRef, useEffect } from "react";
import { Upload, FileJson, AlertCircle, History } from "lucide-react";
import { TraceData } from "@/types/trace";

interface JSONInputProps {
  onTraceLoad: (data: TraceData) => void;
}

const STORAGE_KEY = "gpu_trace_last_json";
const STORAGE_FILENAME_KEY = "gpu_trace_last_filename";

export default function JSONInput({ onTraceLoad }: JSONInputProps) {
  const [jsonText, setJsonText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [lastFileName, setLastFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load saved JSON from localStorage on mount
  useEffect(() => {
    try {
      const savedJSON = localStorage.getItem(STORAGE_KEY);
      const savedFileName = localStorage.getItem(STORAGE_FILENAME_KEY);

      if (savedJSON) {
        setJsonText(savedJSON);
      }

      if (savedFileName) {
        setLastFileName(savedFileName);
      }
    } catch (err) {
      console.error("Failed to load saved trace:", err);
    }
  }, []);

  const validateAndLoad = (text: string, fileName?: string) => {
    try {
      const data = JSON.parse(text) as TraceData;

      // Basic validation
      if (!data.records || !Array.isArray(data.records)) {
        throw new Error(
          "Invalid trace data: missing or invalid 'records' array"
        );
      }

      if (
        !data.total_instructions ||
        typeof data.total_instructions !== "number"
      ) {
        throw new Error("Invalid trace data: missing 'total_instructions'");
      }

      // Save to localStorage
      try {
        localStorage.setItem(STORAGE_KEY, text);
        if (fileName) {
          localStorage.setItem(STORAGE_FILENAME_KEY, fileName);
          setLastFileName(fileName);
        }
      } catch (err) {
        console.error("Failed to save to localStorage:", err);
      }

      setError(null);
      onTraceLoad(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid JSON format");
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      setJsonText(text);
      validateAndLoad(text, file.name);
    };
    reader.readAsText(file);
  };

  const handleLoadLastFile = () => {
    if (jsonText) {
      validateAndLoad(jsonText);
    }
  };

  const handleTextSubmit = () => {
    if (!jsonText.trim()) {
      setError("Please enter JSON data");
      return;
    }
    validateAndLoad(jsonText);
  };

  const loadSampleData = () => {
    // Sample trace data for demonstration
    const sampleData: TraceData = {
      trace_version: "2.1",
      program: "Sample Transformer",
      total_instructions: 3,
      records: [
        {
          cycle: 1,
          pc: 1,
          instruction: "0x10000002",
          asm: "0x10 dest=0 src1=0 src2=2",
          exec_time_us: 42,
          hw_ctx: { sm_id: 0, warp_id: 0, lane_id: 0, active_mask: "0xFF" },
          perf: {
            latency: 1,
            stall_cycles: 0,
            stall_reason: "NONE",
            pipe_stage: "WRITEBACK",
            core_id: 0,
            simd_width: 8,
          },
          lanes: Array(8)
            .fill(null)
            .map((_, i) => ({
              lane_id: i,
              R: [
                2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,
              ],
            })),
        },
        {
          cycle: 2,
          pc: 2,
          instruction: "0x10010003",
          asm: "0x10 dest=1 src1=0 src2=3",
          exec_time_us: 26,
          hw_ctx: { sm_id: 0, warp_id: 0, lane_id: 0, active_mask: "0xFF" },
          perf: {
            latency: 1,
            stall_cycles: 0,
            stall_reason: "NONE",
            pipe_stage: "WRITEBACK",
            core_id: 0,
            simd_width: 8,
          },
          memory_access: [
            { lane: 0, type: "read", addr: 4, val: 42 },
            { lane: 1, type: "read", addr: 8, val: 100 },
          ],
          lanes: Array(8)
            .fill(null)
            .map((_, i) => ({
              lane_id: i,
              R: [
                2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,
              ],
            })),
        },
        {
          cycle: 3,
          pc: 3,
          instruction: "0x13010001",
          asm: "0x13 dest=1 src1=0 src2=1",
          exec_time_us: 27,
          hw_ctx: { sm_id: 0, warp_id: 0, lane_id: 0, active_mask: "0xFF" },
          perf: {
            latency: 1,
            stall_cycles: 0,
            stall_reason: "NONE",
            pipe_stage: "WRITEBACK",
            core_id: 0,
            simd_width: 8,
          },
          memory_access: [
            { lane: 0, type: "write", addr: 4, val: 123 },
            { lane: 1, type: "write", addr: 8, val: 200 },
          ],
          lanes: Array(8)
            .fill(null)
            .map((_, i) => ({
              lane_id: i,
              R: [
                2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0,
              ],
            })),
        },
      ],
    };

    const jsonStr = JSON.stringify(sampleData, null, 2);
    setJsonText(jsonStr);
    setError(null);
  };

  return (
    <div className="h-full w-full bg-slate-950 flex items-center justify-center p-8">
      <div className="max-w-4xl w-full space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center space-x-3">
            <FileJson className="text-emerald-500" size={32} />
            <h1 className="text-3xl font-bold text-white">
              GPU Trace Architect
            </h1>
          </div>
          <p className="text-slate-400">Load GPU trace data in JSON format</p>
        </div>

        {/* File Upload */}
        <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
          <h3 className="text-sm font-semibold text-slate-300 mb-4 flex items-center">
            <Upload size={16} className="mr-2" />
            Upload JSON File
          </h3>

          {/* Show last uploaded file if exists */}
          {lastFileName && (
            <div className="mb-4 p-3 bg-slate-800/50 rounded border border-slate-700 flex items-center justify-between">
              <div className="flex items-center space-x-2 text-sm">
                <History size={16} className="text-emerald-500" />
                <span className="text-slate-400">Last file:</span>
                <span className="text-emerald-400 font-mono">
                  {lastFileName}
                </span>
              </div>
              <button
                onClick={handleLoadLastFile}
                className="px-3 py-1 text-xs bg-emerald-900/30 text-emerald-400 rounded border border-emerald-900/50 hover:bg-emerald-900/50 transition-colors font-semibold"
              >
                Load Again
              </button>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full px-4 py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 transition-colors flex items-center justify-center space-x-2"
          >
            <Upload size={18} />
            <span>Choose JSON File</span>
          </button>
        </div>

        {/* Manual JSON Input */}
        <div className="bg-slate-900 rounded-lg border border-slate-800 p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-sm font-semibold text-slate-300 flex items-center">
              <FileJson size={16} className="mr-2" />
              Paste JSON Data
            </h3>
            <button
              onClick={loadSampleData}
              className="text-xs px-3 py-1 bg-blue-900/30 text-blue-400 rounded border border-blue-900/50 hover:bg-blue-900/50 transition-colors"
            >
              Load Sample Data
            </button>
          </div>
          <textarea
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            placeholder='Paste your JSON trace data here...\n\nExample:\n{\n  "trace_version": "2.1",\n  "program": "My Program",\n  "total_instructions": 10,\n  "records": [...]\n}'
            className="w-full h-64 bg-slate-950 text-slate-200 font-mono text-sm p-4 rounded border border-slate-800 focus:border-emerald-500 focus:outline-none resize-none"
          />
          <button
            onClick={handleTextSubmit}
            className="mt-4 w-full px-4 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg transition-colors"
          >
            Load Trace Data
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-900/20 border border-red-900/50 rounded-lg p-4 flex items-start space-x-3">
            <AlertCircle
              className="text-red-400 flex-shrink-0 mt-0.5"
              size={18}
            />
            <div>
              <h4 className="text-red-400 font-semibold text-sm">Error</h4>
              <p className="text-red-300 text-sm mt-1">{error}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
