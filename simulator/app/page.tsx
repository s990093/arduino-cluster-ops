"use client";

import React, { useState } from "react";
import { TraceData } from "@/types/trace";
import TraceAnalyzer from "@/components/TraceAnalyzer";
import JSONInput from "@/components/JSONInput";

export default function Home() {
  const [traceData, setTraceData] = useState<TraceData | null>(null);
  const [showInput, setShowInput] = useState(true);

  const handleTraceLoad = (data: TraceData) => {
    setTraceData(data);
    setShowInput(false);
  };

  const handleReset = () => {
    setTraceData(null);
    setShowInput(true);
  };

  return (
    <div className="h-screen w-screen overflow-hidden">
      {showInput || !traceData ? (
        <JSONInput onTraceLoad={handleTraceLoad} />
      ) : (
        <TraceAnalyzer traceData={traceData} onReset={handleReset} />
      )}
    </div>
  );
}
