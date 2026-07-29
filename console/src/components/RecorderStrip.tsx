import { useEffect, useRef, useState } from "react";

import type { Decision } from "@/api/queries";

const STROKE: Record<Decision, string> = {
  allow: "var(--color-ink)",
  mask: "var(--color-mask)",
  block: "var(--color-block)",
};

// Fixed 400x60 viewBox. Baseline runs flat, then deflects once around
// x=180-230 — a calm trace for ALLOW, a moderate bump for MASK, a sharp
// spike for BLOCK, exactly like a seismograph needle registering an event.
function buildPath(decision: Decision): string {
  switch (decision) {
    case "allow":
      return "M0,30 L70,30 L90,27 L110,31 L130,30 L400,30";
    case "mask":
      return "M0,30 L170,30 L195,13 L210,30 L400,30";
    case "block":
      return "M0,30 L170,30 L188,4 L198,30 L210,52 L222,30 L400,30";
  }
}

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export function RecorderStrip({
  status,
  decision,
}: {
  status: "idle" | "pending" | "done";
  decision?: Decision;
}) {
  const pathRef = useRef<SVGPathElement>(null);
  const [dashLength, setDashLength] = useState(0);
  const d = decision ? buildPath(decision) : "M0,30 L400,30";

  useEffect(() => {
    if (pathRef.current) {
      setDashLength(pathRef.current.getTotalLength());
    }
  }, [d]);

  const animate = status === "done" && decision !== undefined && !prefersReducedMotion();

  return (
    <div className="relative h-16 overflow-hidden border-y border-hairline-strong bg-ink/[0.02] dark:border-hairline-strong-dark dark:bg-ink-dark/[0.04]">
      <svg viewBox="0 0 400 60" preserveAspectRatio="none" className="h-full w-full">
        {Array.from({ length: 20 }, (_, i) => (
          <line
            key={i}
            x1={i * 20}
            x2={i * 20}
            y1={0}
            y2={60}
            className="stroke-ink/10 dark:stroke-ink-dark/10"
            strokeWidth={1}
          />
        ))}
        <path
          ref={pathRef}
          d={d}
          fill="none"
          stroke={decision ? STROKE[decision] : "var(--color-ink-muted)"}
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          style={
            animate && dashLength
              ? {
                  strokeDasharray: dashLength,
                  strokeDashoffset: dashLength,
                  animation: "trace-draw 0.9s ease-out forwards",
                }
              : status === "pending"
                ? { opacity: 0.25 }
                : undefined
          }
        />
      </svg>
      {status === "pending" && (
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="recording-pulse font-mono text-xs text-ink-muted dark:text-ink-muted-dark">
            recording…
          </span>
        </div>
      )}
    </div>
  );
}
