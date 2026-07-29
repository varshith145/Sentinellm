import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TimeseriesPoint } from "@/api/queries";

const tickStyle = { fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--color-ink-muted)" };

/**
 * p95 only, not p50/p95/p99: GET /api/v1/stats/timeseries's TimeseriesPoint
 * only carries p95_ms per bucket (see console_models.py) — the backend
 * doesn't compute p50/p99 per-bucket, only as a single aggregate over the
 * whole window (StatsSummary, shown in the metric cards above this chart).
 * A 3-line chart here would be UI describing data the API doesn't have.
 */
export function LatencyChart({ points }: { points: TimeseriesPoint[] }) {
  const data = points.map((p) => ({
    ...p,
    bucket: new Date(p.bucket).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data}>
        <CartesianGrid stroke="var(--color-hairline)" strokeWidth={1} />
        <XAxis dataKey="bucket" tick={tickStyle} tickMargin={8} stroke="var(--color-hairline-strong)" />
        <YAxis unit="ms" tick={tickStyle} stroke="var(--color-hairline-strong)" />
        <Tooltip
          contentStyle={{
            background: "var(--color-paper)",
            border: "1px solid var(--color-ink)",
            borderRadius: 0,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
          }}
          labelStyle={{ color: "var(--color-ink)" }}
        />
        {/* Graphite, not a severity ink — latency itself isn't a BLOCK
            event; vermillion stays reserved for actual blocked decisions
            (see DESIGN.md's Two-Ink Rule). */}
        <Line
          type="linear"
          dataKey="p95_ms"
          name="p95"
          stroke="var(--color-ink)"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
