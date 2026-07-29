import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { TimeseriesPoint } from "@/api/queries";

// Severity inks, plus graphite for the calm ALLOW trace — the same three
// colors used everywhere else in the system (see DESIGN.md's Two-Ink Rule;
// ALLOW itself isn't a "third ink," it's the resting/ink-black state).
const COLORS = {
  allow: "var(--color-ink)",
  mask: "var(--color-mask)",
  block: "var(--color-block)",
};

const tickStyle = { fontFamily: "var(--font-mono)", fontSize: 11, fill: "var(--color-ink-muted)" };

export function DecisionTimeseriesChart({ points }: { points: TimeseriesPoint[] }) {
  const data = points.map((p) => ({
    ...p,
    bucket: new Date(p.bucket).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data}>
        {/* Fine ruled grid on both axes — the calibration-grid device used
            in RecorderStrip, not a default Cartesian grid. */}
        <CartesianGrid stroke="var(--color-hairline)" strokeWidth={1} />
        <XAxis dataKey="bucket" tick={tickStyle} tickMargin={8} stroke="var(--color-hairline-strong)" />
        <YAxis allowDecimals={false} tick={tickStyle} stroke="var(--color-hairline-strong)" />
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
        <Legend
          wrapperStyle={{
            fontFamily: "var(--font-sans)",
            fontSize: 12,
            color: "var(--color-ink-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        />
        <Area
          type="linear"
          dataKey="allow"
          stackId="1"
          stroke={COLORS.allow}
          fill={COLORS.allow}
          fillOpacity={0.12}
        />
        <Area
          type="linear"
          dataKey="mask"
          stackId="1"
          stroke={COLORS.mask}
          fill={COLORS.mask}
          fillOpacity={0.35}
        />
        <Area
          type="linear"
          dataKey="block"
          stackId="1"
          stroke={COLORS.block}
          fill={COLORS.block}
          fillOpacity={0.45}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
