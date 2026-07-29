import { useState } from "react";

import { useStatsSummary, useStatsTimeseries } from "@/api/queries";
import { DecisionTimeseriesChart } from "@/components/DecisionTimeseriesChart";
import { ErrorState } from "@/components/ErrorState";
import { LatencyChart } from "@/components/LatencyChart";
import { MetricCard } from "@/components/MetricCard";

const WINDOWS = [
  { label: "1h", window: "1h", bucket: "5m" },
  { label: "24h", window: "24h", bucket: "1h" },
  { label: "7d", window: "7d", bucket: "12h" },
];

export function MetricsPage() {
  const [selectedWindow, setSelectedWindow] = useState(WINDOWS[1]!);

  const summary = useStatsSummary(selectedWindow.window);
  const timeseries = useStatsTimeseries(selectedWindow.window, selectedWindow.bucket);

  const blockRate =
    summary.data && summary.data.total > 0
      ? (((summary.data.by_decision.block ?? 0) / summary.data.total) * 100).toFixed(1) +
        "%"
      : "—";

  const topRule = summary.data?.top_rules[0];
  const topRuleType: string | undefined = topRule?.[0];
  const topRuleCount: number | undefined = topRule?.[1];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between border-b-2 border-ink pb-2 dark:border-ink-dark">
        <h1 className="text-sm font-semibold tracking-[0.15em] text-ink uppercase dark:text-ink-dark">
          Metrics
        </h1>
        <div className="flex border border-hairline-strong dark:border-hairline-strong-dark">
          {WINDOWS.map((w, i) => (
            <button
              key={w.label}
              type="button"
              onClick={() => setSelectedWindow(w)}
              className={`px-3 py-1 font-mono text-xs font-medium ${
                i > 0 ? "border-l border-hairline-strong dark:border-hairline-strong-dark" : ""
              } ${
                w.label === selectedWindow.label
                  ? "bg-ink text-paper dark:bg-ink-dark dark:text-paper-dark"
                  : "text-ink-muted hover:bg-ink/[0.05] dark:text-ink-muted-dark dark:hover:bg-ink-dark/[0.08]"
              }`}
            >
              {w.label}
            </button>
          ))}
        </div>
      </div>

      {summary.isError && <ErrorState message={summary.error.message} />}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard label="Total requests" value={summary.data ? String(summary.data.total) : "—"} />
        <MetricCard label="Block rate" value={blockRate} />
        <MetricCard
          label="p95 latency"
          value={summary.data ? `${summary.data.p95_ms.toFixed(0)}ms` : "—"}
        />
        <MetricCard
          label="Top entity type"
          value={topRuleType ?? "—"}
          sublabel={topRuleCount !== undefined ? `${topRuleCount} triggers` : undefined}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="border border-hairline-strong p-4 dark:border-hairline-strong-dark">
          <h2 className="mb-2 text-xs font-medium tracking-wide text-ink-muted uppercase dark:text-ink-muted-dark">
            Decisions over time
          </h2>
          {timeseries.data && <DecisionTimeseriesChart points={timeseries.data} />}
        </div>
        <div className="border border-hairline-strong p-4 dark:border-hairline-strong-dark">
          <h2 className="mb-2 text-xs font-medium tracking-wide text-ink-muted uppercase dark:text-ink-muted-dark">
            p95 latency over time
          </h2>
          {timeseries.data && <LatencyChart points={timeseries.data} />}
        </div>
      </div>
    </div>
  );
}
