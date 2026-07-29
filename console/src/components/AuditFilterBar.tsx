import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";

import type { Decision } from "@/api/queries";

const TIME_PRESETS = [
  { label: "1h", ms: 60 * 60 * 1000 },
  { label: "24h", ms: 24 * 60 * 60 * 1000 },
  { label: "7d", ms: 7 * 24 * 60 * 60 * 1000 },
];

// The backend filters by a single `decision` value, not a set — see
// GET /api/v1/audit's `decision` query param. A multi-select control in
// the UI that the API can't actually honor, so this is a single-select.
const DECISIONS: Decision[] = ["allow", "mask", "block"];

/**
 * Local state mirrors the URL param and updates it debounced, so the input
 * feels instant. Writing straight to useSearchParams on every keystroke
 * (no local buffer) drops characters under fast typing — each set() call
 * triggers a router navigation/re-render, and that round-trip is slower
 * than keystroke rate, so the input's value (read back from the URL) goes
 * stale mid-type and characters get lost. Found this by actually typing
 * into the running app, not just eyeballing the code.
 */
function useDebouncedUrlParam(
  params: URLSearchParams,
  setParam: (key: string, value: string | null) => void,
  key: string,
  delayMs = 300,
) {
  const [local, setLocal] = useState(params.get(key) ?? "");

  useEffect(() => {
    setLocal(params.get(key) ?? "");
    // Only resync from the URL when the key itself changes externally
    // (e.g. "Clear filters") — not on every params object identity change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.get(key)]);

  useEffect(() => {
    const handle = setTimeout(() => {
      setParam(key, local || null);
    }, delayMs);
    return () => clearTimeout(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [local]);

  return [local, setLocal] as const;
}

export function AuditFilterBar() {
  const [params, setParams] = useSearchParams();

  function setParam(key: string, value: string | null) {
    const next = new URLSearchParams(params);
    if (value) {
      next.set(key, value);
    } else {
      next.delete(key);
    }
    setParams(next, { replace: true });
  }

  const [ruleId, setRuleId] = useDebouncedUrlParam(params, setParam, "rule_id");
  const [q, setQ] = useDebouncedUrlParam(params, setParam, "q");

  function applyPreset(ms: number) {
    const from = new Date(Date.now() - ms).toISOString();
    setParam("from", from);
  }

  function clearAll() {
    setParams(new URLSearchParams(), { replace: true });
    setRuleId("");
    setQ("");
  }

  return (
    <div className="flex flex-wrap items-center gap-3 border border-hairline-strong p-3 dark:border-hairline-strong-dark">
      <div className="flex border border-hairline-strong dark:border-hairline-strong-dark">
        {TIME_PRESETS.map((preset, i) => (
          <button
            key={preset.label}
            type="button"
            onClick={() => applyPreset(preset.ms)}
            className={`px-2 py-1 font-mono text-xs font-medium text-ink-muted hover:bg-ink/[0.05] dark:text-ink-muted-dark dark:hover:bg-ink-dark/[0.08] ${
              i > 0 ? "border-l border-hairline-strong dark:border-hairline-strong-dark" : ""
            }`}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <select
        value={params.get("decision") ?? ""}
        onChange={(e) => setParam("decision", e.target.value || null)}
        className="border border-hairline-strong bg-paper px-2 py-1 font-mono text-sm text-ink uppercase focus:border-ink focus:outline-none dark:border-hairline-strong-dark dark:bg-paper-dark dark:text-ink-dark dark:focus:border-ink-dark"
      >
        <option value="">All decisions</option>
        {DECISIONS.map((d) => (
          <option key={d} value={d}>
            {d}
          </option>
        ))}
      </select>

      <input
        type="text"
        value={ruleId}
        onChange={(e) => setRuleId(e.target.value)}
        placeholder="Entity type (e.g. AWS_KEY)"
        className="border border-hairline-strong bg-paper px-2 py-1 font-mono text-sm text-ink placeholder:text-ink-muted focus:border-ink focus:outline-none dark:border-hairline-strong-dark dark:bg-paper-dark dark:text-ink-dark dark:placeholder:text-ink-muted-dark dark:focus:border-ink-dark"
      />

      <input
        type="text"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Search redacted content…"
        className="min-w-48 flex-1 border border-hairline-strong bg-paper px-2 py-1 text-sm text-ink placeholder:text-ink-muted focus:border-ink focus:outline-none dark:border-hairline-strong-dark dark:bg-paper-dark dark:text-ink-dark dark:placeholder:text-ink-muted-dark dark:focus:border-ink-dark"
      />

      {(params.get("decision") || params.get("rule_id") || params.get("q") || params.get("from")) && (
        <button
          type="button"
          onClick={clearAll}
          className="text-xs font-medium text-ink-muted hover:text-ink dark:text-ink-muted-dark dark:hover:text-ink-dark"
        >
          Clear filters
        </button>
      )}
    </div>
  );
}
