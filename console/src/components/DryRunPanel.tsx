import { useState } from "react";

import { useDryRun, type PolicyRule } from "@/api/queries";
import { DecisionBadge } from "@/components/DecisionBadge";
import { RecorderStrip } from "@/components/RecorderStrip";
import { RedactedText } from "@/components/RedactedText";

export function DryRunPanel({ policy }: { policy: PolicyRule }) {
  const [text, setText] = useState("");
  const dryRun = useDryRun();

  function handleTest() {
    if (!text.trim()) return;
    dryRun.mutate({ id: policy.id, text });
  }

  const status = dryRun.isPending ? "pending" : dryRun.isSuccess ? "done" : "idle";

  return (
    <div className="space-y-3 border border-hairline-strong p-4 dark:border-hairline-strong-dark">
      <h3 className="text-xs font-semibold tracking-wide text-ink uppercase dark:text-ink-dark">
        Dry-run: {policy.name}
      </h3>
      <p className="text-xs text-ink-muted dark:text-ink-muted-dark">
        Tests sample text against the live rules with this rule&apos;s current settings
        applied, regardless of whether it&apos;s enabled. No audit record is written.
      </p>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        placeholder="Paste sample text to test…"
        className="block w-full border border-hairline-strong bg-paper px-3 py-1.5 text-sm text-ink focus:border-ink focus:outline-none dark:border-hairline-strong-dark dark:bg-paper-dark dark:text-ink-dark dark:focus:border-ink-dark"
      />

      <button
        type="button"
        onClick={handleTest}
        disabled={dryRun.isPending || !text.trim()}
        className="border-2 border-ink bg-ink px-4 py-1.5 text-sm font-medium text-paper transition-colors hover:bg-transparent hover:text-ink disabled:opacity-50 dark:border-ink-dark dark:bg-ink-dark dark:text-paper-dark dark:hover:bg-transparent dark:hover:text-ink-dark"
      >
        {dryRun.isPending ? "Recording…" : "Test"}
      </button>

      {(status === "pending" || status === "done") && (
        <RecorderStrip
          key={dryRun.submittedAt}
          status={status}
          decision={dryRun.data?.decision}
        />
      )}

      {dryRun.isError && (
        <p className="font-mono text-sm text-block dark:text-block-dark">
          {dryRun.error.message}
        </p>
      )}

      {dryRun.data && (
        <div className="space-y-2 border border-hairline-strong bg-ink/[0.02] p-3 dark:border-hairline-strong-dark dark:bg-ink-dark/[0.04]">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-ink dark:text-ink-dark">Decision:</span>
            <DecisionBadge
              key={dryRun.submittedAt}
              decision={dryRun.data.decision}
              animateIn
            />
            <span className="font-mono text-xs text-ink-muted dark:text-ink-muted-dark">
              {dryRun.data.latency_ms.toFixed(1)}ms
            </span>
          </div>

          {dryRun.data.reasons.length > 0 && (
            <ul className="list-inside list-disc font-mono text-xs text-ink-muted dark:text-ink-muted-dark">
              {dryRun.data.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}

          <div>
            <p className="text-xs font-medium tracking-wide text-ink-muted uppercase dark:text-ink-muted-dark">
              Redacted text
            </p>
            <p className="mt-0.5 border border-hairline-strong bg-paper px-2 py-1 font-mono text-xs text-ink dark:border-hairline-strong-dark dark:bg-paper-dark dark:text-ink-dark">
              <RedactedText text={dryRun.data.redacted_text} />
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
