import { useAuditDetail } from "@/api/queries";
import { DecisionBadge } from "@/components/DecisionBadge";
import { RedactedText } from "@/components/RedactedText";

export function AuditDetailDrawer({
  recordId,
  onClose,
}: {
  recordId: string;
  onClose: () => void;
}) {
  const { data, isLoading, isError, error } = useAuditDetail(recordId);

  return (
    <div className="fixed inset-0 z-20 flex justify-end">
      <button
        aria-label="Close detail drawer"
        onClick={onClose}
        className="absolute inset-0 bg-ink/30 dark:bg-black/50"
      />
      <div className="relative flex h-full w-full max-w-md flex-col overflow-y-auto bg-paper p-5 shadow-xl dark:bg-paper-dark">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-semibold tracking-wide text-ink uppercase dark:text-ink-dark">
            Audit record
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-sm text-ink-muted hover:text-ink dark:text-ink-muted-dark dark:hover:text-ink-dark"
          >
            Close
          </button>
        </div>

        {isLoading && (
          <p className="mt-4 text-sm text-ink-muted dark:text-ink-muted-dark">Loading…</p>
        )}
        {isError && (
          <p className="mt-4 font-mono text-sm text-block dark:text-block-dark">
            {error.message}
          </p>
        )}

        {data && (
          <div className="mt-4 space-y-4 text-sm">
            <div className="flex items-center gap-2">
              <DecisionBadge decision={data.decision} />
              <span className="font-mono text-ink-muted dark:text-ink-muted-dark">
                {new Date(data.ts).toLocaleString()}
              </span>
            </div>

            <dl className="grid grid-cols-2 gap-2 text-xs">
              <dt className="text-ink-muted dark:text-ink-muted-dark">Model</dt>
              <dd className="font-mono text-ink dark:text-ink-dark">{data.model}</dd>
              <dt className="text-ink-muted dark:text-ink-muted-dark">Input decision</dt>
              <dd className="font-mono text-ink dark:text-ink-dark">{data.input_decision}</dd>
              <dt className="text-ink-muted dark:text-ink-muted-dark">Output decision</dt>
              <dd className="font-mono text-ink dark:text-ink-dark">
                {data.output_decision ?? "—"}
              </dd>
              <dt className="text-ink-muted dark:text-ink-muted-dark">Policy</dt>
              <dd className="font-mono text-ink dark:text-ink-dark">{data.policy_id}</dd>
              <dt className="text-ink-muted dark:text-ink-muted-dark">Tenant</dt>
              <dd className="font-mono text-ink dark:text-ink-dark">
                {data.tenant_id ?? "—"}
              </dd>
            </dl>

            <div>
              <p className="text-xs font-medium tracking-wide text-ink uppercase dark:text-ink-dark">
                Latency breakdown
              </p>
              <dl className="mt-1 grid grid-cols-3 gap-2 text-xs">
                <div>
                  <dt className="text-ink-muted dark:text-ink-muted-dark">Detection</dt>
                  <dd className="font-mono text-ink dark:text-ink-dark">
                    {data.detection_latency_ms}ms
                  </dd>
                </div>
                <div>
                  <dt className="text-ink-muted dark:text-ink-muted-dark">LLM</dt>
                  <dd className="font-mono text-ink dark:text-ink-dark">
                    {data.llm_latency_ms ?? "—"}ms
                  </dd>
                </div>
                <div>
                  <dt className="text-ink-muted dark:text-ink-muted-dark">Total</dt>
                  <dd className="font-mono text-ink dark:text-ink-dark">
                    {data.total_latency_ms}ms
                  </dd>
                </div>
              </dl>
            </div>

            {data.reasons.length > 0 && (
              <div>
                <p className="text-xs font-medium tracking-wide text-ink uppercase dark:text-ink-dark">
                  Reasons
                </p>
                <ul className="mt-1 list-inside list-disc font-mono text-xs text-ink-muted dark:text-ink-muted-dark">
                  {data.reasons.map((r, i) => (
                    <li key={i}>{r}</li>
                  ))}
                </ul>
              </div>
            )}

            <div>
              <p className="text-xs font-medium tracking-wide text-ink uppercase dark:text-ink-dark">
                Redacted prompt
              </p>
              <p className="mt-1 border border-hairline-strong bg-ink/[0.03] p-2 font-mono text-xs whitespace-pre-wrap text-ink dark:border-hairline-strong-dark dark:bg-ink-dark/[0.06] dark:text-ink-dark">
                <RedactedText text={data.prompt_redacted} />
              </p>
            </div>

            {data.response_redacted && (
              <div>
                <p className="text-xs font-medium tracking-wide text-ink uppercase dark:text-ink-dark">
                  Redacted response
                </p>
                <p className="mt-1 border border-hairline-strong bg-ink/[0.03] p-2 font-mono text-xs whitespace-pre-wrap text-ink dark:border-hairline-strong-dark dark:bg-ink-dark/[0.06] dark:text-ink-dark">
                  <RedactedText text={data.response_redacted} />
                </p>
              </div>
            )}

            {Object.keys(data.input_redactions).length > 0 && (
              <div>
                <p className="text-xs font-medium tracking-wide text-ink uppercase dark:text-ink-dark">
                  Input redactions
                </p>
                <p className="mt-1 font-mono text-xs text-ink-muted dark:text-ink-muted-dark">
                  {Object.entries(data.input_redactions)
                    .map(([type, count]) => `${type} × ${count}`)
                    .join(", ")}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
