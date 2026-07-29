import type { AuditRecord } from "@/api/queries";
import { DecisionBadge } from "@/components/DecisionBadge";
import { EmptyState } from "@/components/EmptyState";
import { RedactedText } from "@/components/RedactedText";

export function AuditTable({
  records,
  onSelect,
}: {
  records: AuditRecord[];
  onSelect: (record: AuditRecord) => void;
}) {
  if (records.length === 0) {
    return (
      <EmptyState
        title="No audit records match these filters"
        description="Try widening the time range or clearing a filter."
      />
    );
  }

  return (
    <div className="overflow-x-auto border border-hairline-strong dark:border-hairline-strong-dark">
      <table className="min-w-full">
        <thead>
          <tr className="border-b-2 border-ink dark:border-ink-dark">
            {["Time", "Decision", "Entity types", "Preview", "Latency"].map((h) => (
              <th
                key={h}
                className="px-4 py-2 text-left text-xs font-medium tracking-wide text-ink-muted uppercase dark:text-ink-muted-dark"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-hairline dark:divide-hairline-dark">
          {records.map((record) => (
            <tr
              key={record.id}
              className="cursor-pointer hover:bg-ink/[0.03] dark:hover:bg-ink-dark/[0.05]"
              onClick={() => onSelect(record)}
            >
              <td className="font-mono text-sm whitespace-nowrap text-ink-muted px-4 py-2 dark:text-ink-muted-dark">
                {new Date(record.ts).toLocaleString()}
              </td>
              <td className="px-4 py-2">
                <DecisionBadge decision={record.decision} />
              </td>
              <td className="px-4 py-2 font-mono text-sm text-ink-muted dark:text-ink-muted-dark">
                {record.entity_types.join(", ") || "—"}
              </td>
              <td className="max-w-xs truncate px-4 py-2 font-mono text-sm text-ink dark:text-ink-dark">
                <RedactedText text={record.request_preview} />
              </td>
              <td className="px-4 py-2 font-mono text-sm whitespace-nowrap text-ink-muted dark:text-ink-muted-dark">
                {record.latency_ms.toFixed(0)}ms
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
