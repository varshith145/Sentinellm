import { useState } from "react";

import { useDeletePolicy, useUpdatePolicy, type PolicyRule } from "@/api/queries";
import { DecisionBadge } from "@/components/DecisionBadge";
import { EmptyState } from "@/components/EmptyState";

export function PolicyTable({
  policies,
  onSelect,
}: {
  policies: PolicyRule[];
  onSelect: (policy: PolicyRule) => void;
}) {
  const updatePolicy = useUpdatePolicy();
  const deletePolicy = useDeletePolicy();
  const [confirmingId, setConfirmingId] = useState<string | null>(null);

  if (policies.length === 0) {
    return (
      <EmptyState
        title="No policies yet"
        description="Create one below to start authoring rules."
      />
    );
  }

  return (
    <div className="overflow-x-auto border border-hairline-strong dark:border-hairline-strong-dark">
      <table className="min-w-full">
        <thead>
          <tr className="border-b-2 border-ink dark:border-ink-dark">
            {["Name", "Entity type", "Action", "Min confidence", "Enabled", "Updated", ""].map(
              (h) => (
                <th
                  key={h}
                  className="px-4 py-2 text-left text-xs font-medium tracking-wide text-ink-muted uppercase dark:text-ink-muted-dark"
                >
                  {h}
                </th>
              ),
            )}
          </tr>
        </thead>
        <tbody className="divide-y divide-hairline dark:divide-hairline-dark">
          {policies.map((policy) => (
            <tr
              key={policy.id}
              className="cursor-pointer hover:bg-ink/[0.03] dark:hover:bg-ink-dark/[0.05]"
              onClick={() => onSelect(policy)}
            >
              <td className="px-4 py-2 text-sm font-medium text-ink dark:text-ink-dark">
                {policy.name}
              </td>
              <td className="px-4 py-2 font-mono text-sm text-ink-muted dark:text-ink-muted-dark">
                {policy.entity_type}
              </td>
              <td className="px-4 py-2">
                <DecisionBadge decision={policy.action} />
              </td>
              <td className="px-4 py-2 font-mono text-sm text-ink-muted dark:text-ink-muted-dark">
                {policy.min_confidence.toFixed(2)}
              </td>
              <td className="px-4 py-2" onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={policy.enabled}
                  onChange={(e) =>
                    updatePolicy.mutate({ id: policy.id, body: { enabled: e.target.checked } })
                  }
                  className="h-4 w-4 accent-ink dark:accent-ink-dark"
                />
              </td>
              <td className="px-4 py-2 font-mono text-sm text-ink-muted dark:text-ink-muted-dark">
                {new Date(policy.updated_at).toLocaleString()}
              </td>
              <td className="px-4 py-2 text-right" onClick={(e) => e.stopPropagation()}>
                {confirmingId === policy.id ? (
                  <span className="flex items-center justify-end gap-2">
                    <span className="text-xs text-ink-muted dark:text-ink-muted-dark">
                      Delete?
                    </span>
                    <button
                      type="button"
                      onClick={() => {
                        deletePolicy.mutate(policy.id);
                        setConfirmingId(null);
                      }}
                      className="text-xs font-medium text-block hover:underline dark:text-block-dark"
                    >
                      Confirm
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfirmingId(null)}
                      className="text-xs text-ink-muted hover:underline dark:text-ink-muted-dark"
                    >
                      Cancel
                    </button>
                  </span>
                ) : (
                  <button
                    type="button"
                    onClick={() => setConfirmingId(policy.id)}
                    className="text-xs font-medium text-ink-muted hover:text-block dark:text-ink-muted-dark dark:hover:text-block-dark"
                  >
                    Delete
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
