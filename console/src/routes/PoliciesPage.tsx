import { useState } from "react";

import { usePolicies, type PolicyRule } from "@/api/queries";
import { DryRunPanel } from "@/components/DryRunPanel";
import { ErrorState } from "@/components/ErrorState";
import { PolicyForm } from "@/components/PolicyForm";
import { PolicyTable } from "@/components/PolicyTable";

export function PoliciesPage() {
  const { data: policies, isLoading, isError, error } = usePolicies();
  const [selected, setSelected] = useState<PolicyRule | null>(null);

  return (
    <div className="space-y-6">
      <h1 className="border-b-2 border-ink pb-2 text-sm font-semibold tracking-[0.15em] text-ink uppercase dark:border-ink-dark dark:text-ink-dark">
        Policy
      </h1>

      {isError && <ErrorState message={error.message} />}
      {isLoading && (
        <p className="font-mono text-sm text-ink-muted dark:text-ink-muted-dark">
          Loading policies…
        </p>
      )}
      {policies && <PolicyTable policies={policies} onSelect={setSelected} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <PolicyForm />
        {selected ? (
          <DryRunPanel policy={selected} />
        ) : (
          <div className="flex items-center justify-center border border-dashed border-hairline-strong p-4 text-sm text-ink-muted dark:border-hairline-strong-dark dark:text-ink-muted-dark">
            Select a policy row to dry-run it
          </div>
        )}
      </div>
    </div>
  );
}
