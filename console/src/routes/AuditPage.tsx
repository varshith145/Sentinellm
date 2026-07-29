import { useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useAuditList, type AuditRecord, type Decision } from "@/api/queries";
import { AuditDetailDrawer } from "@/components/AuditDetailDrawer";
import { AuditFilterBar } from "@/components/AuditFilterBar";
import { AuditTable } from "@/components/AuditTable";
import { ErrorState } from "@/components/ErrorState";

export function AuditPage() {
  const [params] = useSearchParams();
  const [selected, setSelected] = useState<AuditRecord | null>(null);

  const filters = {
    from: params.get("from") ?? undefined,
    decision: (params.get("decision") as Decision | null) ?? undefined,
    rule_id: params.get("rule_id") ?? undefined,
    q: params.get("q") ?? undefined,
  };

  const {
    data,
    isLoading,
    isError,
    error,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useAuditList(filters);

  const records = data?.pages.flatMap((page) => page.items) ?? [];

  return (
    <div className="space-y-4">
      <h1 className="border-b-2 border-ink pb-2 text-sm font-semibold tracking-[0.15em] text-ink uppercase dark:border-ink-dark dark:text-ink-dark">
        Audit
      </h1>

      <AuditFilterBar />

      {isError && <ErrorState message={error.message} />}
      {isLoading && (
        <p className="font-mono text-sm text-ink-muted dark:text-ink-muted-dark">
          Loading audit records…
        </p>
      )}
      {data && <AuditTable records={records} onSelect={setSelected} />}

      {hasNextPage && (
        <div className="flex justify-center">
          <button
            type="button"
            onClick={() => fetchNextPage()}
            disabled={isFetchingNextPage}
            className="border border-hairline-strong px-4 py-1.5 text-sm font-medium text-ink hover:bg-ink/[0.05] disabled:opacity-50 dark:border-hairline-strong-dark dark:text-ink-dark dark:hover:bg-ink-dark/[0.08]"
          >
            {isFetchingNextPage ? "Loading…" : "Load more"}
          </button>
        </div>
      )}

      {selected && (
        <AuditDetailDrawer recordId={selected.id} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}
