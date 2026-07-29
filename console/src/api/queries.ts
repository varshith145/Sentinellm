import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { api, unwrap } from "./client";
import type { components } from "./types";

export type Decision = components["schemas"]["Decision"];
export type EntityType = components["schemas"]["EntityType"];
export type PolicyRule = components["schemas"]["PolicyRule"];
export type PolicyCreate = components["schemas"]["PolicyCreate"];
export type PolicyUpdate = components["schemas"]["PolicyUpdate"];
export type AuditRecord = components["schemas"]["AuditRecord"];
export type AuditDetail = components["schemas"]["AuditDetail"];
export type StatsSummary = components["schemas"]["StatsSummary"];
export type TimeseriesPoint = components["schemas"]["TimeseriesPoint"];
export type DryRunResponse = components["schemas"]["DryRunResponse"];

export const ENTITY_TYPES: EntityType[] = [
  "EMAIL",
  "PHONE",
  "SSN",
  "CREDIT_CARD",
  "AWS_KEY",
  "GITHUB_TOKEN",
  "JWT",
  "PASSWORD",
  "PERSON_NAME",
  "GENERIC_PII",
  "GENERIC_SECRET",
];

// --- Policies ---

export function usePolicies() {
  return useQuery({
    queryKey: ["policies"],
    queryFn: async () =>
      unwrap(await api.GET("/api/v1/policies", { params: { query: { limit: 200 } } }))
        .items,
  });
}

export function useCreatePolicy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: PolicyCreate) =>
      unwrap(await api.POST("/api/v1/policies", { body })),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["policies"] });
    },
  });
}

export function useUpdatePolicy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, body }: { id: string; body: PolicyUpdate }) =>
      unwrap(
        await api.PATCH("/api/v1/policies/{policy_id}", {
          params: { path: { policy_id: id } },
          body,
        }),
      ),
    // Optimistic update: flipping the enabled toggle should feel instant,
    // with a rollback if the write fails (e.g. the public read-only demo).
    onMutate: async ({ id, body }) => {
      if (body.enabled === undefined) return;
      await queryClient.cancelQueries({ queryKey: ["policies"] });
      const previous = queryClient.getQueryData<PolicyRule[]>(["policies"]);
      queryClient.setQueryData<PolicyRule[]>(["policies"], (old) =>
        old?.map((p) => (p.id === id ? { ...p, enabled: body.enabled! } : p)),
      );
      return { previous };
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(["policies"], context.previous);
      }
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["policies"] });
    },
  });
}

export function useDeletePolicy() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => {
      const res = await api.DELETE("/api/v1/policies/{policy_id}", {
        params: { path: { policy_id: id } },
      });
      if (res.error) {
        const detail =
          typeof res.error === "object" && res.error !== null && "detail" in res.error
            ? String((res.error as { detail: unknown }).detail)
            : "Delete failed";
        throw new Error(detail);
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["policies"] });
    },
  });
}

export function useDryRun() {
  return useMutation({
    mutationFn: async ({ id, text }: { id: string; text: string }) =>
      unwrap(
        await api.POST("/api/v1/policies/{policy_id}/dry-run", {
          params: { path: { policy_id: id } },
          body: { text },
        }),
      ),
  });
}

// --- Audit ---

export interface AuditFilters {
  from?: string;
  to?: string;
  decision?: Decision;
  rule_id?: string;
  q?: string;
}

export function useAuditList(filters: AuditFilters) {
  return useInfiniteQuery({
    queryKey: ["audit", filters],
    queryFn: async ({ pageParam }: { pageParam?: string }) =>
      unwrap(
        await api.GET("/api/v1/audit", {
          params: {
            query: {
              from: filters.from,
              to: filters.to,
              decision: filters.decision,
              rule_id: filters.rule_id,
              q: filters.q,
              cursor: pageParam,
              limit: 50,
            },
          },
        }),
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
  });
}

export function useAuditDetail(id: string | null) {
  return useQuery({
    queryKey: ["audit", "detail", id],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/audit/{record_id}", {
          params: { path: { record_id: id! } },
        }),
      ),
    enabled: id !== null,
  });
}

// --- Stats ---

const POLL_INTERVAL_MS = 10_000;

export function useStatsSummary(window: string) {
  return useQuery({
    queryKey: ["stats", "summary", window],
    // Cast (not just an annotation) pins `top_rules` as `[string, number][]`:
    // TS's own structural inference through openapi-fetch's generics widens
    // that tuple to `(string | number)[]` before it ever reaches useQuery —
    // an annotation alone doesn't override an already-widened inferred type.
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/stats/summary", { params: { query: { window } } }),
      ) as StatsSummary,
    refetchInterval: POLL_INTERVAL_MS,
  });
}

export function useStatsTimeseries(window: string, bucket: string) {
  return useQuery({
    queryKey: ["stats", "timeseries", window, bucket],
    queryFn: async () =>
      unwrap(
        await api.GET("/api/v1/stats/timeseries", {
          params: { query: { window, bucket } },
        }),
      ),
    refetchInterval: POLL_INTERVAL_MS,
  });
}
