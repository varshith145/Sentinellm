import createClient from "openapi-fetch";

import type { paths } from "./types";

const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// Only sent if you've set it — the public HF Spaces demo runs with auth
// disabled (console_api_key empty) and relies on the read-only guard
// instead, so most deployments never need this.
const apiKey = import.meta.env.VITE_CONSOLE_API_KEY;

export const api = createClient<paths>({
  baseUrl,
  headers: apiKey ? { "X-API-Key": apiKey } : undefined,
  // openapi-fetch defaults to `globalThis.fetch` captured once at
  // createClient() call time (module-eval time, since `api` is a module
  // top-level const) — a plain reference, not a live lookup. MSW's Node
  // interceptor patches `globalThis.fetch` by reassigning it, which only
  // takes effect for code that reads `globalThis.fetch` at call time. This
  // wrapper does that live lookup so tests (which patch fetch in a
  // `beforeAll` that runs after these imports evaluate) actually get
  // intercepted, rather than silently hitting the real network.
  fetch: (...args: Parameters<typeof fetch>) => globalThis.fetch(...args),
});

/**
 * openapi-fetch never throws on non-2xx — it returns `{ data, error }`.
 * This normalizes that into a thrown Error so TanStack Query's built-in
 * error handling (isError, error, onError) works without every call site
 * re-checking `error` by hand.
 */
export function unwrap<T>(result: { data?: T; error?: unknown }): T {
  if (result.error !== undefined) {
    const detail =
      typeof result.error === "object" &&
      result.error !== null &&
      "detail" in result.error
        ? String((result.error as { detail: unknown }).detail)
        : "Request failed";
    throw new Error(detail);
  }
  if (result.data === undefined) {
    throw new Error("Empty response");
  }
  return result.data;
}
