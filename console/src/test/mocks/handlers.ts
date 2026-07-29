import { http, HttpResponse } from "msw";

// Wildcard host match (`*/api/v1/...`) rather than a fixed base URL — tests
// shouldn't depend on which VITE_API_BASE_URL happens to resolve in
// whatever mode Vitest loads env files under.

export const samplePolicy = {
  id: "11111111-1111-1111-1111-111111111111",
  name: "EMAIL default",
  description: null,
  entity_type: "EMAIL",
  category: "PII",
  action: "mask",
  min_confidence: 0.7,
  enabled: true,
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-01T00:00:00Z",
};

export const handlers = [
  http.get("*/api/v1/policies", () =>
    HttpResponse.json({ items: [samplePolicy], next_cursor: null }),
  ),

  http.post("*/api/v1/policies", async ({ request }) => {
    const body = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json(
      {
        id: "22222222-2222-2222-2222-222222222222",
        description: null,
        min_confidence: 0.5,
        enabled: true,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
        ...body,
      },
      { status: 201 },
    );
  }),

  http.get("*/api/v1/audit", () => HttpResponse.json({ items: [], next_cursor: null })),
];
