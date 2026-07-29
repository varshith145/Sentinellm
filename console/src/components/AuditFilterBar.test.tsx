import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useSearchParams } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { AuditFilterBar } from "./AuditFilterBar";

// Renders the URL's current query string as visible text, so the test can
// assert on it without reaching into react-router internals.
function LocationProbe() {
  const [params] = useSearchParams();
  return <div data-testid="location-probe">{params.toString()}</div>;
}

function renderFilterBar() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/audit"]}>
        <AuditFilterBar />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("AuditFilterBar", () => {
  it("writes the decision filter into the URL query string", async () => {
    const user = userEvent.setup();
    renderFilterBar();

    await user.selectOptions(screen.getByRole("combobox"), "block");

    await waitFor(() =>
      expect(screen.getByTestId("location-probe")).toHaveTextContent("decision=block"),
    );
  });

  it("writes the free-text search into the URL query string (debounced)", async () => {
    const user = userEvent.setup();
    renderFilterBar();

    await user.type(screen.getByPlaceholderText(/search redacted content/i), "aws");

    await waitFor(() =>
      expect(screen.getByTestId("location-probe")).toHaveTextContent("q=aws"),
    );
  });
});
