import { screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/test/test-utils";

import { PolicyTable } from "./PolicyTable";

describe("PolicyTable", () => {
  it("renders the empty state when given zero policies", () => {
    renderWithProviders(<PolicyTable policies={[]} onSelect={vi.fn()} />);

    expect(screen.getByText(/no policies yet/i)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});
