import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AuditTable } from "./AuditTable";

describe("AuditTable", () => {
  it("renders the empty state when the API returns zero rows", () => {
    render(<AuditTable records={[]} onSelect={vi.fn()} />);

    expect(screen.getByText(/no audit records match these filters/i)).toBeInTheDocument();
    expect(screen.queryByRole("table")).not.toBeInTheDocument();
  });
});
