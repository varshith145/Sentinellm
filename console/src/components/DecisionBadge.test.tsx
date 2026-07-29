import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DecisionBadge } from "./DecisionBadge";

describe("DecisionBadge", () => {
  it.each([
    ["allow", "text-ink"],
    ["mask", "text-mask"],
    ["block", "text-block"],
  ] as const)("renders the %s variant with %s styling", (decision, colorHint) => {
    render(<DecisionBadge decision={decision} />);

    const badge = screen.getByText(decision);
    expect(badge).toBeInTheDocument();
    expect(badge.className).toContain(colorHint);
  });
});
