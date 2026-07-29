import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { server } from "@/test/mocks/server";
import { renderWithProviders } from "@/test/test-utils";

import { PolicyForm } from "./PolicyForm";

describe("PolicyForm", () => {
  it("shows a validation error when name is empty on submit", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PolicyForm />);

    await user.click(screen.getByRole("button", { name: /create policy/i }));

    expect(await screen.findByText(/name is required/i)).toBeInTheDocument();
  });

  it("calls the create endpoint with the correct payload shape", async () => {
    let capturedBody: unknown;
    server.use(
      http.post("*/api/v1/policies", async ({ request }) => {
        capturedBody = await request.json();
        return HttpResponse.json(
          {
            id: "x",
            description: null,
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
            ...(capturedBody as object),
          },
          { status: 201 },
        );
      }),
    );

    const user = userEvent.setup();
    renderWithProviders(<PolicyForm />);

    await user.type(screen.getByLabelText(/^name$/i), "Block IBAN secrets");
    await user.selectOptions(screen.getByLabelText(/entity type/i), "AWS_KEY");
    await user.selectOptions(screen.getByLabelText(/^action$/i), "block");
    await user.click(screen.getByRole("button", { name: /create policy/i }));

    await waitFor(() =>
      expect(capturedBody).toMatchObject({
        name: "Block IBAN secrets",
        entity_type: "AWS_KEY",
        action: "block",
        enabled: true,
      }),
    );
  });

  it("surfaces a server 4xx as a visible error message", async () => {
    server.use(
      http.post("*/api/v1/policies", () =>
        HttpResponse.json(
          { detail: "This deployment is read-only — policy writes are disabled." },
          { status: 403 },
        ),
      ),
    );

    const user = userEvent.setup();
    renderWithProviders(<PolicyForm />);

    await user.type(screen.getByLabelText(/^name$/i), "x");
    await user.click(screen.getByRole("button", { name: /create policy/i }));

    expect(await screen.findByText(/read-only/i)).toBeInTheDocument();
  });

  it("resets the name field after a successful create", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PolicyForm />);

    const nameInput = screen.getByLabelText(/^name$/i);
    await user.type(nameInput, "Temp policy");
    await user.click(screen.getByRole("button", { name: /create policy/i }));

    await waitFor(() => expect(nameInput).toHaveValue(""));
  });
});
