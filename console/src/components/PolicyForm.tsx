import { useState } from "react";

import { ENTITY_TYPES, useCreatePolicy, type Decision, type EntityType } from "@/api/queries";

const ACTIONS: Decision[] = ["allow", "mask", "block"];

const inputClass =
  "mt-1 block w-full border border-hairline-strong bg-paper px-3 py-1.5 text-sm text-ink focus:border-ink focus:outline-none dark:border-hairline-strong-dark dark:bg-paper-dark dark:text-ink-dark dark:focus:border-ink-dark";

export function PolicyForm({ onCreated }: { onCreated?: () => void }) {
  const [name, setName] = useState("");
  const [entityType, setEntityType] = useState<EntityType>("EMAIL");
  const [action, setAction] = useState<Decision>("mask");
  const [minConfidence, setMinConfidence] = useState(0.5);
  const [enabled, setEnabled] = useState(true);
  const [nameError, setNameError] = useState<string | null>(null);

  const createPolicy = useCreatePolicy();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) {
      setNameError("Name is required");
      return;
    }
    setNameError(null);

    createPolicy.mutate(
      {
        name: name.trim(),
        entity_type: entityType,
        action,
        min_confidence: minConfidence,
        enabled,
      },
      {
        onSuccess: () => {
          setName("");
          setMinConfidence(0.5);
          setEnabled(true);
          onCreated?.();
        },
      },
    );
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-4 border border-hairline-strong p-4 dark:border-hairline-strong-dark"
    >
      <h3 className="text-xs font-semibold tracking-wide text-ink uppercase dark:text-ink-dark">
        New policy
      </h3>

      <div>
        <label
          htmlFor="policy-name"
          className="block text-sm font-medium text-ink dark:text-ink-dark"
        >
          Name
        </label>
        <input
          id="policy-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className={inputClass}
          placeholder="Block IBAN-like secrets"
        />
        {nameError && <p className="mt-1 text-sm text-block dark:text-block-dark">{nameError}</p>}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div>
          <label
            htmlFor="policy-entity-type"
            className="block text-sm font-medium text-ink dark:text-ink-dark"
          >
            Entity type
          </label>
          <select
            id="policy-entity-type"
            value={entityType}
            onChange={(e) => setEntityType(e.target.value as EntityType)}
            className={`${inputClass} font-mono`}
          >
            {ENTITY_TYPES.map((et) => (
              <option key={et} value={et}>
                {et}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label
            htmlFor="policy-action"
            className="block text-sm font-medium text-ink dark:text-ink-dark"
          >
            Action
          </label>
          <select
            id="policy-action"
            value={action}
            onChange={(e) => setAction(e.target.value as Decision)}
            className={`${inputClass} font-mono uppercase`}
          >
            {ACTIONS.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label
            htmlFor="policy-min-confidence"
            className="block text-sm font-medium text-ink dark:text-ink-dark"
          >
            Min confidence
          </label>
          <input
            id="policy-min-confidence"
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={minConfidence}
            onChange={(e) => setMinConfidence(Number(e.target.value))}
            className={`${inputClass} font-mono`}
          />
        </div>
      </div>

      <label className="flex items-center gap-2 text-sm text-ink dark:text-ink-dark">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => setEnabled(e.target.checked)}
          className="h-4 w-4 accent-ink dark:accent-ink-dark"
        />
        Enabled
      </label>

      {createPolicy.isError && (
        <p className="font-mono text-sm text-block dark:text-block-dark">
          {createPolicy.error.message}
        </p>
      )}

      <button
        type="submit"
        disabled={createPolicy.isPending}
        className="border-2 border-ink bg-ink px-4 py-1.5 text-sm font-medium text-paper transition-colors hover:bg-transparent hover:text-ink disabled:opacity-50 dark:border-ink-dark dark:bg-ink-dark dark:text-paper-dark dark:hover:bg-transparent dark:hover:text-ink-dark"
      >
        {createPolicy.isPending ? "Creating…" : "Create policy"}
      </button>
    </form>
  );
}
