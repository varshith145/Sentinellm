import { Fragment } from "react";

const TOKEN_RE = /(\[REDACTED_\w*\])/g;

/**
 * Marks redaction tokens in their severity ink, like a technician circling
 * an anomaly on a chart — the signature device named in the surface brief,
 * applied everywhere redacted gateway text is shown (dry-run result, audit
 * preview, audit detail drawer).
 *
 * Category comes straight from the token text itself: the backend collapses
 * every secret entity type (AWS_KEY, GITHUB_TOKEN, JWT, PASSWORD,
 * GENERIC_SECRET) to the literal token `[REDACTED_SECRET]` (see
 * gateway/app/redact.py's REDACTION_TOKENS), so `[REDACTED_SECRET]` is
 * always a block-severity mark and every other `[REDACTED_*]` token is
 * always PII — no separate entity-type lookup needed, this is exact, not a
 * heuristic.
 */
export function RedactedText({ text }: { text: string }) {
  const parts = text.split(TOKEN_RE);

  return (
    <>
      {parts.map((part, i) => {
        if (!TOKEN_RE.test(part)) {
          TOKEN_RE.lastIndex = 0;
          return <Fragment key={i}>{part}</Fragment>;
        }
        TOKEN_RE.lastIndex = 0;
        const isSecret = part === "[REDACTED_SECRET]";
        return (
          <mark
            key={i}
            className={`rounded-none bg-transparent underline decoration-2 underline-offset-2 ${
              isSecret
                ? "text-block decoration-block dark:text-block-dark dark:decoration-block-dark"
                : "text-mask decoration-mask dark:text-mask-dark dark:decoration-mask-dark"
            }`}
          >
            {part}
          </mark>
        );
      })}
    </>
  );
}
