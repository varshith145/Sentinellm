# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**Primary (product truth):** Security/platform engineers at organizations adopting LLMs (ChatGPT, Copilot, internal Ollama/OpenAI deployments) who need to stop employees from leaking PII and secrets into prompts, and who currently have no audit trail of what was sent to a model.

**Primary (current, confirmed audience):** Technical interviewers/hiring managers at **Apple**, evaluating this as a portfolio project for a role where front-end craft is explicitly weighed alongside backend/ML/security engineering — the build plan that drove this session's work states this directly ("a design-literate company," "the role is explicitly about intuitive front-end experiences"). This is not a toy demo built to *look* real — it is built to real production standards specifically because that audience will inspect the engineering, not just the screenshots.

## Product Purpose

SentinelLM is a self-hosted AI gateway that sits between any application and an LLM backend. It inspects every prompt and every response through a three-pass detection pipeline (regex, Microsoft Presidio, a fine-tuned DistilBERT semantic NER model), enforces a configurable policy (allow/mask/block per entity type), and logs every decision to an audit trail that stores only redacted content.

Success has two, deliberately non-competing halves:
1. **It actually works** — it catches obfuscated/informal PII and secrets that regex-only scanners miss, with every performance and accuracy number backed by a committed, rerunnable script (no asserted metric that can't be regenerated on demand).
2. **It is legible and credible to a technical evaluator in minutes** — the dry-run panel, a live policy edit that visibly changes a decision, and real (not mocked) metrics are the pitch, not decoration on top of one.

## Positioning

Three-pass detection (regex + Presidio + fine-tuned semantic NER) that catches what pattern-matching-only tools structurally cannot: obfuscated/informal PII like "reach me at john dot smith at company dot com." Fully self-hosted — no data ever leaves the deployment to a third-party scanning API (unlike Lakera Guard, Nightfall AI). Scans both input and output; most open-source alternatives (LLM Guard) only scan input. Positioned against gateway-only tools (LiteLLM, Portkey AI Gateway) that handle routing/observability but do no security scanning at all, and against Presidio itself, which is a library SentinelLM uses as one of three passes, not a competing product.

The console (in progress) is what turns "has a REST API" into "developers actually author policy and watch decisions here" — the gap every competitor above leaves as a config file or a raw API.

## Operating Context

- **Local/full stack:** `docker compose up` — gateway (FastAPI) + Postgres + Streamlit admin dashboard + Prometheus + Grafana (5-panel dashboard, zero-click provisioned) + Jaeger tracing, in front of a local Ollama model or OpenAI-compatible backend.
- **Public demo:** Hugging Face Spaces (free tier, no card) — detection-only (`/scan`), SQLite, no LLM in the path, deliberately scoped because that tier can't hold persistent state reliably across sleep cycles.
- **Console (new, this session):** `console/` — Vite + React 18 + TypeScript (strict) + Tailwind + TanStack Query + React Router + Recharts, talking to a versioned `/api/v1/*` REST surface (policy CRUD + dry-run, audit search with cursor pagination, live stats) via a client generated from the gateway's own OpenAPI schema. Intended to also get a public, **read-only** deployment on Vercel pointed at the HF Spaces backend — a server-side flag (`SENTINELLM_CONSOLE_READ_ONLY`) blocks writes so that link is safe to hand to a stranger. Not yet deployed as of this writing; the steps are documented, execution needs dashboard access this session didn't have.
- **Model training:** separate Docker training image (`model/train.py`, `Dockerfile.train`) producing a fine-tuned DistilBERT checkpoint, uploaded independently to the Hugging Face Hub (`varshith145/sentinellm-pii-ner`) since the model isn't committed to git.

## Capabilities and Constraints

- Three detectors run concurrently (`asyncio.gather`); findings are deduplicated by span overlap, ties broken by confidence then detector priority (semantic > presidio > regex).
- Policy is **DB-authoritative at runtime**: a `policies` table, seeded once from `gateway/policies/default.yaml` if empty. Console writes (`/api/v1/policies`) change live gateway behavior immediately — verified end-to-end in this session (a PATCH flipped a real `/scan` decision with no restart). The tradeoff this cost: the old "edit the YAML, no rebuild needed" workflow no longer applies once the table is populated.
- Detection is CPU-bound, not I/O-bound. Measured on Apple M4 (10 cores): a single `--workers 1` process (the actual `gateway/Dockerfile` config) saturates at **~130 req/s** regardless of concurrency. Below that ceiling, p95 is **~86ms**; above it, requests queue and p95 rises to **~913ms at 100 concurrent** — that's queueing delay on an overloaded process, not per-request cost increasing. This is why the resume's original "p95 under 50ms at 100 concurrent" claim was rewritten rather than kept.
- Pipeline-level evaluation (regex + Presidio + semantic combined, IoU ≥ 0.5 span matching, same 61-example held-out split the raw model reports on): PII F1 0.9818, SECRET F1 0.9231, micro F1 0.9574 — higher than the raw model alone (F1 0.849), because the shipped system is the ensemble, not the model in isolation. The precision=1.0 result was interrogated (checked stable across match-strictness thresholds, checked zero false positives on hard negatives) before being trusted.
- Entity taxonomy: `EMAIL, PHONE, SSN, CREDIT_CARD, AWS_KEY, GITHUB_TOKEN, JWT, PASSWORD, PERSON_NAME, GENERIC_PII, GENERIC_SECRET`. Decisions: `ALLOW, MASK, BLOCK`. "Dry-run" = test a candidate policy rule against sample text with no audit record written.
- Test suite: 281 pytest (backend, Python 3.12/SQLite/semantic-disabled locally) + 11 Vitest/RTL (console). The README's test-count badge is intentionally left at its last CI-confirmed value until the new counts are green on CI's actual environment (Python 3.11 + real Postgres) — a local number does not overwrite a badge.
- **Undecided / explicitly not done yet:** the console is not deployed to Vercel; the HF Spaces environment variables for the read-only console guard are not yet set; the README's test-count and any resume document outside this repo have not been updated to reflect this session's numbers.

## Brand Commitments

Name: **SentinelLM**. The README badges and Hugging Face Spaces `Dockerfile`/frontmatter still carry a shield emoji 🛡️ and `colorFrom: indigo`/`colorTo: blue` — that identity belongs to the gateway's existing public-facing docs/demo and was never touched. The **console** has its own committed visual system, "The Instrument Room" (a seismograph/polygraph recording-desk world — see `DESIGN.md`), deliberately distinct from that indigo mark: paper ground, ink structure, two severity inks (vermillion/BLOCK, ochre/MASK). Full detail in `DESIGN.md` and `.impeccable/surfaces/console.md`.

## Evidence on Hand

- `bench/results.md` — real load-test output (Apple M4, 10 cores), including the concurrency-10-vs-100 comparison and the single-worker-saturation finding.
- `eval/results.md` — real pipeline-level precision/recall/F1, confusion matrix, and the match-strictness robustness check.
- `docs/model_eval.json` — the raw semantic model's own eval (F1 0.849, precision 0.789, recall 0.918, 61 test examples).
- Live public demo: https://huggingface.co/spaces/varshith145/sentinellm (detection-only).
- `docs_prd.md` (dated 2026-02-11) — the original planning PRD. Useful for early intent, problem statement, and competitive landscape, but **incomplete** (cuts off mid-document; sections on Known Limitations and Future Work were never written) and **predates** the console, the observability stack, and DB-authoritative policy — treat as historical context, not a current spec.
- **No real customers, testimonials, case studies, production deployments, or third-party benchmarks exist.** This is a portfolio/pre-product project; future work must not invent any of these.

## Product Principles

1. Every claimed number is regenerable by one committed command — `bench/load_test.py`, `eval/run_eval.py` — never an asserted metric without a script that reproduces it.
2. Detection is layered on purpose: fast structural matching, backstopped by general-purpose NLP, backstopped by a model trained specifically for the obfuscation case pattern-matching structurally can't reach.
3. Self-hosted and audit-first — raw sensitive data never leaves the deployment or gets persisted; only redacted content is ever stored.
4. Runtime-editable beats restart-required wherever the tradeoff is worth it, and the tradeoff is stated, not hidden, when something is lost (e.g., losing the YAML hot-reload workflow when policy became DB-authoritative).
5. Every surface — including the public demo — degrades honestly rather than silently: read-only guards instead of hidden write access, graceful detector fallback, a clean 503 instead of a crash.
