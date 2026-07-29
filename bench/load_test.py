#!/usr/bin/env python3
"""
Load test for SentinelLM's detection pipeline.

Targets POST /scan — the detect-only endpoint (regex + Presidio + semantic,
no LLM call). That's a deliberate choice, not a shortcut: /scan measures
exactly the thing the "detection pipeline" latency claim is about, with no
Ollama/OpenAI round-trip in the way to muddy the number. It is NOT the same
as end-to-end /v1/chat/completions latency, which also waits on the LLM
backend — this script does not claim to measure that, and the output says so.

Usage:
    python bench/load_test.py --url http://localhost:8000 --concurrency 100 --duration 60

Requires a running gateway (`docker compose up gateway db` or
`uvicorn app.main:app` from gateway/) — this hits real HTTP over the network,
not an in-process ASGI transport, so the number includes real socket/HTTP
overhead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Mixed payloads: clean text, PII-triggering, secret-triggering — the same
# three categories the policy engine treats differently (ALLOW / MASK / BLOCK),
# so the load test exercises all three code paths, not just the fast/empty one.
PAYLOADS: list[str] = [
    # Clean — should ALLOW
    "What's the weather like in San Francisco today?",
    "Summarize the quarterly report in three bullet points.",
    "Write a haiku about autumn leaves.",
    "Explain how a hash table works.",
    # PII — should MASK
    "My email is jane.doe@example.com, please follow up.",
    "Call me at 555-867-5309 when you get a chance.",
    "My SSN is 456-78-9012 for the background check.",
    "Charge it to 4111-1111-1111-1111, that's my Visa.",
    "reach me at john dot smith at company dot com",  # obfuscated — semantic pass
    "my social is four five six dash seven eight dash nine zero one two",
    # Secrets — should BLOCK
    "export AWS_KEY=AKIAIOSFODNN7EXAMPLE",
    "here's the github token: ghp_1234567890abcdefghijklmnopqrstuvwxyz",
    "the jwt is eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "password: hunter2 for the staging box",
]


@dataclass
class Result:
    latency_ms: float
    status: int
    decision: str | None


@dataclass
class Stats:
    results: list[Result] = field(default_factory=list)
    errors: int = 0

    def add(self, r: Result) -> None:
        self.results.append(r)


async def worker(
    client: httpx.AsyncClient,
    stats: Stats,
    stop_at: float,
    semaphore: asyncio.Semaphore,
) -> None:
    i = 0
    while time.monotonic() < stop_at:
        payload = PAYLOADS[i % len(PAYLOADS)]
        i += 1
        async with semaphore:
            t0 = time.perf_counter()
            try:
                resp = await client.post("/scan", json={"text": payload}, timeout=10.0)
                latency_ms = (time.perf_counter() - t0) * 1000
                decision = (
                    resp.json().get("decision") if resp.status_code == 200 else None
                )
                stats.add(
                    Result(
                        latency_ms=latency_ms,
                        status=resp.status_code,
                        decision=decision,
                    )
                )
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
                stats.errors += 1


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, round(p * (len(values) - 1))))
    return values[idx]


async def run(url: str, concurrency: int, duration: float) -> dict:
    stats = Stats()
    semaphore = asyncio.Semaphore(concurrency)
    stop_at = time.monotonic() + duration

    async with httpx.AsyncClient(base_url=url) as client:
        # One warm-up request outside the timing window — the semantic model's
        # first inference includes lazy CUDA/MKL init that isn't representative.
        try:
            await client.post("/scan", json={"text": "warm up"}, timeout=30.0)
        except httpx.HTTPError as e:
            raise SystemExit(
                f"Could not reach {url} — is the gateway running? ({e})"
            ) from e

        start_wall = time.monotonic()
        workers = [
            asyncio.create_task(worker(client, stats, stop_at, semaphore))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)
        elapsed = time.monotonic() - start_wall

    latencies = [r.latency_ms for r in stats.results]
    by_decision: dict[str, int] = {}
    for r in stats.results:
        if r.decision:
            by_decision[r.decision] = by_decision.get(r.decision, 0) + 1

    return {
        "url": url,
        "concurrency": concurrency,
        "duration_s": round(elapsed, 1),
        "total_requests": len(stats.results),
        "errors": stats.errors,
        "requests_per_sec": round(len(stats.results) / elapsed, 1) if elapsed else 0,
        "p50_ms": round(percentile(latencies, 0.50), 2),
        "p95_ms": round(percentile(latencies, 0.95), 2),
        "p99_ms": round(percentile(latencies, 0.99), 2),
        "max_ms": round(max(latencies), 2) if latencies else 0,
        "by_decision": by_decision,
    }


def write_results_md(result: dict, path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = f"""# SentinelLM load test results

Generated: {ts}
Command: `python bench/load_test.py --url {result["url"]} --concurrency {result["concurrency"]} --duration {result["duration_s"]}`

**This measures `/scan` (detection pipeline: regex + Presidio + semantic NER)
over real HTTP, not `/v1/chat/completions` end-to-end.** The LLM call is a
separate, network- and model-dependent cost that this number does not include
— see the script docstring for why that's the honest thing to measure here.

| Metric | Value |
|---|---|
| Concurrency | {result["concurrency"]} |
| Duration | {result["duration_s"]}s |
| Total requests | {result["total_requests"]} |
| Errors | {result["errors"]} |
| Throughput | {result["requests_per_sec"]} req/s |
| p50 latency | {result["p50_ms"]} ms |
| p95 latency | {result["p95_ms"]} ms |
| p99 latency | {result["p99_ms"]} ms |
| max latency | {result["max_ms"]} ms |

Decision breakdown: {result["by_decision"]}
"""
    path.write_text(md)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    result = await run(args.url, args.concurrency, args.duration)
    print(json.dumps(result, indent=2))

    out_path = Path(__file__).parent / "results.md"
    write_results_md(result, out_path)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
