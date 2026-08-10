#!/usr/bin/env python3
"""
Fixed-baseline load test sweep for SentinelLM's /scan endpoint.

Sweeps concurrency levels 1, 10, 50, 100 against the same fixed payload set,
same duration per level, and a seeded deterministic payload order — all four
levels run inside one process invocation and get written to a single
BENCHMARKS.md at the end, so a later level can never silently overwrite an
earlier one's numbers.

Targets POST /scan (regex + Presidio + semantic detection, no LLM call) over
real HTTP — same rationale as bench/load_test.py: this measures the
detection pipeline the "latency" claim is actually about, not
/v1/chat/completions, which also waits on an LLM backend.

One-command reproduction (gateway must already be running — see README for
how to start it):

    python benchmarks/loadtest.py --url http://localhost:8000

Payload order is deterministic: a fixed payload list is shuffled once with
`random.Random(seed)` (default seed 42, override with --seed) and then each
concurrent worker round-robins through that fixed order starting at its own
worker index. Same seed, same url, same durations -> same requests in the
same order, every time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent.parent

# Same three categories the policy engine treats differently (ALLOW / MASK /
# BLOCK), so the sweep exercises all three code paths, not just one.
PAYLOADS: list[str] = [
    "What's the weather like in San Francisco today?",
    "Summarize the quarterly report in three bullet points.",
    "Write a haiku about autumn leaves.",
    "Explain how a hash table works.",
    "My email is jane.doe@example.com, please follow up.",
    "Call me at 555-867-5309 when you get a chance.",
    "My SSN is 456-78-9012 for the background check.",
    "Charge it to 4111-1111-1111-1111, that's my Visa.",
    "reach me at john dot smith at company dot com",
    "my social is four five six dash seven eight dash nine zero one two",
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


async def worker(
    client: httpx.AsyncClient,
    stats: Stats,
    stop_at: float,
    payload_order: list[str],
    start_index: int,
) -> None:
    i = start_index
    while time.monotonic() < stop_at:
        payload = payload_order[i % len(payload_order)]
        i += 1
        t0 = time.perf_counter()
        try:
            resp = await client.post("/scan", json={"text": payload}, timeout=10.0)
            latency_ms = (time.perf_counter() - t0) * 1000
            decision = (
                resp.json().get("decision") if resp.status_code == 200 else None
            )
            stats.results.append(
                Result(latency_ms=latency_ms, status=resp.status_code, decision=decision)
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError):
            stats.errors += 1


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = max(0, min(len(values) - 1, round(p * (len(values) - 1))))
    return values[idx]


async def run_level(
    client: httpx.AsyncClient, concurrency: int, duration: float, payload_order: list[str]
) -> dict:
    stats = Stats()
    stop_at = time.monotonic() + duration
    start_wall = time.monotonic()
    workers = [
        asyncio.create_task(worker(client, stats, stop_at, payload_order, w))
        for w in range(concurrency)
    ]
    await asyncio.gather(*workers)
    elapsed = time.monotonic() - start_wall

    latencies = [r.latency_ms for r in stats.results]
    by_decision: dict[str, int] = {}
    for r in stats.results:
        if r.decision:
            by_decision[r.decision] = by_decision.get(r.decision, 0) + 1

    return {
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


def git_info() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    sha = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    return {"sha": sha, "dirty": dirty}


def hardware_info() -> dict:
    mem_gb = "unknown"
    try:
        if platform.system() == "Darwin":
            raw = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            ).stdout.strip()
            mem_gb = f"{int(raw) / (1024**3):.0f}GB"
        elif platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        mem_gb = f"{kb / (1024**2):.0f}GB"
                        break
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass

    return {
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "memory": mem_gb,
    }


def read_eval_micro() -> dict | None:
    """Read the micro-average row eval/run_eval.py just wrote to eval/results.md.

    Parses rather than re-running the eval harness in-process: this script's
    job is the HTTP load test, and eval/run_eval.py already loads the full
    detector stack (torch, presidio, spacy) on its own. Run
    `python eval/run_eval.py` first so this row is current, not stale.
    """
    results_path = REPO_ROOT / "eval" / "results.md"
    if not results_path.exists():
        return None
    match = re.search(
        r"\*\*Micro-average\*\* \| \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\*",
        results_path.read_text(),
    )
    if not match:
        return None
    return {"precision": match[1], "recall": match[2], "f1": match[3]}


async def fetch_health(url: str) -> dict:
    async with httpx.AsyncClient(base_url=url) as client:
        resp = await client.get("/health", timeout=10.0)
        resp.raise_for_status()
        return resp.json()


def write_benchmarks_md(
    levels: list[dict],
    url: str,
    duration: float,
    seed: int,
    workers: str,
    db_backend: str,
    health: dict,
    path: Path,
) -> None:
    git = git_info()
    hw = hardware_info()
    eval_micro = read_eval_micro()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dirty_note = " (dirty tree — uncommitted changes present)" if git["dirty"] else ""

    rows = "\n".join(
        f"| {lv['concurrency']} | {lv['requests_per_sec']} req/s | {lv['p50_ms']}ms | "
        f"{lv['p95_ms']}ms | {lv['p99_ms']}ms | {lv['total_requests']} | {lv['errors']} |"
        for lv in levels
    )

    c1 = next((lv for lv in levels if lv["concurrency"] == 1), None)
    c1_note = ""
    if c1 and c1["total_requests"] < 200:
        c1_note = (
            f"\nNote: concurrency=1 collected only {c1['total_requests']} requests in "
            f"{duration}s (one request in flight at a time caps sample size) — its p99 is "
            "close to its max and not directly comparable to the p99 at higher concurrency.\n"
        )

    md = f"""# SentinelLM benchmarks

Generated: {ts}
Git SHA: `{git["sha"]}`{dirty_note}
Hardware: {hw["machine"]}, {hw["cpu_count"]} cores, {hw["memory"]} RAM, {hw["system"]}

## Reproduce

```bash
python benchmarks/loadtest.py --url {url}
```

Requires a running gateway (see README for startup). Deterministic by
construction: fixed payload set, fixed duration per level ({duration}s),
payload order seeded with `--seed {seed}` (default 42) and round-robined
per worker — same inputs always produce the same request sequence.

## Gateway config at time of run

| Setting | Value |
|---|---|
| Workers | {workers} |
| Active detectors | {", ".join(health.get("detectors", [])) or "unknown"} |
| Policy ID | {health.get("policy_id", "unknown")} |
| DB backend | {db_backend} |

## Load test: `/scan` (detection pipeline, no LLM in the path)

| Concurrency | Throughput | p50 | p95 | p99 | Requests | Errors |
|---|---|---|---|---|---|---|
{rows}
{c1_note}
This measures `/scan` (regex + Presidio + semantic NER) over real HTTP —
not `/v1/chat/completions` end-to-end, which also pays an LLM round-trip.
See [`bench/results.md`](bench/results.md) for the prior single-point
optimization investigation (short-circuit + multi-worker) this baseline
was measured on top of.

## Eval: ensemble micro-F1

```bash
python eval/run_eval.py
```
"""
    if eval_micro:
        md += f"""
| Metric | Value |
|---|---|
| Micro-F1 | {eval_micro["f1"]} |
| Precision | {eval_micro["precision"]} |
| Recall | {eval_micro["recall"]} |

Read from `eval/results.md` at write time — run the command above first so
that file is current before running this sweep. Full methodology in
[`eval/results.md`](eval/results.md).
"""
    else:
        md += """
`eval/results.md` not found — run the command above first, then re-run this
sweep so its numbers get pulled in here.
"""
    path.write_text(md)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrencies", default="1,10,50,100")
    parser.add_argument("--duration", type=float, default=20.0, help="seconds per level")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", default="4 (uvicorn, per gateway/Dockerfile default)")
    parser.add_argument("--db-backend", default="sqlite (aiosqlite)")
    parser.add_argument("--out", default=str(REPO_ROOT / "BENCHMARKS.md"))
    args = parser.parse_args()

    concurrencies = [int(c) for c in args.concurrencies.split(",")]
    payload_order = PAYLOADS.copy()
    random.Random(args.seed).shuffle(payload_order)

    async with httpx.AsyncClient(base_url=args.url) as client:
        try:
            await client.post("/scan", json={"text": "warm up"}, timeout=30.0)
        except httpx.HTTPError as e:
            raise SystemExit(f"Could not reach {args.url} — is the gateway running? ({e})") from e

        levels = []
        for concurrency in concurrencies:
            result = await run_level(client, concurrency, args.duration, payload_order)
            print(json.dumps(result, indent=2))
            levels.append(result)

    health = await fetch_health(args.url)

    out_path = Path(args.out)
    write_benchmarks_md(
        levels, args.url, args.duration, args.seed, args.workers, args.db_backend, health, out_path
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
