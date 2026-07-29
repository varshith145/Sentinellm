"""
Prometheus metrics for SentinelLM Gateway.

Exposes operational counters and histograms consumed by the /metrics endpoint.
Scraped by Prometheus every ~15s; visualised in Grafana or any compatible tool.

Metrics defined here:

  sentinellm_requests_total          Counter  — requests by decision (ALLOW/MASK/BLOCK)
  sentinellm_detections_total        Counter  — entity detections by type and detector
  sentinellm_detection_confidence    Histogram — per-finding confidence score, by detector
  sentinellm_blocks_total            Counter  — blocked requests by entity type that caused the block
  sentinellm_detection_latency_secs  Histogram — time for the three-pass detection pipeline
  sentinellm_llm_latency_secs        Histogram — time waiting for the LLM backend
  sentinellm_request_latency_secs    Histogram — total end-to-end request time
  sentinellm_active_detectors        Gauge     — number of active detector passes (set at startup)
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Request-level counters ─────────────────────────────────────────────────────

REQUEST_COUNTER = Counter(
    "sentinellm_requests_total",
    "Total requests processed by SentinelLM, labelled by policy decision.",
    ["decision"],  # "ALLOW", "MASK", "BLOCK"
)

# ── Detection counters ─────────────────────────────────────────────────────────

DETECTION_COUNTER = Counter(
    "sentinellm_detections_total",
    "Total entity detections, labelled by entity type and the detector that found it.",
    ["entity_type", "detector"],  # e.g. entity_type="EMAIL", detector="regex"
)

BLOCKS_COUNTER = Counter(
    "sentinellm_blocks_total",
    "Blocked requests broken down by the entity type that triggered the block.",
    ["entity_type"],
)

DETECTION_CONFIDENCE = Histogram(
    "sentinellm_detection_confidence",
    "Distribution of per-finding confidence scores, labelled by detector.",
    ["detector"],
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0],
)

# ── Latency histograms ─────────────────────────────────────────────────────────

DETECTION_LATENCY = Histogram(
    "sentinellm_detection_latency_secs",
    "Time in seconds for the full three-pass detection pipeline (regex + Presidio + semantic).",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0],
)

LLM_LATENCY = Histogram(
    "sentinellm_llm_latency_secs",
    "Time in seconds waiting for the LLM backend to respond.",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

REQUEST_LATENCY = Histogram(
    "sentinellm_request_latency_secs",
    "Total end-to-end request processing time in seconds.",
    # Fine-grained low end for the honest per-request number (see
    # bench/results.md), extended past 1s because a single-worker process
    # under load queues real requests well past that — a load test measured
    # p95=913ms / max=1338ms at 100 concurrent. Buckets that stop at 1.0
    # dump every overloaded request into +Inf and make histogram_quantile
    # lie on the Grafana p95 panel exactly when it matters most.
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── Gauges ─────────────────────────────────────────────────────────────────────

ACTIVE_DETECTORS = Gauge(
    "sentinellm_active_detectors",
    "Number of active detection passes currently loaded (max 3: regex, presidio, semantic).",
)
