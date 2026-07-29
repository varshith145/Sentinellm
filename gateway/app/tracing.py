"""
OpenTelemetry wiring for the gateway.

`setup_tracing(app)` auto-instruments FastAPI (one span per HTTP request,
capturing route, status code, and duration) and exports via OTLP/HTTP to the
collector configured in settings — the bundled Jaeger service in
docker-compose by default.

`tracer` is used for the manual spans inside the request pipeline (see
main.py's chat_completions): detection, policy evaluation, and audit write
each get their own child span, so a single trace shows the full path —
gateway → detector inference → policy eval → audit write — as it's actually
timed, not just as one opaque HTTP span.

Disabled by default (settings.otel_enabled=False) so pytest and a plain
`uvicorn` run don't spend time retrying a collector that isn't running.
"""

import logging

from app.config import settings
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger("sentinellm")

tracer = trace.get_tracer("sentinellm")


def setup_tracing(app: FastAPI) -> None:
    """Instrument `app` and register the OTLP exporter. No-op if disabled."""
    if not settings.otel_enabled:
        return

    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: settings.otel_service_name})
    )
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    logger.info(
        f"OpenTelemetry tracing enabled, exporting to {settings.otel_exporter_endpoint}"
    )
