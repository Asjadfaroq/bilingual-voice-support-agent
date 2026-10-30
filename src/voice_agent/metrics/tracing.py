import base64
import hashlib
import hmac
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from livekit.agents.telemetry import set_tracer_provider
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

from voice_agent.config import Settings
from voice_agent.privacy.pii import mask_pii

TRACE_NAME = "support_call"
OTEL_TRACES_PATH = "/api/public/otel/v1/traces"
CALLER_ID_LENGTH = 12
TOKENS_PER_PRICE_UNIT = 1_000_000
ModelPrices = dict[str, tuple[float, float]]


@dataclass(frozen=True)
class CallTraceContext:
    call_id: str
    # Read at export time, so the root span, which ends last, carries the final values.
    caller_key: Callable[[], str | None]
    languages: Callable[[], list[str]]


def pseudonymous_caller_id(caller_key: str | None, secret: str) -> str:
    if not caller_key:
        return "caller-unknown"
    digest = hmac.new(secret.encode(), caller_key.encode(), hashlib.sha256).hexdigest()
    return f"caller-{digest[:CALLER_ID_LENGTH]}"


def mask_attribute(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [mask_pii(item) if isinstance(item, str) else item for item in value]
    return value


def model_cost(
    prices: ModelPrices, model: str, input_tokens: int, output_tokens: int
) -> dict[str, float] | None:
    if model not in prices:
        return None
    input_price, output_price = prices[model]
    input_cost = input_tokens * input_price / TOKENS_PER_PRICE_UNIT
    output_cost = output_tokens * output_price / TOKENS_PER_PRICE_UNIT
    return {"input": input_cost, "output": output_cost, "total": input_cost + output_cost}


def model_call_cost(attributes: dict[str, Any], prices: ModelPrices) -> dict[str, float] | None:
    if "gen_ai.usage.input_tokens" not in attributes:
        return None
    model = attributes.get("gen_ai.response.model") or attributes.get("gen_ai.request.model")
    output_tokens = attributes.get("gen_ai.usage.output_tokens", 0)
    return model_cost(prices, model, attributes["gen_ai.usage.input_tokens"], output_tokens)


class LangfuseSpanExporter(SpanExporter):
    """Masks PII in every exported string and adds the call's trace attributes and model cost."""

    def __init__(
        self,
        inner: SpanExporter,
        context: CallTraceContext,
        caller_id_secret: str,
        prices: ModelPrices,
    ) -> None:
        self._inner = inner
        self._context = context
        self._caller_id_secret = caller_id_secret
        self._prices = prices

    def _trace_attributes(self) -> dict[str, Any]:
        caller_id = pseudonymous_caller_id(self._context.caller_key(), self._caller_id_secret)
        return {
            "langfuse.trace.name": TRACE_NAME,
            "langfuse.session.id": self._context.call_id,
            "langfuse.user.id": caller_id,
            "langfuse.trace.metadata.languages": ",".join(self._context.languages()),
        }

    def _prepare(self, span: ReadableSpan) -> ReadableSpan:
        attributes = {
            name: mask_attribute(value) for name, value in (span.attributes or {}).items()
        }
        if (cost := model_call_cost(attributes, self._prices)) is not None:
            attributes["langfuse.observation.cost_details"] = json.dumps(cost)
        elif "gen_ai.request.model" in attributes:
            # LiveKit's wrapper spans name the model too; without this Langfuse counts them as
            # extra generations and estimates tokens and cost for them a second time.
            attributes["langfuse.observation.type"] = "span"
        attributes.update(self._trace_attributes())
        events = [
            Event(
                event.name,
                {name: mask_attribute(value) for name, value in (event.attributes or {}).items()},
                event.timestamp,
            )
            for event in span.events
        ]
        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=attributes,
            events=events,
            links=span.links,
            kind=span.kind,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([self._prepare(span) for span in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def start_call_tracing(settings: Settings, context: CallTraceContext) -> TracerProvider | None:
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None
    public_key = settings.langfuse_public_key.get_secret_value()
    secret_key = settings.langfuse_secret_key.get_secret_value()
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    otlp = OTLPSpanExporter(
        endpoint=f"{settings.langfuse_host}{OTEL_TRACES_PATH}",
        headers={"Authorization": f"Basic {credentials}", "x-langfuse-ingestion-version": "4"},
    )
    exporter = LangfuseSpanExporter(
        otlp, context, secret_key, settings.model_prices_usd_per_million_tokens
    )
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    # Content stays in the trace so Langfuse can show the conversation; it is masked above.
    set_tracer_provider(provider, allow_pii=True)
    return provider
