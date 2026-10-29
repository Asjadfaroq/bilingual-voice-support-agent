import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from voice_agent.metrics.tracing import (
    CallTraceContext,
    LangfuseSpanExporter,
    model_call_cost,
    pseudonymous_caller_id,
)

PRICES = {"claude-haiku-5-5": (0.10, 0.50)}


def exported_spans(languages: list[str], *spans: tuple[str, dict]) -> list:
    captured = InMemorySpanExporter()
    context = CallTraceContext(
        call_id="room-42", caller_key=lambda: "+966551234567", languages=lambda: languages
    )
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(LangfuseSpanExporter(captured, context, "secret", PRICES))
    )
    tracer = provider.get_tracer("test")
    for name, attributes in spans:
        with tracer.start_as_current_span(name, attributes=attributes):
            pass
    return list(captured.get_finished_spans())


def test_conversation_content_is_masked_before_it_leaves_the_process() -> None:
    messages = json.dumps([{"role": "user", "content": "I am sara@example.com, 0551234567"}])

    [span] = exported_spans(["en"], ("llm_request", {"gen_ai.input.messages": messages}))

    exported = span.attributes["gen_ai.input.messages"]
    assert "sara@example.com" not in exported and "0551234567" not in exported
    assert json.loads(exported)[0]["content"] == "I am [email], [phone]"


def test_model_calls_carry_their_cost_from_the_configured_prices() -> None:
    usage = {
        "gen_ai.request.model": "claude-haiku-5-5",
        "gen_ai.usage.input_tokens": 2_000,
        "gen_ai.usage.output_tokens": 400,
    }

    [span] = exported_spans(["en"], ("llm_request", usage))

    cost = json.loads(span.attributes["langfuse.observation.cost_details"])
    assert cost == {"input": 0.0002, "output": 0.0002, "total": 0.0004}


def test_wrapper_spans_that_only_name_the_model_are_not_counted_as_generations() -> None:
    [wrapper] = exported_spans(["en"], ("llm_node", {"gen_ai.request.model": "claude-haiku-5-5"}))

    assert wrapper.attributes["langfuse.observation.type"] == "span"
    assert "langfuse.observation.cost_details" not in wrapper.attributes


def test_unpriced_or_tokenless_spans_get_no_cost() -> None:
    assert model_call_cost({"gen_ai.request.model": "other-model"}, PRICES) is None
    assert model_call_cost({"gen_ai.request.model": "claude-haiku-5-5"}, PRICES) is None


def test_every_span_names_the_call_with_a_pseudonymous_caller_and_its_languages() -> None:
    [span] = exported_spans(["ar", "en"], ("agent_session", {}))

    assert span.attributes["langfuse.session.id"] == "room-42"
    assert span.attributes["langfuse.trace.metadata.languages"] == "ar,en"
    assert span.attributes["langfuse.user.id"].startswith("caller-")
    assert "551234567" not in span.attributes["langfuse.user.id"]


def test_caller_id_is_stable_per_caller_and_does_not_reveal_the_number() -> None:
    first = pseudonymous_caller_id("+966551234567", "secret")

    assert first == pseudonymous_caller_id("+966551234567", "secret")
    assert first != pseudonymous_caller_id("+966551234568", "secret")
    assert pseudonymous_caller_id(None, "secret") == "caller-unknown"
