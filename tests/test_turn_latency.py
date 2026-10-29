import logging
from typing import cast

import pytest
from livekit import rtc
from livekit.agents import AgentSession, ConversationItemAddedEvent
from livekit.agents.llm import ChatMessage, MetricsReport

from voice_agent.metrics.turn_latency import (
    TurnLatency,
    describe_turn_latency,
    log_latency_for_each_turn,
    measure_turn_latency,
)

USER_METRICS: MetricsReport = {"end_of_turn_delay": 0.41, "transcription_delay": 0.05}
ASSISTANT_METRICS: MetricsReport = {
    "e2e_latency": 0.92,
    "llm_node_ttft": 0.31,
    "tts_node_ttfb": 0.15,
}


def test_turn_latency_combines_caller_and_agent_stages() -> None:
    assert measure_turn_latency(USER_METRICS, ASSISTANT_METRICS) == TurnLatency(
        end_to_end=0.92,
        end_of_turn_delay=0.41,
        transcription_delay=0.05,
        llm_time_to_first_token=0.31,
        tts_time_to_first_byte=0.15,
    )


def test_reply_without_end_to_end_time_such_as_the_greeting_is_not_a_measured_turn() -> None:
    assert measure_turn_latency({}, {"tts_node_ttfb": 0.2}) is None


def test_missing_stage_is_reported_as_unknown_rather_than_zero() -> None:
    latency = measure_turn_latency({}, {"e2e_latency": 0.8})

    assert latency is not None
    assert "stt unknown" in describe_turn_latency(latency, budget_seconds=1.0)


@pytest.mark.parametrize(("end_to_end", "verdict"), [(0.92, "within"), (1.4, "OVER")])
def test_latency_line_says_whether_the_turn_met_the_budget(end_to_end: float, verdict: str) -> None:
    latency = TurnLatency(end_to_end, 0.4, 0.05, 0.3, 0.15)

    line = describe_turn_latency(latency, budget_seconds=1.0)

    assert line.startswith(f"turn latency {end_to_end:.2f}s {verdict} budget 1.00s")


def emit_message(session: rtc.EventEmitter, role: str, metrics: MetricsReport) -> None:
    message = ChatMessage(role=role, content=["hello"], metrics=metrics)
    session.emit("conversation_item_added", ConversationItemAddedEvent(item=message))


def test_each_answered_turn_logs_one_latency_line(caplog: pytest.LogCaptureFixture) -> None:
    session = rtc.EventEmitter()
    log_latency_for_each_turn(cast(AgentSession, session), budget_seconds=1.0)

    with caplog.at_level(logging.INFO, logger="voice_agent.metrics.turn_latency"):
        emit_message(session, "assistant", {"tts_node_ttfb": 0.2})
        emit_message(session, "user", USER_METRICS)
        emit_message(session, "assistant", ASSISTANT_METRICS)

    [record] = caplog.records
    assert record.getMessage().startswith("turn latency 0.92s within budget")
    assert record.turn_latency["end_of_turn_delay"] == 0.41
