import logging
from dataclasses import asdict, dataclass

from livekit.agents import AgentSession, ConversationItemAddedEvent
from livekit.agents.llm import ChatMessage, MetricsReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnLatency:
    end_to_end: float
    end_of_turn_delay: float | None
    transcription_delay: float | None
    llm_time_to_first_token: float | None
    tts_time_to_first_byte: float | None


def measure_turn_latency(
    user_metrics: MetricsReport, assistant_metrics: MetricsReport
) -> TurnLatency | None:
    # Replies that do not answer a caller turn, such as the greeting, have no end-to-end time.
    if "e2e_latency" not in assistant_metrics:
        return None
    return TurnLatency(
        end_to_end=assistant_metrics["e2e_latency"],
        end_of_turn_delay=user_metrics.get("end_of_turn_delay"),
        transcription_delay=user_metrics.get("transcription_delay"),
        llm_time_to_first_token=assistant_metrics.get("llm_node_ttft"),
        tts_time_to_first_byte=assistant_metrics.get("tts_node_ttfb"),
    )


def format_seconds(seconds: float | None) -> str:
    return "unknown" if seconds is None else f"{seconds:.2f}s"


def describe_turn_latency(latency: TurnLatency, budget_seconds: float) -> str:
    verdict = "within" if latency.end_to_end <= budget_seconds else "OVER"
    return (
        f"turn latency {format_seconds(latency.end_to_end)} {verdict} budget "
        f"{format_seconds(budget_seconds)}: "
        f"end of turn {format_seconds(latency.end_of_turn_delay)}, "
        f"stt {format_seconds(latency.transcription_delay)}, "
        f"llm first token {format_seconds(latency.llm_time_to_first_token)}, "
        f"tts first byte {format_seconds(latency.tts_time_to_first_byte)}"
    )


def log_latency_for_each_turn(session: AgentSession, budget_seconds: float) -> None:
    latest_user_metrics: MetricsReport = {}

    def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
        nonlocal latest_user_metrics
        if not isinstance(event.item, ChatMessage):
            return
        if event.item.role == "user":
            latest_user_metrics = event.item.metrics
            return
        if event.item.role != "assistant":
            return
        latency = measure_turn_latency(latest_user_metrics, event.item.metrics)
        if latency is not None:
            logger.info(
                describe_turn_latency(latency, budget_seconds),
                extra={"turn_latency": asdict(latency)},
            )

    session.on("conversation_item_added", on_conversation_item_added)
