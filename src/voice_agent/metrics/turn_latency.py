import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass

from livekit.agents import AgentSession, ConversationItemAddedEvent, get_job_context
from livekit.agents.llm import ChatMessage, MetricsReport

logger = logging.getLogger(__name__)
# Publishing is fire-and-forget; the tasks are kept so they are not garbage collected mid-send.
pending_publishes: set[asyncio.Task[None]] = set()


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


def publish_measurement(topic: str | None, measurement: dict[str, object]) -> None:
    if topic is None:
        return
    room = get_job_context().room
    task = asyncio.create_task(
        room.local_participant.publish_data(json.dumps(measurement), reliable=True, topic=topic)
    )
    pending_publishes.add(task)
    task.add_done_callback(pending_publishes.discard)


def reply_filter_hold_reporter(topic: str | None) -> Callable[[float], None]:
    def report_hold(seconds: float) -> None:
        logger.info("reply filter held the first sentence for %.2fs", seconds)
        publish_measurement(topic, {"kind": "reply_filter_hold", "seconds": seconds})

    return report_hold


def log_latency_for_each_turn(
    session: AgentSession, budget_seconds: float, report_topic: str | None = None
) -> None:
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
            publish_measurement(report_topic, {"kind": "turn", **asdict(latency)})

    session.on("conversation_item_added", on_conversation_item_added)
