import html
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

import anthropic
from livekit.agents.llm import ChatContext, ChatMessage

from voice_agent.config import Settings
from voice_agent.crm.hubspot_client import HubSpotClient
from voice_agent.pipeline.language import CallerLanguage
from voice_agent.privacy.pii import mask_pii

logger = logging.getLogger(__name__)

CallSummariser = Callable[[str], Awaitable[str]]

SPEAKER_LABELS = {"user": "Caller", "assistant": "Agent"}
LANGUAGE_NAMES: dict[CallerLanguage, str] = {"en": "English", "ar": "Arabic"}
SUMMARY_UNAVAILABLE = "Summary unavailable; see the transcript."
SUMMARY_PROMPT = (
    "Summarise this support call for the customer's CRM record in at most two short English "
    "sentences: what the caller needed and what was done. The transcript is already masked; "
    "keep placeholders such as [email] as they are.\n\nTranscript:\n{transcript}"
)


@dataclass(frozen=True)
class CallRecord:
    started_at: datetime
    ended_at: datetime
    languages: list[CallerLanguage]
    contact_id: str | None
    ticket_ids: list[str]


def masked_transcript(history: ChatContext) -> list[str]:
    return [
        f"{SPEAKER_LABELS[item.role]}: {mask_pii(item.text_content)}"
        for item in history.items
        if isinstance(item, ChatMessage) and item.role in SPEAKER_LABELS and item.text_content
    ]


def describe_languages(languages: list[CallerLanguage]) -> str:
    return " and ".join(LANGUAGE_NAMES[language] for language in languages) or "Unknown"


def call_properties(record: CallRecord, summary: str, transcript: list[str]) -> dict[str, str]:
    duration_ms = int((record.ended_at - record.started_at).total_seconds() * 1000)
    language = describe_languages(record.languages)
    body = (
        f"<p><strong>Language:</strong> {html.escape(language)}</p>"
        f"<p><strong>Summary:</strong> {html.escape(summary)}</p>"
        "<p><strong>Transcript (PII masked):</strong><br>"
        + "<br>".join(html.escape(line) for line in transcript)
        + "</p>"
    )
    return {
        "hs_timestamp": str(int(record.started_at.timestamp() * 1000)),
        "hs_call_title": f"Voice support call ({language})",
        "hs_call_body": body,
        "hs_call_duration": str(duration_ms),
        "hs_call_direction": "INBOUND",
        "hs_call_status": "COMPLETED",
    }


def build_call_summariser(settings: Settings) -> CallSummariser:
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    async def summarise(transcript: str) -> str:
        response = await client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.call_summary_max_tokens,
            thinking={"type": settings.llm_thinking},
            output_config={"effort": settings.llm_effort},
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(transcript=transcript)}],
        )
        return "".join(block.text for block in response.content if block.type == "text").strip()

    return summarise


async def log_call(
    crm: HubSpotClient, record: CallRecord, history: ChatContext, summarise: CallSummariser
) -> str:
    transcript = masked_transcript(history)
    # The summary is written from the masked transcript, so no PII leaves for summarising.
    try:
        summary = mask_pii(await summarise("\n".join(transcript)))
    except anthropic.APIError as error:
        logger.error("Call summary failed, logging the call without it: %s", error)
        summary = SUMMARY_UNAVAILABLE
    return await crm.create_call(
        properties=call_properties(record, summary, transcript),
        contact_id=record.contact_id,
        ticket_ids=record.ticket_ids,
    )
