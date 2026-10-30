import json
from datetime import UTC, datetime, timedelta

import anthropic
import httpx
import respx
from livekit.agents.llm import ChatContext, ChatMessage

from voice_agent.crm.call_log import SUMMARY_UNAVAILABLE, CallRecord, log_call, masked_transcript
from voice_agent.crm.hubspot_client import CALL_TO_TICKET, HubSpotClient

STARTED_AT = datetime(2026, 10, 29, 9, 0, tzinfo=UTC)
RECORD = CallRecord(
    started_at=STARTED_AT,
    ended_at=STARTED_AT + timedelta(minutes=2, seconds=5),
    languages=["ar", "en"],
    contact_id="7",
    ticket_ids=["901"],
)


def call_history() -> ChatContext:
    history = ChatContext.empty()
    history.add_message(role="assistant", content="Hi, Arabic or English is fine.")
    history.add_message(role="user", content="My email is sara@example.com and phone 0551234567")
    history.add_message(role="assistant", content="The ticket is open.")
    return history


async def summary_of(transcript: str) -> str:
    return "Caller sara@example.com asked for a ticket; it was opened."


async def failing_summary(transcript: str) -> str:
    raise anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.test"))


async def test_call_is_logged_with_duration_language_summary_and_masked_transcript(
    hubspot_client: HubSpotClient, hubspot_api: respx.MockRouter
) -> None:
    calls = hubspot_api.post("/crm/v3/objects/calls").mock(
        return_value=httpx.Response(201, json={"id": "55"})
    )

    assert await log_call(hubspot_client, RECORD, call_history(), summary_of) == "55"

    sent = json.loads(calls.calls.last.request.content)
    properties = sent["properties"]
    assert properties["hs_call_duration"] == "125000"
    assert properties["hs_call_title"] == "Voice support call (Arabic and English)"
    assert "Caller: My email is [email] and phone [phone]" in properties["hs_call_body"]
    assert "sara@example.com" not in properties["hs_call_body"]
    assert "0551234567" not in properties["hs_call_body"]
    assert [link["to"]["id"] for link in sent["associations"]] == ["7", "901"]
    assert sent["associations"][1]["types"][0]["associationTypeId"] == CALL_TO_TICKET


async def test_summary_failure_still_logs_the_call(
    hubspot_client: HubSpotClient, hubspot_api: respx.MockRouter
) -> None:
    calls = hubspot_api.post("/crm/v3/objects/calls").mock(
        return_value=httpx.Response(201, json={"id": "55"})
    )

    await log_call(hubspot_client, RECORD, call_history(), failing_summary)

    assert (
        SUMMARY_UNAVAILABLE
        in json.loads(calls.calls.last.request.content)["properties"]["hs_call_body"]
    )


def test_the_transcript_follows_when_each_turn_was_spoken_not_when_it_was_recorded() -> None:
    # A caller who speaks during the greeting is in the history before it, because the
    # greeting is recorded when it finishes playing.
    history = ChatContext.empty()
    history.items.append(ChatMessage(role="user", content=["Hi, I need help"], created_at=11.0))
    history.items.append(
        ChatMessage(role="assistant", content=["Hi, Arabic or English is fine."], created_at=10.0)
    )

    assert masked_transcript(history) == [
        "Agent: Hi, Arabic or English is fine.",
        "Caller: Hi, I need help",
    ]
