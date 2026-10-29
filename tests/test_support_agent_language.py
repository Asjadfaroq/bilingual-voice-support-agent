from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from livekit.agents import Agent, ModelSettings, UserInputTranscribedEvent
from livekit.agents.language import LanguageCode
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import elevenlabs

from voice_agent.conversation.call_state import CallState
from voice_agent.conversation.instructions import (
    CALLER_IDENTIFIED_NOTE,
    REPLY_LANGUAGE_NOTES,
)
from voice_agent.conversation.support_agent import LanguageSettings, SupportAgent
from voice_agent.crm.hubspot_client import HubSpotContact


class RecordingTTS:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update_options(self, **options: Any) -> None:
        self.updates.append(options)


class ScriptedModel:
    """Stands in for the LLM: records each generation's context and streams scripted text."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies) or ["Sure."]
        self.contexts: list[ChatContext] = []

    async def generate(
        self, agent: Agent, chat_ctx: ChatContext, *args: Any
    ) -> AsyncGenerator[str, None]:
        self.contexts.append(chat_ctx)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        for word in reply.split(" "):
            yield word + " "


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> ScriptedModel:
    scripted = ScriptedModel()
    monkeypatch.setattr(Agent.default, "llm_node", scripted.generate)
    return scripted


def make_agent(tts: RecordingTTS | None, call: CallState | None = None) -> SupportAgent:
    settings = LanguageSettings("sarah", {"ar": "arabic-voice"}, 0.8, 6)
    return SupportAgent(settings, cast(elevenlabs.TTS, tts), call or CallState())


async def reply_to(agent: SupportAgent, history: ChatContext, caller_text: str) -> str:
    history.add_message(role="user", content=caller_text)
    return "".join([chunk async for chunk in agent.llm_node(history, [], ModelSettings())]).strip()


def system_notes(chat_ctx: ChatContext) -> list[str]:
    return [
        item.text_content or ""
        for item in chat_ctx.items
        if isinstance(item, ChatMessage) and item.role == "system"
    ]


async def test_arabic_turn_switches_the_voice_and_asks_for_an_arabic_reply(
    model: ScriptedModel,
) -> None:
    model.replies = ["باقة النمو تكلف مئة وتسعة وأربعين ريالًا."]
    tts = RecordingTTS()

    await reply_to(make_agent(tts), ChatContext.empty(), "كم سعر باقة النمو؟")

    assert system_notes(model.contexts[-1]) == [REPLY_LANGUAGE_NOTES["ar"]]
    assert tts.updates == [{"language": "ar", "voice_id": "arabic-voice"}]


async def test_reply_language_note_never_enters_the_call_history(model: ScriptedModel) -> None:
    history = ChatContext.empty()

    await reply_to(make_agent(RecordingTTS()), history, "How much is the Growth plan?")

    assert system_notes(history) == []


async def test_switching_back_to_english_mid_call_switches_the_voice_and_call_language_back(
    model: ScriptedModel,
) -> None:
    tts, call = RecordingTTS(), CallState()
    agent, history = make_agent(tts, call), ChatContext.empty()
    model.replies = [
        "Growth is one hundred and forty-nine riyals.",
        "نعم، ندعم مدى.",
        "Starter is forty-nine riyals.",
    ]
    await reply_to(agent, history, "How much is the Growth plan?")
    await reply_to(agent, history, "هل تدعمون مدى؟")

    await reply_to(agent, history, "Thanks, and the Starter plan?")

    assert [update["language"] for update in tts.updates] == ["en", "ar", "en"]
    assert call.language == "en"


async def test_stt_language_decides_a_mixed_script_turn_it_heard(model: ScriptedModel) -> None:
    model.replies = ["لحظة، أتحقق لك."]
    tts = RecordingTTS()
    agent = make_agent(tts)
    mixed = "أبغى أعرف عن Growth plan"
    agent._remember_stt_language(
        UserInputTranscribedEvent(transcript=mixed, is_final=True, language=LanguageCode("ar"))
    )

    await reply_to(agent, ChatContext.empty(), mixed)

    assert tts.updates == [{"language": "ar", "voice_id": "arabic-voice"}]


async def test_a_caller_identified_from_their_line_is_named_to_every_generation(
    model: ScriptedModel,
) -> None:
    call = CallState(caller_phone="+966551234567", contact=HubSpotContact("101", "Sara"))

    await reply_to(make_agent(None, call), ChatContext.empty(), "What's my ticket status?")

    assert CALLER_IDENTIFIED_NOTE.format(first_name="Sara") in system_notes(model.contexts[-1])
