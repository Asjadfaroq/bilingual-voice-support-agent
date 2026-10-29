from typing import Any, cast

import pytest
from livekit.agents import Agent, ModelSettings, UserInputTranscribedEvent
from livekit.agents.language import LanguageCode
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import elevenlabs

from voice_agent.conversation.instructions import REPLY_LANGUAGE_NOTES
from voice_agent.conversation.support_agent import LanguageSettings, SupportAgent


class RecordingTTS:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update_options(self, **options: Any) -> None:
        self.updates.append(options)


@pytest.fixture
def generation_contexts(monkeypatch: pytest.MonkeyPatch) -> list[ChatContext]:
    captured: list[ChatContext] = []

    def capture_generation(agent: Agent, chat_ctx: ChatContext, *args: Any) -> None:
        captured.append(chat_ctx)

    monkeypatch.setattr(Agent.default, "llm_node", capture_generation)
    return captured


def make_agent(tts: RecordingTTS | None) -> SupportAgent:
    settings = LanguageSettings(
        default_voice_id="sarah",
        voice_id_by_language={"ar": "arabic-voice"},
        script_dominance_threshold=0.8,
    )
    return SupportAgent(settings, reply_tts=cast(elevenlabs.TTS, tts))


def generate_after(agent: SupportAgent, history: ChatContext, caller_text: str) -> None:
    history.add_message(role="user", content=caller_text)
    agent.llm_node(history, [], ModelSettings())


def system_notes(chat_ctx: ChatContext) -> list[str]:
    return [
        item.text_content or ""
        for item in chat_ctx.items
        if isinstance(item, ChatMessage) and item.role == "system"
    ]


def test_arabic_turn_switches_the_voice_and_asks_for_an_arabic_reply(
    generation_contexts: list[ChatContext],
) -> None:
    tts = RecordingTTS()
    agent = make_agent(tts)

    generate_after(agent, ChatContext.empty(), "كم سعر باقة النمو؟")

    assert system_notes(generation_contexts[-1]) == [REPLY_LANGUAGE_NOTES["ar"]]
    assert tts.updates == [{"language": "ar", "voice_id": "arabic-voice"}]


def test_reply_language_note_never_enters_the_call_history(
    generation_contexts: list[ChatContext],
) -> None:
    history = ChatContext.empty()

    generate_after(make_agent(RecordingTTS()), history, "How much is the Growth plan?")

    assert system_notes(history) == []


def test_every_generation_in_a_turn_gets_the_note_but_the_voice_switches_once(
    generation_contexts: list[ChatContext],
) -> None:
    tts = RecordingTTS()
    agent = make_agent(tts)
    history = ChatContext.empty()
    generate_after(agent, history, "هل تدعمون مدى؟")

    agent.llm_node(history, [], ModelSettings())

    assert [system_notes(ctx) for ctx in generation_contexts] == [[REPLY_LANGUAGE_NOTES["ar"]]] * 2
    assert len(tts.updates) == 1


def test_switching_back_to_english_mid_call_switches_the_voice_back(
    generation_contexts: list[ChatContext],
) -> None:
    tts = RecordingTTS()
    agent = make_agent(tts)
    history = ChatContext.empty()
    generate_after(agent, history, "How much is the Growth plan?")
    generate_after(agent, history, "هل تدعمون مدى؟")

    generate_after(agent, history, "Thanks, and the Starter plan?")

    assert system_notes(generation_contexts[-1]) == [REPLY_LANGUAGE_NOTES["en"]]
    assert [update["language"] for update in tts.updates] == ["en", "ar", "en"]
    assert tts.updates[-1]["voice_id"] == "sarah"


def test_stt_language_decides_a_mixed_script_turn_it_heard(
    generation_contexts: list[ChatContext],
) -> None:
    tts = RecordingTTS()
    agent = make_agent(tts)
    mixed = "أبغى أعرف عن Growth plan"
    agent._remember_stt_language(
        UserInputTranscribedEvent(transcript=mixed, is_final=True, language=LanguageCode("ar"))
    )

    generate_after(agent, ChatContext.empty(), mixed)

    assert tts.updates == [{"language": "ar", "voice_id": "arabic-voice"}]


def test_turn_without_letters_keeps_the_callers_current_language(
    generation_contexts: list[ChatContext],
) -> None:
    agent = make_agent(RecordingTTS())
    history = ChatContext.empty()
    generate_after(agent, history, "هل تدعمون مدى؟")

    generate_after(agent, history, "4412")

    assert system_notes(generation_contexts[-1]) == [REPLY_LANGUAGE_NOTES["ar"]]


def test_text_mode_without_a_voice_still_asks_for_the_callers_language(
    generation_contexts: list[ChatContext],
) -> None:
    generate_after(make_agent(None), ChatContext.empty(), "مرحبا")

    assert system_notes(generation_contexts[-1]) == [REPLY_LANGUAGE_NOTES["ar"]]
