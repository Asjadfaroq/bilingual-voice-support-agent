from collections.abc import AsyncGenerator
from dataclasses import dataclass

from livekit.agents import Agent, FlushSentinel, ModelSettings, UserInputTranscribedEvent
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm import ChatChunk, ChatContext, ChatMessage, Tool
from livekit.plugins import elevenlabs

from voice_agent.conversation.instructions import (
    GREETING,
    REPLY_LANGUAGE_NOTES,
    SUPPORT_INSTRUCTIONS,
)
from voice_agent.conversation.ticket_tools import (
    create_ticket,
    get_ticket_status,
    request_human_callback,
)
from voice_agent.conversation.tools import (
    find_or_create_contact,
    save_confirmed_request,
    search_knowledge_base,
)
from voice_agent.pipeline.language import (
    CallerLanguage,
    choose_reply_voice,
    detect_caller_language,
)


@dataclass(frozen=True)
class LanguageSettings:
    default_voice_id: str
    voice_id_by_language: dict[str, str]
    script_dominance_threshold: float


def latest_caller_text(chat_ctx: ChatContext) -> str | None:
    for item in reversed(chat_ctx.items):
        if isinstance(item, ChatMessage) and item.role == "user":
            return item.text_content or ""
    return None


class SupportAgent(Agent):
    def __init__(
        self, language_settings: LanguageSettings, reply_tts: elevenlabs.TTS | None
    ) -> None:
        super().__init__(
            instructions=SUPPORT_INSTRUCTIONS,
            tools=[
                search_knowledge_base,
                find_or_create_contact,
                create_ticket,
                get_ticket_status,
                request_human_callback,
                save_confirmed_request,
                # The agent says goodbye before calling end_call, so the tool asks for no reply.
                EndCallTool(end_instructions=None, ignore_on_enter=True),
            ],
        )
        self._language_settings = language_settings
        # None in text mode, where there is no voice to switch.
        self._reply_tts = reply_tts
        self._caller_language: CallerLanguage | None = None
        self._caller_languages: list[CallerLanguage] = []
        self._latest_final_transcript = ""
        self._latest_final_stt_language: str | None = None

    @property
    def caller_languages(self) -> list[CallerLanguage]:
        return list(self._caller_languages)

    async def on_enter(self) -> None:
        self.session.on("user_input_transcribed", self._remember_stt_language)
        self.session.say(GREETING)

    def _remember_stt_language(self, event: UserInputTranscribedEvent) -> None:
        if event.is_final and event.transcript.strip():
            self._latest_final_transcript = event.transcript.strip()
            self._latest_final_stt_language = event.language

    def _language_of(self, caller_text: str) -> CallerLanguage | None:
        # Typed turns have no STT event, so the STT language only counts for the turn it heard.
        heard_this_turn = bool(self._latest_final_transcript) and caller_text.strip().endswith(
            self._latest_final_transcript
        )
        stt_language = self._latest_final_stt_language if heard_this_turn else None
        detected = detect_caller_language(
            caller_text, stt_language, self._language_settings.script_dominance_threshold
        )
        return detected or self._caller_language

    def _switch_voice_to(self, language: CallerLanguage) -> None:
        if language == self._caller_language:
            return
        self._caller_language = language
        if language not in self._caller_languages:
            self._caller_languages.append(language)
        if self._reply_tts is None:
            return
        voice = choose_reply_voice(
            language,
            self._language_settings.default_voice_id,
            self._language_settings.voice_id_by_language,
        )
        self._reply_tts.update_options(language=voice.language, voice_id=voice.voice_id)

    def llm_node(
        self, chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
    ) -> AsyncGenerator[ChatChunk | str | FlushSentinel, None]:
        # Every generation, spoken or typed, before or after a tool call, gets the reply
        # language; the note lives only in this generation's context, never in the history.
        caller_text = latest_caller_text(chat_ctx)
        language = self._language_of(caller_text) if caller_text is not None else None
        if language is not None:
            self._switch_voice_to(language)
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(role="system", content=REPLY_LANGUAGE_NOTES[language])
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)
