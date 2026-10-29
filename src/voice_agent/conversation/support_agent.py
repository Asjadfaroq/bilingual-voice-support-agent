from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import Agent, FlushSentinel, ModelSettings, UserInputTranscribedEvent
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm import ChatChunk, ChatContext, Tool
from livekit.plugins import elevenlabs
from opentelemetry import trace

from voice_agent.conversation.call_state import CallState, caller_identity_note
from voice_agent.conversation.confirmation import caller_messages
from voice_agent.conversation.instructions import (
    GREETING,
    REPLY_LANGUAGE_NOTES,
    SUPPORT_INSTRUCTIONS,
    WRONG_LANGUAGE_RETRY_NOTES,
)
from voice_agent.conversation.reply_filter import guarded_reply, prose_language, spoken_sentences
from voice_agent.conversation.ticket_tools import (
    create_ticket,
    get_ticket_status,
    request_human_callback,
)
from voice_agent.conversation.tools import (
    find_or_create_contact,
    record_caller_answer,
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
    reply_language_min_letters: int


class SupportAgent(Agent):
    def __init__(
        self,
        language_settings: LanguageSettings,
        reply_tts: elevenlabs.TTS | None,
        call: CallState,
    ) -> None:
        super().__init__(
            instructions=SUPPORT_INSTRUCTIONS,
            tools=[
                search_knowledge_base,
                find_or_create_contact,
                create_ticket,
                get_ticket_status,
                request_human_callback,
                record_caller_answer,
                # The agent says goodbye before calling end_call, so the tool asks for no reply.
                EndCallTool(end_instructions=None, ignore_on_enter=True),
            ],
        )
        self._language_settings = language_settings
        self._call = call
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
        self._call.language = language
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
        # language and who the caller is; the notes live only in this generation's context.
        messages = caller_messages(chat_ctx)
        language = self._language_of(messages[-1].text_content or "") if messages else None
        chat_ctx = chat_ctx.copy()
        if note := caller_identity_note(self._call):
            chat_ctx.add_message(role="system", content=note)
        if language is None:
            return Agent.default.llm_node(self, chat_ctx, tools, model_settings)
        trace.get_current_span().set_attribute("voice_agent.reply_language", language)
        self._switch_voice_to(language)
        chat_ctx.add_message(role="system", content=REPLY_LANGUAGE_NOTES[language])
        return guarded_reply(
            lambda context: Agent.default.llm_node(self, context, tools, model_settings),
            chat_ctx,
            language,
            WRONG_LANGUAGE_RETRY_NOTES[language],
            self._language_settings.reply_language_min_letters,
        )

    def _spoken_language(self, sentence: str) -> CallerLanguage:
        min_letters = self._language_settings.reply_language_min_letters
        return self._caller_language or prose_language(sentence, min_letters) or "en"

    def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncGenerator[rtc.AudioFrame, None]:
        # Numbers, priorities and Latin terms become words here, for every spoken sentence:
        # the LLM's replies and the read-backs the tools say themselves.
        spoken = spoken_sentences(text, self._spoken_language)
        return Agent.default.tts_node(self, spoken, model_settings)
