from collections.abc import AsyncGenerator, AsyncIterable

from livekit import rtc
from livekit.agents import Agent, ModelSettings, UserInputTranscribedEvent
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm import ChatContext, Tool
from livekit.plugins import elevenlabs
from opentelemetry import trace

from voice_agent.conversation import knowledge_first
from voice_agent.conversation.call_state import CallState, caller_identity_note
from voice_agent.conversation.confirmation import caller_messages
from voice_agent.conversation.instructions import (
    GREETING,
    REPLY_LANGUAGE_NOTES,
    WRONG_LANGUAGE_RETRY_NOTES,
)
from voice_agent.conversation.reply_filter import (
    ReplyChunk,
    ReportHold,
    guarded_reply,
    ignore_hold,
    spoken_sentences,
)
from voice_agent.conversation.system_notes import with_system_notes
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
    LanguageSettings,
    detect_caller_language,
    prose_language,
    reply_voice_for,
)

CRM_TOOLS = [
    find_or_create_contact,
    create_ticket,
    get_ticket_status,
    request_human_callback,
    record_caller_answer,
]


class SupportAgent(Agent):
    def __init__(
        self,
        language_settings: LanguageSettings,
        reply_tts: elevenlabs.TTS | None,
        call: CallState,
        *,
        kb_search_mode: knowledge_first.KnowledgeSearchMode,
        report_reply_filter_hold: ReportHold = ignore_hold,
    ) -> None:
        super().__init__(
            instructions=knowledge_first.SUPPORT_INSTRUCTIONS[kb_search_mode],
            tools=[
                *([search_knowledge_base] if kb_search_mode == "tool" else []),
                *CRM_TOOLS,
                # The agent says goodbye before calling end_call, so the tool asks for no reply.
                EndCallTool(end_instructions=None, ignore_on_enter=True),
            ],
        )
        self._language_settings = language_settings
        self._call = call
        self._kb_search_mode = kb_search_mode
        self._report_reply_filter_hold = report_reply_filter_hold
        # None in text mode, where there is no voice to switch.
        self._reply_tts = reply_tts
        self._caller_language: CallerLanguage | None = None
        self._latest_final_transcript = ""
        self._latest_final_stt_language: str | None = None

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
        if language not in self._call.languages:
            self._call.languages.append(language)
        if self._reply_tts is None:
            return
        voice = reply_voice_for(language, self._language_settings)
        self._reply_tts.update_options(language=voice.language, voice_id=voice.voice_id)

    def llm_node(
        self, chat_ctx: ChatContext, tools: list[Tool], model_settings: ModelSettings
    ) -> AsyncGenerator[ReplyChunk, None]:
        # Every generation, spoken or typed, before or after a tool call, gets the reply
        # language and who the caller is; the notes live only in this generation's context.
        messages = caller_messages(chat_ctx)
        caller_text = (messages[-1].text_content or "") if messages else ""
        language = self._language_of(caller_text) if messages else None
        searches = knowledge_first.searches_before_reply(chat_ctx, self._call)
        notes = [note] if (note := caller_identity_note(self._call)) else []
        notes += knowledge_first.earlier_knowledge_notes(self._call, self._kb_search_mode)
        if language is not None:
            trace.get_current_span().set_attribute("voice_agent.reply_language", language)
            self._switch_voice_to(language)
            notes.append(REPLY_LANGUAGE_NOTES[language])
        chat_ctx = with_system_notes(chat_ctx, *notes)

        def generate(context: ChatContext) -> AsyncGenerator[ReplyChunk, None]:
            return Agent.default.llm_node(self, context, tools, model_settings)

        def start_reply(context: ChatContext) -> AsyncGenerator[ReplyChunk, None]:
            if language is None:
                return generate(context)
            retry_note = WRONG_LANGUAGE_RETRY_NOTES[language]
            min_letters = self._language_settings.reply_language_min_letters
            report_hold = self._report_reply_filter_hold
            return guarded_reply(generate, context, language, retry_note, min_letters, report_hold)

        if self._kb_search_mode == "every_turn" and caller_text and searches:
            resources = self.session.userdata
            return knowledge_first.reply_with_knowledge(
                start_reply, chat_ctx, caller_text, resources
            )
        return start_reply(chat_ctx)

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
