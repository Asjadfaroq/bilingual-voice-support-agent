import dataclasses
import logging
from typing import Any

import aiohttp
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.language import LanguageCode
from livekit.agents.utils import is_given
from livekit.plugins import elevenlabs
from livekit.plugins.elevenlabs import stt as elevenlabs_stt

from voice_agent.pipeline.stt_input import SpeechGatedReplayChannel, SpeechInputOptions

logger = logging.getLogger(__name__)

COMMITTED_TRANSCRIPT_TYPES = ("committed_transcript", "committed_transcript_with_timestamps")


def recover_empty_commit(message: dict[str, Any], pending_partial_text: str) -> dict[str, Any]:
    """Scribe realtime sometimes commits an empty transcript for an Arabic turn that follows
    an English one, although its partial transcript holds the whole sentence; without this
    the caller's turn is lost (measured, see docs/DECISIONS.md)."""
    is_empty_commit = message.get("message_type") in COMMITTED_TRANSCRIPT_TYPES and not message.get(
        "text"
    )
    if is_empty_commit and pending_partial_text:
        return {**message, "text": pending_partial_text}
    return message


def drop_repeated_commit(message: dict[str, Any], previous_final_text: str) -> dict[str, Any]:
    """A commit of silence sometimes comes back with the previous commit's text and no detected
    language (measured in docs/DECISIONS.md); spoken again, it would be a second caller turn."""
    is_repeat = (
        message.get("message_type") in COMMITTED_TRANSCRIPT_TYPES
        and "language_code" in message
        and message["language_code"] is None
        and message.get("text") == previous_final_text
    )
    return {**message, "text": ""} if is_repeat and previous_final_text else message


class ResilientElevenLabsStream(elevenlabs_stt.SpeechStream):
    def __init__(self, *, input_options: SpeechInputOptions, **stream_options: Any) -> None:
        super().__init__(**stream_options)
        self._input_ch = SpeechGatedReplayChannel(input_options)
        self._connections = 0
        self._previous_final_text = ""

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        websocket = await super()._connect_ws()
        self._connections += 1
        if self._connections > 1:
            replayed = self._input_ch.replay_unconfirmed()
            logger.warning(
                "ElevenLabs STT reconnected (connection %d), replaying %.2fs of unconfirmed audio",
                self._connections,
                replayed,
            )
        return websocket

    def _process_stream_event(self, data: dict[str, Any]) -> None:
        # Scribe sends each commit twice; only the copy the plugin treats as final confirms it.
        if data.get("message_type") in COMMITTED_TRANSCRIPT_TYPES:
            logger.debug(
                "Scribe %s: %r (language %s)",
                data["message_type"],
                data.get("text"),
                data.get("language_code"),
            )
        if data.get("message_type") == self._final_message_type:
            self._input_ch.confirm_commit()
            data = drop_repeated_commit(data, self._previous_final_text)
            self._previous_final_text = data.get("text") or self._previous_final_text
        super()._process_stream_event(recover_empty_commit(data, self._last_partial_text))


class BilingualElevenLabsSTT(elevenlabs.STT):
    def __init__(
        self,
        *,
        liveness_timeout_seconds: float,
        input_options: SpeechInputOptions,
        **stt_options: Any,
    ) -> None:
        super().__init__(**stt_options)
        self._liveness_timeout_seconds = liveness_timeout_seconds
        self._input_options = input_options

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> elevenlabs_stt.SpeechStream:
        stream = ResilientElevenLabsStream(
            input_options=self._input_options,
            stt=self,
            opts=self._opts,
            conn_options=dataclasses.replace(
                conn_options, liveness_timeout=self._liveness_timeout_seconds
            ),
            language=LanguageCode(language) if is_given(language) else self._opts.language_code,
            http_session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream
