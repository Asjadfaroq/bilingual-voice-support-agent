from typing import Any

from livekit.agents import (
    NOT_GIVEN,
    NotGivenOr,
    TurnHandlingOptions,
    inference,
    vad,
)
from livekit.agents.llm import LLMStream
from livekit.agents.utils import is_given
from livekit.plugins import anthropic, elevenlabs, silero

from voice_agent.config import Settings
from voice_agent.pipeline.stt_input import SpeechInputOptions
from voice_agent.pipeline.stt_stream import BilingualElevenLabsSTT


class ReasoningControlledLLM(anthropic.LLM):
    """The plugin only forwards thinking and effort per call, so they are attached here."""

    def __init__(self, *, reasoning_options: dict[str, Any], **plugin_options: Any) -> None:
        super().__init__(**plugin_options)
        self._reasoning_options = reasoning_options

    def chat(
        self, *, extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN, **kwargs: Any
    ) -> LLMStream:
        caller_options = extra_kwargs if is_given(extra_kwargs) else {}
        return super().chat(extra_kwargs={**self._reasoning_options, **caller_options}, **kwargs)


def build_stt(settings: Settings) -> BilingualElevenLabsSTT:
    return BilingualElevenLabsSTT(
        liveness_timeout_seconds=settings.stt_liveness_timeout_seconds,
        input_options=SpeechInputOptions(
            speech_rms_level=settings.stt_speech_rms_level,
            preroll_seconds=settings.stt_speech_preroll_seconds,
            max_replay_seconds=settings.stt_max_replay_seconds,
        ),
        api_key=settings.eleven_api_key.get_secret_value(),
        model=settings.stt_model,
        language_code=settings.stt_language or NOT_GIVEN,
        secondary_languages=settings.stt_secondary_languages or NOT_GIVEN,
        include_language_detection=True,
        sample_rate=settings.stt_sample_rate_hz,
    )


def build_llm(settings: Settings) -> ReasoningControlledLLM:
    return ReasoningControlledLLM(
        reasoning_options={
            "thinking": {"type": settings.llm_thinking},
            "output_config": {"effort": settings.llm_effort},
        },
        api_key=settings.anthropic_api_key.get_secret_value(),
        model=settings.llm_model,
        # The CRM tools depend on each other (a ticket needs the contact saved first); run in
        # parallel, a ticket was proposed before its contact existed and the caller was told
        # it had failed.
        parallel_tool_calls=False,
    )


def build_tts(settings: Settings) -> elevenlabs.TTS:
    return elevenlabs.TTS(
        api_key=settings.eleven_api_key.get_secret_value(),
        model=settings.tts_model,
        voice_id=settings.tts_voice_id,
    )


def build_vad(settings: Settings) -> vad.VAD:
    return silero.VAD.load(min_silence_duration=settings.vad_min_silence_duration_seconds)


def build_turn_handling(settings: Settings) -> TurnHandlingOptions:
    return TurnHandlingOptions(
        turn_detection=inference.TurnDetector(
            version=settings.turn_detector_version,
            api_key=settings.livekit_api_key.get_secret_value(),
            api_secret=settings.livekit_api_secret.get_secret_value(),
        ),
        endpointing={
            "min_delay": settings.endpointing_min_delay_seconds,
            "max_delay": settings.endpointing_max_delay_seconds,
        },
        interruption={
            "enabled": True,
            "mode": settings.interruption_mode,
            "min_duration": settings.interruption_min_duration_seconds,
        },
        preemptive_generation={"enabled": settings.preemptive_generation_enabled},
    )
