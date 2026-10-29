from typing import Any

import pytest
from livekit.agents.llm import ChatContext
from livekit.plugins import anthropic

from voice_agent.config import Settings
from voice_agent.pipeline.components import build_llm, build_stt, build_tts, build_turn_handling


def test_providers_use_the_models_and_voice_from_config(test_settings: Settings) -> None:
    assert build_stt(test_settings).model == test_settings.stt_model
    assert build_llm(test_settings).model == test_settings.llm_model
    tts = build_tts(test_settings)
    assert tts.model == test_settings.tts_model
    assert tts._opts.voice_id == test_settings.tts_voice_id


def test_every_llm_request_carries_the_configured_thinking_and_effort(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    forwarded: dict[str, Any] = {}
    monkeypatch.setattr(anthropic.LLM, "chat", lambda self, **kwargs: forwarded.update(kwargs))

    build_llm(test_settings).chat(chat_ctx=ChatContext.empty())

    assert forwarded["extra_kwargs"] == {
        "thinking": {"type": "disabled"},
        "output_config": {"effort": "low"},
    }


def test_per_call_options_are_kept_alongside_the_reasoning_options(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    forwarded: dict[str, Any] = {}
    monkeypatch.setattr(anthropic.LLM, "chat", lambda self, **kwargs: forwarded.update(kwargs))

    build_llm(test_settings).chat(chat_ctx=ChatContext.empty(), extra_kwargs={"max_tokens": 64})

    assert forwarded["extra_kwargs"]["max_tokens"] == 64
    assert forwarded["extra_kwargs"]["thinking"] == {"type": "disabled"}


def test_barge_in_is_enabled_with_the_configured_mode_and_minimum_speech(
    test_settings: Settings,
) -> None:
    turn_handling = build_turn_handling(test_settings)

    assert turn_handling["interruption"] == {
        "enabled": True,
        "mode": test_settings.interruption_mode,
        "min_duration": test_settings.interruption_min_duration_seconds,
    }
    assert turn_handling["endpointing"]["min_delay"] == test_settings.endpointing_min_delay_seconds
