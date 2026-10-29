from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    livekit_url: str
    livekit_api_key: SecretStr
    livekit_api_secret: SecretStr

    eleven_api_key: SecretStr
    elevenlabs_base_url: str = "https://api.elevenlabs.io/v1"
    stt_model: str = "scribe_v2_realtime"
    stt_sample_rate_hz: int = 16000
    tts_model: str = "eleven_flash_v2_5"
    # "Sarah": a calm premade voice that ElevenLabs verifies for both Arabic and English.
    tts_voice_id: str = "EXAVITQu4vr4xnSDxMaL"
    # Per-language voice overrides, e.g. {"ar": "<voice id>"}; unset languages use tts_voice_id.
    tts_voice_id_by_language: dict[str, str] = {}

    # Unset means ElevenLabs detects the language of every committed transcript.
    stt_language: str | None = None
    stt_secondary_languages: list[str] = []
    # Share of a transcript's letters that must be in one script for the script to decide
    # the caller's language; mixed transcripts fall back to the STT's detected language.
    script_dominance_threshold: float = 0.8

    anthropic_api_key: SecretStr
    llm_model: str = "claude-haiku-5-5"
    # Thinking adds seconds before the first token, which a caller hears as dead air.
    llm_thinking: Literal["disabled", "adaptive"] = "disabled"
    llm_effort: Literal["low", "medium", "high"] = "low"

    turn_detector_version: Literal["v1", "v1-mini"] = "v1"
    endpointing_min_delay_seconds: float = 0.5
    endpointing_max_delay_seconds: float = 3.0
    # "adaptive" is LiveKit's cloud model that ignores backchannels; "vad" is local and free.
    interruption_mode: Literal["vad", "adaptive"] = "vad"
    interruption_min_duration_seconds: float = 0.5
    preemptive_generation_enabled: bool = True
    vad_min_silence_duration_seconds: float = 0.55
    latency_budget_seconds: float = 1.0
    worker_initialize_timeout_seconds: float = 60.0
    # One warm process means a call never waits for the embedding model to load.
    worker_idle_processes: int = 1

    hubspot_access_token: SecretStr
    hubspot_base_url: str = "https://api-eu1.hubapi.com"

    provider_timeout_seconds: float = 10.0

    test_audio_directory: Path = Path("evals/audio")
    # "Brian": a different voice from the agent's, also verified for Arabic and English.
    test_caller_voice_id: str = "nPczCjzI2devNBz1zQrb"

    database_url: str = "postgresql://voice_agent:voice_agent@localhost:5434/voice_agent"
    db_pool_min_size: int = 1
    db_pool_max_size: int = 2
    # A caller waits in silence for this long if the database is down, so it stays short.
    db_connection_timeout_seconds: float = 2.0
    kb_directory: Path = Path("data/kb")
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dimensions: int = 384
    kb_chunk_max_characters: int = 1200
    kb_top_k: int = 3
    kb_min_similarity: float = 0.72
    kb_rerank_enabled: bool = False
    kb_rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    kb_rerank_candidate_count: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
