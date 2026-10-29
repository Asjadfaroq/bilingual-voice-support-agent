import asyncio
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx

from voice_agent.config import Settings, get_settings

MANIFEST_NAME = "utterances.jsonl"
BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class TestUtterance:
    id: str
    language: str
    text: str


def read_utterances(audio_directory: Path) -> list[TestUtterance]:
    lines = (audio_directory / MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
    return [TestUtterance(**json.loads(line)) for line in lines if line.strip()]


async def synthesise_pcm(client: httpx.AsyncClient, text: str, settings: Settings) -> bytes:
    response = await client.post(
        f"{settings.elevenlabs_base_url}/text-to-speech/{settings.test_caller_voice_id}",
        params={"output_format": f"pcm_{settings.stt_sample_rate_hz}"},
        headers={"xi-api-key": settings.eleven_api_key.get_secret_value()},
        json={"text": text, "model_id": settings.tts_model},
    )
    if response.status_code != httpx.codes.OK:
        raise RuntimeError(
            f"ElevenLabs TTS failed for {text!r}: HTTP {response.status_code} {response.text}"
        )
    return response.content


def write_wav(path: Path, pcm: bytes, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm)


async def make_missing_test_audio() -> int:
    settings = get_settings()
    characters_sent = 0
    async with httpx.AsyncClient(timeout=settings.provider_timeout_seconds) as client:
        for utterance in read_utterances(settings.test_audio_directory):
            path = settings.test_audio_directory / f"{utterance.id}.wav"
            if path.exists():
                continue
            pcm = await synthesise_pcm(client, utterance.text, settings)
            write_wav(path, pcm, settings.stt_sample_rate_hz)
            characters_sent += len(utterance.text)
            print(f"wrote {path}")
    return characters_sent


if __name__ == "__main__":
    print(f"ElevenLabs characters sent: {asyncio.run(make_missing_test_audio())}")
