import asyncio
import json
import os
import statistics
import sys
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
from livekit import api

from voice_agent.config import Settings

LATENCY_TOPIC = "turn_latency"
WORKER_READY_LINE = "registered worker"
# The worker loads the embedding model and VAD before it registers.
WORKER_REGISTER_TIMEOUT_SECONDS = 120.0
POLL_SECONDS = 0.05


def read_manifest(settings: Settings, language: str) -> list[dict[str, Any]]:
    manifest = settings.test_audio_directory / "utterances.jsonl"
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    return [entry for entry in entries if entry["language"] == language]


def load_utterances(settings: Settings, language: str) -> list[tuple[str, np.ndarray]]:
    utterances = []
    for entry in read_manifest(settings, language):
        with wave.open(str(settings.test_audio_directory / f"{entry['id']}.wav")) as wav_file:
            if wav_file.getframerate() != settings.stt_sample_rate_hz:
                raise ValueError(f"{entry['id']}.wav is not {settings.stt_sample_rate_hz} Hz")
            pcm = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        utterances.append((entry["id"], pcm))
    return utterances


async def elevenlabs_characters_left(settings: Settings) -> int:
    async with httpx.AsyncClient(timeout=settings.provider_timeout_seconds) as client:
        response = await client.get(
            f"{settings.elevenlabs_base_url}/user/subscription",
            headers={"xi-api-key": settings.eleven_api_key.get_secret_value()},
        )
    response.raise_for_status()
    subscription = response.json()
    return subscription["character_limit"] - subscription["character_count"]


async def wait_until(condition: Callable[[], bool], timeout_seconds: float) -> bool:
    deadline = time.perf_counter() + timeout_seconds
    while not condition():
        if time.perf_counter() > deadline:
            return False
        await asyncio.sleep(POLL_SECONDS)
    return True


async def start_worker(
    overrides: dict[str, str], agent_name: str, log_path: Path, log_level: str
) -> asyncio.subprocess.Process:
    environment = {**os.environ, **overrides}
    environment |= {
        "AGENT_NAME": agent_name,
        "TURN_LATENCY_TOPIC": LATENCY_TOPIC,
        "CALL_LOGGING_ENABLED": "false",
    }
    command = [sys.executable, "-m", "livekit.agents", "start", "src/voice_agent/worker.py"]
    with log_path.open("w") as log_file:
        worker = await asyncio.create_subprocess_exec(
            *command,
            "--dev",
            "--log-level",
            log_level,
            env=environment,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
        )
    registered = await wait_until(
        lambda: WORKER_READY_LINE in log_path.read_text(), WORKER_REGISTER_TIMEOUT_SECONDS
    )
    if not registered:
        if worker.returncode is None:
            worker.kill()
        raise RuntimeError(f"The bench worker did not register; see {log_path}")
    return worker


async def livekit_round_trip_ms(settings: Settings, samples: int = 5) -> float:
    # The median TCP connect time to LiveKit Cloud, so each run records the network it ran on.
    host = urlparse(settings.livekit_url).hostname or ""
    times = []
    for _ in range(samples):
        started = time.perf_counter()
        _, writer = await asyncio.open_connection(host, 443)
        times.append((time.perf_counter() - started) * 1000)
        writer.close()
        await writer.wait_closed()
    return round(statistics.median(times), 1)


def caller_token(settings: Settings, room_name: str, identity: str, agent_name: str) -> str:
    # The room asks for the bench's own worker by name, so no other running worker joins it.
    dispatch = api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=agent_name)])
    return (
        api.AccessToken(
            settings.livekit_api_key.get_secret_value(),
            settings.livekit_api_secret.get_secret_value(),
        )
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .with_room_config(dispatch)
        .to_jwt()
    )
