import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voice_agent.config import Settings

DEMO_AUDIO_DIRECTORY = Path("evals/audio/demo")
SCRIPTS_PATH = DEMO_AUDIO_DIRECTORY / "scripts.json"
PART_GAP_SECONDS = 0.25


@dataclass(frozen=True)
class SpokenPart:
    text: str
    # Another voice for this part, such as an email address said the English way in Arabic.
    voice: str | None = None


@dataclass(frozen=True)
class CallerLine:
    clip: str
    # Spoken with macOS `say` the first time; no parts means a recording from the latency bench.
    parts: tuple[SpokenPart, ...] = ()


@dataclass(frozen=True)
class DemoScript:
    voice: str
    lines: tuple[CallerLine, ...]


def load_script(language: str) -> DemoScript:
    scripts = json.loads(SCRIPTS_PATH.read_text(encoding="utf-8"))
    script = scripts[language]
    lines = tuple(
        CallerLine(line["clip"], tuple(SpokenPart(**part) for part in line.get("parts", [])))
        for line in script["lines"]
    )
    return DemoScript(script["voice"], lines)


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path)) as wav_file:
        return np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)


def spoken_pcm(part: SpokenPart, voice: str, rate_hz: int, folder: Path) -> np.ndarray:
    path = folder / "part.wav"
    audio_format = f"--data-format=LEI16@{rate_hz}"
    command = ["say", "-v", part.voice or voice, "-o", str(path), "--file-format=WAVE"]
    subprocess.run([*command, audio_format, part.text], check=True)
    return read_pcm(path)


def line_audio(line: CallerLine, voice: str, settings: Settings) -> np.ndarray:
    if not line.parts:
        return read_pcm(settings.test_audio_directory / f"{line.clip}.wav")
    path, rate_hz = DEMO_AUDIO_DIRECTORY / f"{line.clip}.wav", settings.stt_sample_rate_hz
    if not path.exists():
        gap = np.zeros(int(rate_hz * PART_GAP_SECONDS), dtype=np.int16)
        with tempfile.TemporaryDirectory() as folder:
            pieces = [spoken_pcm(part, voice, rate_hz, Path(folder)) for part in line.parts]
        pcm = np.concatenate([piece for each in pieces for piece in (each, gap)][:-1])
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(pcm.itemsize)
            wav_file.setframerate(rate_hz)
            wav_file.writeframes(pcm.tobytes())
    return read_pcm(path)
