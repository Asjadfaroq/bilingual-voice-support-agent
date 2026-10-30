from collections import deque
from dataclasses import dataclass

import numpy as np
from livekit import rtc
from livekit.agents import stt
from livekit.agents.utils import aio

AudioInput = rtc.AudioFrame | stt.RecognizeStream._FlushSentinel


@dataclass(frozen=True)
class SpeechInputOptions:
    speech_rms_level: float
    preroll_seconds: float
    max_replay_seconds: float


def seconds_of(items: deque[AudioInput]) -> float:
    return sum(item.duration for item in items if isinstance(item, rtc.AudioFrame))


class SpeechGatedReplayChannel(aio.Chan[AudioInput]):
    """The STT's input channel. It holds audio back until the caller first speaks, and it
    remembers what it handed out since the last commit the server confirmed, so the audio of a
    connection that dropped is sent again on the next one."""

    def __init__(self, options: SpeechInputOptions) -> None:
        super().__init__()
        self._options = options
        self._unconfirmed: deque[AudioInput] = deque()
        self._replay: deque[AudioInput] = deque()
        self._preroll: deque[AudioInput] = deque()
        self._speech_started = False

    async def recv(self) -> AudioInput:
        item = self._replay.popleft() if self._replay else await self._next_once_speech_started()
        self._unconfirmed.append(item)
        while self.unconfirmed_seconds() > self._options.max_replay_seconds:
            self._unconfirmed.popleft()
        return item

    async def _next_once_speech_started(self) -> AudioInput:
        # A Scribe session whose first ~10 s held no speech committed the next turns empty or cut
        # short for about 40 s (measured, see docs/DECISIONS.md), and a call starts with the
        # greeting, so nothing before the caller's first words is sent but a short pre-roll.
        # A flush also opens the gate: the VAD heard speech, even if it was too quiet here.
        while not self._speech_started:
            item = await super().recv()
            if self._is_silent(item):
                self._preroll.append(item)
                while seconds_of(self._preroll) > self._options.preroll_seconds:
                    self._preroll.popleft()
                continue
            self._speech_started = True
            self._replay = deque([*self._preroll, item])
            self._preroll = deque()
            return self._replay.popleft()
        return await super().recv()

    def unconfirmed_seconds(self) -> float:
        return seconds_of(self._unconfirmed)

    def _is_silent(self, item: AudioInput) -> bool:
        if not isinstance(item, rtc.AudioFrame):
            return False
        samples = np.frombuffer(item.data, dtype=np.int16).astype(np.float64)
        return (
            samples.size == 0
            or float(np.sqrt(np.mean(samples**2))) < self._options.speech_rms_level
        )

    def replay_unconfirmed(self) -> float:
        # Scribe works through audio at about real time and commits what it has heard so far, so
        # replaying the silence since the last commit left a backlog that cut the next commits
        # short and spilled words into later turns; only the speech is sent again.
        while self._unconfirmed and self._is_silent(self._unconfirmed[0]):
            self._unconfirmed.popleft()
        replayed = self.unconfirmed_seconds()
        self._replay = self._unconfirmed + self._replay
        self._unconfirmed = deque()
        return replayed

    def confirm_commit(self) -> None:
        while self._unconfirmed:
            if not isinstance(self._unconfirmed.popleft(), rtc.AudioFrame):
                return
