import asyncio
import json
import time
from dataclasses import dataclass, field

import numpy as np
from livekit import rtc

FRAME_SECONDS = 0.02
# A short source queue keeps "the caller stopped speaking" within a frame or two of the truth.
SOURCE_QUEUE_MS = 60
AGENT_STATE_ATTRIBUTE = "lk.agent.state"
# Root-mean-square level of 16-bit audio above which a frame is speech, not silence or noise.
SPEECH_RMS_LEVEL = 300.0
# Any agent audio above this level is sound; between this and SPEECH_RMS_LEVEL it is a quiet start.
SOUND_RMS_LEVEL = 20.0
# Agent audio that starts after at least this much quiet is a new onset (a new speech).
ONSET_GAP_SECONDS = 0.3


@dataclass
class AgentObservations:
    audio_onsets: list[float] = field(default_factory=list)
    sound_onsets: list[float] = field(default_factory=list)
    states: list[tuple[float, str]] = field(default_factory=list)
    measurements: list[tuple[float, dict[str, object]]] = field(default_factory=list)
    last_audible_at: float = 0.0


def frame_rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0


def trailing_quiet_seconds(pcm: np.ndarray, sample_rate_hz: int) -> float:
    loud = np.flatnonzero(np.abs(pcm) > SPEECH_RMS_LEVEL)
    return 0.0 if loud.size == 0 else (pcm.size - 1 - loud[-1]) / sample_rate_hz


class BenchCaller:
    def __init__(self, room: rtc.Room, sample_rate_hz: int, latency_topic: str) -> None:
        self.observations = AgentObservations()
        self._room = room
        self._sample_rate_hz = sample_rate_hz
        self._latency_topic = latency_topic
        self._source = rtc.AudioSource(sample_rate_hz, 1, queue_size_ms=SOURCE_QUEUE_MS)
        # The source takes one frame at a time; the utterance holds this for its whole length.
        self._capture_lock = asyncio.Lock()
        self._last_sound_at = 0.0
        self._tasks: set[asyncio.Task[None]] = set()
        room.on("track_subscribed", self._listen_to_agent)
        room.on("participant_attributes_changed", self._record_agent_state)
        room.on("data_received", self._record_measurement)

    async def start(self) -> None:
        track = rtc.LocalAudioTrack.create_audio_track("caller", self._source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self._room.local_participant.publish_track(track, options)
        self._keep(self._send_silence_between_turns())

    def _keep(self, coroutine: object) -> None:
        task = asyncio.ensure_future(coroutine)  # type: ignore[arg-type]
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _frame(self, samples: np.ndarray) -> rtc.AudioFrame:
        return rtc.AudioFrame(samples.tobytes(), self._sample_rate_hz, 1, samples.size)

    async def _send_silence_between_turns(self) -> None:
        # A microphone never goes quiet on the wire; the STT and VAD expect continuous audio.
        silence = np.zeros(int(self._sample_rate_hz * FRAME_SECONDS), dtype=np.int16)
        while True:
            async with self._capture_lock:
                await self._source.capture_frame(self._frame(silence))

    async def say(self, pcm: np.ndarray) -> float:
        """Streams the utterance in real time and returns when its last speech is heard."""
        frame_samples = int(self._sample_rate_hz * FRAME_SECONDS)
        async with self._capture_lock:
            for start in range(0, pcm.size, frame_samples):
                await self._source.capture_frame(self._frame(pcm[start : start + frame_samples]))
            finished_at = time.time() + self._source.queued_duration
        return finished_at - trailing_quiet_seconds(pcm, self._sample_rate_hz)

    def _listen_to_agent(
        self, track: rtc.Track, _: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO and participant.kind == (
            rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
        ):
            self._keep(self._watch_agent_audio(track))

    async def _watch_agent_audio(self, track: rtc.Track) -> None:
        async for event in rtc.AudioStream(track, sample_rate=self._sample_rate_hz):
            level, now = frame_rms(np.frombuffer(event.frame.data, dtype=np.int16)), time.time()
            if level >= SOUND_RMS_LEVEL and now - self._last_sound_at > ONSET_GAP_SECONDS:
                self.observations.sound_onsets.append(now)
            if level >= SOUND_RMS_LEVEL:
                self._last_sound_at = now
            if level < SPEECH_RMS_LEVEL:
                continue
            if now - self.observations.last_audible_at > ONSET_GAP_SECONDS:
                self.observations.audio_onsets.append(now)
            self.observations.last_audible_at = now

    def _record_agent_state(self, changed: dict[str, str], participant: rtc.Participant) -> None:
        if AGENT_STATE_ATTRIBUTE in changed:
            state = changed[AGENT_STATE_ATTRIBUTE]
            self.observations.states.append((time.time(), state))

    def _record_measurement(self, packet: rtc.DataPacket) -> None:
        if packet.topic == self._latency_topic:
            measurement = json.loads(packet.data)
            self.observations.measurements.append((time.time(), measurement))

    def latest_state(self) -> str | None:
        return self.observations.states[-1][1] if self.observations.states else None
