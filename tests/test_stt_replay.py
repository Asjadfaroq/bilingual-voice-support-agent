import numpy as np
import pytest
from livekit import rtc
from livekit.agents import stt

from voice_agent.pipeline.stt_input import AudioInput, SpeechGatedReplayChannel, SpeechInputOptions

SAMPLE_RATE_HZ = 16000
SPEECH_RMS = 150.0
LOUD = 1000
PREROLL_SECONDS = 0.5
FLUSH = stt.RecognizeStream._FlushSentinel


def frame(seconds: float, marker: int, level: int = LOUD) -> rtc.AudioFrame:
    samples = int(SAMPLE_RATE_HZ * seconds)
    pcm = np.full(samples, level + marker, dtype=np.int16)
    return rtc.AudioFrame(pcm.tobytes(), SAMPLE_RATE_HZ, 1, samples)


def channel_keeping(max_replay_seconds: float) -> SpeechGatedReplayChannel:
    return SpeechGatedReplayChannel(
        SpeechInputOptions(SPEECH_RMS, PREROLL_SECONDS, max_replay_seconds)
    )


async def receive(channel: SpeechGatedReplayChannel, count: int) -> list[AudioInput]:
    return [await channel.recv() for _ in range(count)]


async def test_audio_sent_on_a_dropped_connection_is_sent_again_before_new_audio() -> None:
    channel = channel_keeping(30.0)
    first, second, new = frame(0.5, 1), frame(0.5, 2), frame(0.5, 3)
    channel.send_nowait(first)
    channel.send_nowait(second)
    await receive(channel, 2)
    channel.send_nowait(new)

    assert channel.replay_unconfirmed() == pytest.approx(1.0)
    assert await receive(channel, 3) == [first, second, new]


async def test_a_confirmed_commit_is_not_sent_again() -> None:
    channel = channel_keeping(30.0)
    committed, commit, next_turn = frame(1.0, 1), FLUSH(), frame(0.5, 2)
    for item in (committed, commit, next_turn):
        channel.send_nowait(item)
    await receive(channel, 3)

    channel.confirm_commit()

    assert channel.replay_unconfirmed() == pytest.approx(0.5)
    assert await channel.recv() is next_turn


async def test_an_unconfirmed_commit_is_sent_again_with_its_audio() -> None:
    channel = channel_keeping(30.0)
    words, commit = frame(1.0, 1), FLUSH()
    channel.send_nowait(words)
    channel.send_nowait(commit)
    await receive(channel, 2)

    channel.replay_unconfirmed()

    assert await receive(channel, 2) == [words, commit]


async def test_only_the_most_recent_audio_is_kept_for_replay() -> None:
    channel = channel_keeping(1.0)
    frames = [frame(0.5, marker) for marker in range(4)]
    for item in frames:
        channel.send_nowait(item)
    await receive(channel, 4)

    assert channel.replay_unconfirmed() == pytest.approx(1.0)
    assert await receive(channel, 2) == frames[2:]


async def test_the_silence_before_the_caller_spoke_again_is_not_replayed() -> None:
    channel = channel_keeping(30.0)
    first_words, silence, speech = frame(0.5, 9), frame(8.0, 0, level=0), frame(0.6, 1)
    for item in (first_words, FLUSH(), silence, speech):
        channel.send_nowait(item)
    await receive(channel, 4)
    channel.confirm_commit()

    assert channel.replay_unconfirmed() == pytest.approx(0.6)
    assert await channel.recv() is speech


async def test_nothing_before_the_callers_first_words_is_sent_but_the_preroll() -> None:
    channel = channel_keeping(30.0)
    greeting_silence = [frame(0.25, 0, level=0) for _ in range(40)]
    first_words = frame(0.5, 1)
    for item in (*greeting_silence, first_words):
        channel.send_nowait(item)

    sent = await receive(channel, 3)

    assert sent == [*greeting_silence[-2:], first_words]


async def test_a_flush_opens_the_gate_for_speech_too_quiet_to_measure() -> None:
    channel = channel_keeping(30.0)
    quiet_words, commit = frame(0.5, 0, level=40), FLUSH()
    channel.send_nowait(quiet_words)
    channel.send_nowait(commit)

    assert await receive(channel, 2) == [quiet_words, commit]
