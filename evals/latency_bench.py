import argparse
import asyncio
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import httpx
from bench_caller import BenchCaller
from bench_report import BenchTurn, describe_turn, save_results, summarise_language, turns_as_json
from bench_setup import (
    LATENCY_TOPIC,
    caller_token,
    elevenlabs_characters_left,
    livekit_round_trip_ms,
    load_utterances,
    start_worker,
    wait_until,
)
from livekit import rtc

from voice_agent.config import Settings, get_settings


def reply_finished(caller: BenchCaller, after: float, quiet_seconds: float) -> Callable[[], bool]:
    def finished() -> bool:
        spoke = any(t > after and state == "speaking" for t, state in caller.observations.states)
        quiet_for = time.time() - caller.observations.last_audible_at
        return spoke and caller.latest_state() == "listening" and quiet_for > quiet_seconds

    return finished


async def bench_language(
    settings: Settings, agent_name: str, language: str, turn_count: int
) -> list[BenchTurn]:
    room_name = f"latency-bench-{language}-{uuid.uuid4().hex[:8]}"
    room = rtc.Room()
    caller = BenchCaller(room, settings.stt_sample_rate_hz, LATENCY_TOPIC)
    token = caller_token(settings, room_name, f"bench-caller-{language}", agent_name)
    await room.connect(settings.livekit_url, token)
    await caller.start()
    quiet, timeout = settings.bench_quiet_before_turn_seconds, settings.bench_reply_timeout_seconds
    if not await wait_until(reply_finished(caller, 0.0, quiet), timeout):
        raise RuntimeError(f"No greeting in room {room_name}")
    utterances = load_utterances(settings, language)
    turns: list[BenchTurn] = []
    left = await elevenlabs_characters_left(settings)
    for index in range(turn_count):
        try:
            left = await elevenlabs_characters_left(settings)
        except httpx.HTTPError as error:
            # A turn costs a few hundred characters at most, so the last balance still guards.
            print(f"Credit check failed ({error!r}); using the last balance, {left}")
        if left < settings.bench_min_elevenlabs_characters_left:
            print(f"Stopping: {left} ElevenLabs characters left")
            break
        utterance_id, pcm = utterances[index % len(utterances)]
        started = time.time()
        ended = await caller.say(pcm)
        await wait_until(reply_finished(caller, ended, quiet), timeout)
        turn = describe_turn(utterance_id, language, caller.observations, started, ended)
        print(f"{language} {index + 1}/{turn_count} {utterance_id}: {turn.stages}")
        turns.append(turn)
    await room.disconnect()
    return turns


async def run_bench(
    label: str, overrides: dict[str, str], languages: list[str], turn_count: int | None
) -> None:
    settings = get_settings()
    characters_before = await elevenlabs_characters_left(settings)
    round_trip_before = await livekit_round_trip_ms(settings)
    agent_name = f"latency-bench-{uuid.uuid4().hex[:8]}"
    log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"{agent_name}.log"
    worker = await start_worker(overrides, agent_name, log_path, settings.bench_worker_log_level)
    turns_per_language = turn_count or settings.bench_turns_per_language
    try:
        turns = {
            language: await bench_language(settings, agent_name, language, turns_per_language)
            for language in languages
        }
    finally:
        worker.terminate()
        await worker.wait()
    characters_used = characters_before - await elevenlabs_characters_left(settings)
    run = {
        "settings": overrides,
        "elevenlabs_characters_used": characters_used,
        "livekit_round_trip_ms": {
            "before": round_trip_before,
            "after": await livekit_round_trip_ms(settings),
        },
        "languages": {language: summarise_language(t) for language, t in turns.items()},
        "turns": [
            turn for language_turns in turns.values() for turn in turns_as_json(language_turns)
        ],
    }
    save_results("voice", label, run)
    print(f"Saved {label}; ElevenLabs characters used: {characters_used}; worker log {log_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure turn latency through a LiveKit room.")
    parser.add_argument("label", help="Name of this configuration in latency.json")
    parser.add_argument("--set", nargs="*", default=[], help="Settings as NAME=value")
    parser.add_argument("--languages", nargs="+", default=["en", "ar"])
    parser.add_argument("--turns", type=int, help="Turns per language (default from settings)")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    overrides = dict(setting.split("=", 1) for setting in arguments.set)
    asyncio.run(run_bench(arguments.label, overrides, arguments.languages, arguments.turns))
