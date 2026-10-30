import argparse
import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bench_caller import BenchCaller
from bench_setup import LATENCY_TOPIC, caller_token, start_worker, wait_until
from demo_audio import line_audio, load_script
from demo_records import logged_call_links
from latency_bench import reply_finished
from livekit import api, rtc

from voice_agent.config import Settings, get_settings

LISTENER_IDENTITY = "demo-listener"
LISTENER_TOKEN_LIFETIME = timedelta(hours=2)
# The call is logged after the caller hangs up, once Claude has summarised it.
CALL_LOG_TIMEOUT_SECONDS = 120.0
POLL_SECONDS = 5.0


def listener_token(settings: Settings, room_name: str) -> str:
    # Hidden and subscribe-only, so the agent never sees the listener and links to the caller.
    grants = api.VideoGrants(
        room_join=True, room=room_name, hidden=True, can_publish=False, can_publish_data=False
    )
    return (
        api.AccessToken(
            settings.livekit_api_key.get_secret_value(),
            settings.livekit_api_secret.get_secret_value(),
        )
        .with_identity(LISTENER_IDENTITY)
        .with_grants(grants)
        .with_ttl(LISTENER_TOKEN_LIFETIME)
        .to_jwt()
    )


async def run_demo(language: str, wait_for_listener: bool) -> None:
    settings, script = get_settings(), load_script(language)
    audio = [line_audio(line, script.voice, settings) for line in script.lines]
    agent_name = f"demo-{uuid.uuid4().hex[:8]}"
    room_name = f"demo-call-{language}-{uuid.uuid4().hex[:6]}"
    log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"{agent_name}.log"
    worker = await start_worker(
        {}, agent_name, log_path, settings.bench_worker_log_level, log_calls=True
    )
    try:
        if wait_for_listener:
            print("\nIn the Agents Playground choose Manual and enter:")
            print(f"  URL:   {settings.livekit_url}")
            print(f"  Token: {listener_token(settings, room_name)}\n")
            await asyncio.to_thread(input, "Press Enter once the Playground says it is connected… ")
        since = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
        room = rtc.Room()
        caller = BenchCaller(room, settings.stt_sample_rate_hz, LATENCY_TOPIC)
        await room.connect(
            settings.livekit_url, caller_token(settings, room_name, f"caller-{language}", None)
        )
        await caller.start()
        async with api.LiveKitAPI(
            settings.livekit_url,
            settings.livekit_api_key.get_secret_value(),
            settings.livekit_api_secret.get_secret_value(),
        ) as livekit:
            request = api.CreateAgentDispatchRequest(agent_name=agent_name, room=room_name)
            await livekit.agent_dispatch.create_dispatch(request)
        quiet, timeout = (
            settings.bench_quiet_before_turn_seconds,
            settings.bench_reply_timeout_seconds,
        )
        await wait_until(reply_finished(caller, 0.0, quiet), timeout)
        for line, pcm in zip(script.lines, audio, strict=True):
            print(f"caller: {line.clip}")
            ended = await caller.say(pcm)
            await wait_until(reply_finished(caller, ended, quiet), timeout)
        await room.disconnect()
        print("Waiting for the call to be logged in HubSpot…")
        links = None
        deadline = asyncio.get_running_loop().time() + CALL_LOG_TIMEOUT_SECONDS
        while links is None and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(POLL_SECONDS)
            links = await logged_call_links(settings, script.email, since)
        print("\n".join(links) if links else f"No call log yet; see {log_path}")
    finally:
        worker.terminate()
        await worker.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a scripted demo call into a LiveKit room.")
    parser.add_argument("language", choices=["en", "ar"])
    parser.add_argument("--no-listener", action="store_true", help="Start without waiting")
    arguments = parser.parse_args()
    asyncio.run(run_demo(arguments.language, not arguments.no_listener))
