import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import aiohttp
import anthropic
import httpx
from livekit import api as livekit_api

from voice_agent.config import Settings, get_settings

TTS_CHECK_PHRASE = "مرحبا"
ERROR_BODY_PREVIEW_CHARACTERS = 200
KEY_PREFIX_CHARACTERS = 4


@dataclass(frozen=True)
class CheckOutcome:
    passed: bool
    reason: str


AccountCheck = Callable[[Settings], Awaitable[CheckOutcome]]


def describe_http_failure(response: httpx.Response) -> CheckOutcome:
    return CheckOutcome(
        False, f"HTTP {response.status_code}: {response.text[:ERROR_BODY_PREVIEW_CHARACTERS]}"
    )


async def check_livekit(settings: Settings) -> CheckOutcome:
    credential_label = (
        f"{urlparse(settings.livekit_url).netloc} with key "
        f"{settings.livekit_api_key.get_secret_value()[:KEY_PREFIX_CHARACTERS]}…"
    )
    async with livekit_api.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key.get_secret_value(),
        settings.livekit_api_secret.get_secret_value(),
    ) as livekit:
        try:
            rooms = await livekit.room.list_rooms(livekit_api.ListRoomsRequest())
        except livekit_api.TwirpError as error:
            if error.status == httpx.codes.UNAUTHORIZED:
                return CheckOutcome(
                    False,
                    f"HTTP 401 from {credential_label}: the server parsed the token but rejected"
                    " this API key and secret pair",
                )
            return CheckOutcome(
                False, f"HTTP {error.status} {error.code}: {error.message or 'no message'}"
            )
    return CheckOutcome(True, f"listed rooms ({len(rooms.rooms)} active) on {credential_label}")


async def check_elevenlabs_stt(settings: Settings) -> CheckOutcome:
    realtime_url = (
        settings.elevenlabs_base_url.replace("https://", "wss://")
        + f"/speech-to-text/realtime?model_id={settings.stt_model}"
        + f"&audio_format=pcm_{settings.stt_sample_rate_hz}&commit_strategy=manual"
    )
    headers = {"xi-api-key": settings.eleven_api_key.get_secret_value()}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.ws_connect(realtime_url, headers=headers) as websocket:
                first_message = await websocket.receive_json()
        except aiohttp.WSServerHandshakeError as error:
            return CheckOutcome(False, f"websocket handshake HTTP {error.status}: {error.message}")
    if first_message.get("message_type") != "session_started":
        return CheckOutcome(False, f"unexpected first message: {first_message}")
    return CheckOutcome(True, f"realtime session started with {settings.stt_model}")


async def check_elevenlabs_tts(settings: Settings) -> CheckOutcome:
    async with httpx.AsyncClient(timeout=settings.provider_timeout_seconds) as client:
        response = await client.post(
            f"{settings.elevenlabs_base_url}/text-to-speech/{settings.tts_voice_id}",
            headers={"xi-api-key": settings.eleven_api_key.get_secret_value()},
            json={"text": TTS_CHECK_PHRASE, "model_id": settings.tts_model},
        )
    if response.status_code != httpx.codes.OK:
        return describe_http_failure(response)
    return CheckOutcome(
        True, f"synthesised {len(response.content)} bytes with {settings.tts_model}"
    )


async def list_one_hubspot_object(settings: Settings, object_type: str) -> CheckOutcome:
    async with httpx.AsyncClient(
        base_url=settings.hubspot_base_url, timeout=settings.provider_timeout_seconds
    ) as client:
        response = await client.get(
            f"/crm/v3/objects/{object_type}",
            params={"limit": 1},
            headers={"Authorization": f"Bearer {settings.hubspot_access_token.get_secret_value()}"},
        )
    if response.status_code != httpx.codes.OK:
        return describe_http_failure(response)
    return CheckOutcome(True, f"read {object_type} from {settings.hubspot_base_url}")


async def check_hubspot_contacts(settings: Settings) -> CheckOutcome:
    return await list_one_hubspot_object(settings, "contacts")


async def check_hubspot_tickets(settings: Settings) -> CheckOutcome:
    return await list_one_hubspot_object(settings, "tickets")


async def check_anthropic(settings: Settings) -> CheckOutcome:
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        timeout=settings.provider_timeout_seconds,
    )
    try:
        model = await client.models.retrieve(settings.llm_model)
    except anthropic.APIStatusError as error:
        return CheckOutcome(False, f"HTTP {error.status_code}: {error.message}")
    return CheckOutcome(True, f"model {model.id} is available")


ACCOUNT_CHECKS: dict[str, AccountCheck] = {
    "LiveKit": check_livekit,
    "ElevenLabs STT": check_elevenlabs_stt,
    "ElevenLabs TTS": check_elevenlabs_tts,
    "HubSpot contacts": check_hubspot_contacts,
    "HubSpot tickets": check_hubspot_tickets,
    "Anthropic": check_anthropic,
}


def redact_secrets(text: str, settings: Settings) -> str:
    secrets = (
        settings.livekit_api_key,
        settings.livekit_api_secret,
        settings.eleven_api_key,
        settings.anthropic_api_key,
        settings.hubspot_access_token,
    )
    for secret in secrets:
        if secret_value := secret.get_secret_value():
            text = text.replace(secret_value, "[redacted]")
    return text


async def run_check(check: AccountCheck, settings: Settings) -> CheckOutcome:
    # Network failures are reported per provider so one outage does not hide the other results.
    try:
        return await asyncio.wait_for(check(settings), settings.provider_timeout_seconds)
    except TimeoutError:
        return CheckOutcome(False, f"no answer within {settings.provider_timeout_seconds} s")
    except (httpx.TransportError, aiohttp.ClientError, anthropic.APIConnectionError) as error:
        return CheckOutcome(False, f"connection failed: {type(error).__name__}: {error}")


async def check_all_accounts() -> bool:
    settings = get_settings()
    outcomes = await asyncio.gather(
        *(run_check(check, settings) for check in ACCOUNT_CHECKS.values())
    )
    for name, outcome in zip(ACCOUNT_CHECKS, outcomes, strict=True):
        status = "OK  " if outcome.passed else "FAIL"
        print(f"{status} {name:<17} {redact_secrets(outcome.reason, settings)}")
    return all(outcome.passed for outcome in outcomes)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(check_all_accounts()) else 1)
