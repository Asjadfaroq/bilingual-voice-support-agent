import math
import re
from collections.abc import AsyncIterator, Iterator
from typing import cast

import httpx
import pytest
import respx
from livekit.agents.llm import ChatContext
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from voice_agent.config import Settings
from voice_agent.conversation.call_state import CallResources, CallState
from voice_agent.crm.hubspot_client import HubSpotClient
from voice_agent.knowledge.ingest import TextEmbedder, create_knowledge_table

FAKE_VOCABULARY = ("plan", "price", "payment", "apple", "office", "hours", "ticket", "export")


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        _env_file=None,
        livekit_url="wss://livekit.test",
        livekit_api_key="test-livekit-key",
        livekit_api_secret="test-livekit-secret",
        eleven_api_key="test-eleven-key",
        anthropic_api_key="test-anthropic-key",
        hubspot_access_token="test-hubspot-token",
    )


@pytest.fixture
def keyword_embedder(test_settings: Settings) -> TextEmbedder:
    """Embeds text by which vocabulary words it contains, so similarity is predictable."""

    def embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            vector = [0.0] * test_settings.embedding_dimensions
            for position, keyword in enumerate(FAKE_VOCABULARY):
                vector[position] = 1.0 if keyword in words else 0.0
            vector[-1] = 0.1
            norm = math.sqrt(sum(component * component for component in vector))
            vectors.append([component / norm for component in vector])
        return vectors

    return embed


@pytest.fixture
async def knowledge_connection(test_settings: Settings) -> AsyncIterator[AsyncConnection]:
    # Every test runs inside one transaction that is rolled back, so the seeded KB survives.
    async with await AsyncConnection.connect(test_settings.database_url) as connection:
        await create_knowledge_table(connection, test_settings.embedding_dimensions)
        await register_vector_async(connection)
        yield connection
        await connection.rollback()


HUBSPOT_TEST_URL = "https://api-eu1.hubapi.test"


@pytest.fixture
async def hubspot_client() -> AsyncIterator[HubSpotClient]:
    http = httpx.AsyncClient(base_url=HUBSPOT_TEST_URL)
    yield HubSpotClient(http, max_attempts=2, retry_backoff_seconds=0)
    await http.aclose()


@pytest.fixture
def hubspot_api() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=HUBSPOT_TEST_URL, assert_all_called=False) as router:
        yield router


def make_call_resources(crm: HubSpotClient, embed: TextEmbedder) -> CallResources:
    return CallResources(
        knowledge_pool=cast(AsyncConnectionPool, None),
        embed=embed,
        reranker=None,
        kb_top_k=3,
        kb_min_similarity=0.5,
        crm=crm,
        ticket_pipeline_id="0",
        new_ticket_stage_id="1",
        ticket_status_limit=3,
        default_country_calling_code="966",
        reply_language_min_letters=6,
        call=CallState(),
    )


class FakeSession:
    def __init__(self) -> None:
        self.history = ChatContext.empty()
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)
        self.history.add_message(role="assistant", content=text)


class FakeRunContext:
    """The parts of LiveKit's RunContext the tools use: per-call resources and history."""

    def __init__(self, resources: CallResources) -> None:
        self.userdata = resources
        self.session = FakeSession()

    def caller_says(self, text: str) -> None:
        self.session.history.add_message(role="user", content=text)

    def agent_says(self, text: str) -> None:
        self.session.history.add_message(role="assistant", content=text)


@pytest.fixture
def call_context(hubspot_client: HubSpotClient, keyword_embedder: TextEmbedder) -> FakeRunContext:
    return FakeRunContext(make_call_resources(hubspot_client, keyword_embedder))
