import math
import re
from collections.abc import AsyncIterator

import pytest
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection

from voice_agent.config import Settings
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
