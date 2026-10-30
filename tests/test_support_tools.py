import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import psycopg
import pytest
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from voice_agent.conversation.call_state import CallResources
from voice_agent.conversation.instructions import (
    KNOWLEDGE_BASE_UNAVAILABLE,
    NO_KNOWLEDGE_BASE_MATCH,
)
from voice_agent.conversation.tools import look_up_knowledge
from voice_agent.crm.hubspot_client import HubSpotClient
from voice_agent.knowledge.ingest import KnowledgeChunk, TextEmbedder, replace_knowledge_chunks

PLANS = KnowledgeChunk("faq#plans", "FAQ", "Plans", "Each plan has a monthly price.")


class SingleConnectionPool:
    """Hands out the test's transaction-scoped connection, so the rollback still applies."""

    def __init__(self, connection: AsyncConnection) -> None:
        self._connection = connection

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection]:
        yield self._connection


class UnreachablePool:
    @asynccontextmanager
    async def connection(self) -> AsyncIterator[AsyncConnection]:
        raise psycopg.OperationalError("connection refused")
        yield


def call_resources(pool: object, embed: TextEmbedder, min_similarity: float) -> CallResources:
    return CallResources(
        knowledge_pool=cast(AsyncConnectionPool, pool),
        embed=embed,
        reranker=None,
        kb_top_k=3,
        kb_min_similarity=min_similarity,
        crm=cast(HubSpotClient, None),
        ticket_pipeline_id="0",
        new_ticket_stage_id="1",
        ticket_status_limit=3,
        default_country_calling_code="966",
        reply_language_min_letters=6,
    )


@pytest.mark.database
async def test_matching_passages_are_returned_with_their_heading_and_text(
    knowledge_connection: AsyncConnection, keyword_embedder: TextEmbedder
) -> None:
    await replace_knowledge_chunks(knowledge_connection, [PLANS], keyword_embedder)
    resources = call_resources(SingleConnectionPool(knowledge_connection), keyword_embedder, 0.1)

    answer = await look_up_knowledge("What is the price of a plan?", resources)

    assert answer == "[faq#plans] Plans\nEach plan has a monthly price."


@pytest.mark.database
async def test_question_without_matching_passage_tells_the_llm_not_to_guess(
    knowledge_connection: AsyncConnection, keyword_embedder: TextEmbedder
) -> None:
    await replace_knowledge_chunks(knowledge_connection, [PLANS], keyword_embedder)
    resources = call_resources(SingleConnectionPool(knowledge_connection), keyword_embedder, 0.5)

    answer = await look_up_knowledge("Do you sell car insurance?", resources)

    assert answer == NO_KNOWLEDGE_BASE_MATCH


async def test_unreachable_database_keeps_the_caller_talking_and_logs_the_error(
    keyword_embedder: TextEmbedder, caplog: pytest.LogCaptureFixture
) -> None:
    resources = call_resources(UnreachablePool(), keyword_embedder, 0.1)

    with caplog.at_level(logging.ERROR):
        answer = await look_up_knowledge(
            "My email is sara@example.com, what plans exist?", resources
        )

    assert answer == KNOWLEDGE_BASE_UNAVAILABLE
    assert "OperationalError: connection refused" in caplog.text
    assert "sara@example.com" not in caplog.text
