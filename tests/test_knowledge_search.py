import pytest
from psycopg import AsyncConnection

from voice_agent.knowledge.ingest import KnowledgeChunk, TextEmbedder, replace_knowledge_chunks
from voice_agent.knowledge.search import Reranker, search_knowledge_base

pytestmark = pytest.mark.database

PLANS = KnowledgeChunk("faq#plans", "FAQ", "Plans", "Each plan has a monthly price.")
PAYMENTS = KnowledgeChunk("faq#payments", "FAQ", "Payments", "Payment works with Apple Pay.")
OFFICE = KnowledgeChunk("hours#core", "Hours", "Office", "The office hours are 8 to 6.")


async def seed(connection: AsyncConnection, embed: TextEmbedder) -> None:
    await replace_knowledge_chunks(connection, [PLANS, PAYMENTS, OFFICE], embed)


async def test_question_returns_the_matching_section_first(
    knowledge_connection: AsyncConnection, keyword_embedder: TextEmbedder
) -> None:
    await seed(knowledge_connection, keyword_embedder)

    matches = await search_knowledge_base(
        "What is the price of each plan?",
        connection=knowledge_connection,
        embed=keyword_embedder,
        top_k=2,
        min_similarity=0.1,
    )

    assert matches[0].source_id == "faq#plans"
    assert matches[0].content == PLANS.content
    assert len(matches) <= 2


async def test_unrelated_question_below_the_similarity_floor_returns_nothing(
    knowledge_connection: AsyncConnection, keyword_embedder: TextEmbedder
) -> None:
    await seed(knowledge_connection, keyword_embedder)

    matches = await search_knowledge_base(
        "Do you sell car insurance?",
        connection=knowledge_connection,
        embed=keyword_embedder,
        top_k=3,
        min_similarity=0.5,
    )

    assert matches == []


async def test_reingesting_replaces_previous_chunks_instead_of_duplicating(
    knowledge_connection: AsyncConnection, keyword_embedder: TextEmbedder
) -> None:
    await seed(knowledge_connection, keyword_embedder)
    await replace_knowledge_chunks(knowledge_connection, [OFFICE], keyword_embedder)

    cursor = await knowledge_connection.execute("SELECT source_id FROM knowledge_chunks")

    assert await cursor.fetchall() == [("hours#core",)]


async def test_questions_are_embedded_with_the_e5_query_prefix(
    knowledge_connection: AsyncConnection, keyword_embedder: TextEmbedder
) -> None:
    await seed(knowledge_connection, keyword_embedder)
    embedded_texts: list[str] = []

    def recording_embedder(texts: list[str]) -> list[list[float]]:
        embedded_texts.extend(texts)
        return keyword_embedder(texts)

    await search_knowledge_base(
        "office hours?",
        connection=knowledge_connection,
        embed=recording_embedder,
        top_k=1,
        min_similarity=0.0,
    )

    assert embedded_texts == ["query: office hours?"]


async def test_reranker_reorders_candidates_and_keeps_only_top_k(
    knowledge_connection: AsyncConnection, keyword_embedder: TextEmbedder
) -> None:
    await seed(knowledge_connection, keyword_embedder)

    def prefer_payments(question: str, passages: list[str]) -> list[float]:
        return [1.0 if "Apple Pay" in passage else 0.0 for passage in passages]

    matches = await search_knowledge_base(
        "What is the price of each plan?",
        connection=knowledge_connection,
        embed=keyword_embedder,
        top_k=1,
        min_similarity=0.0,
        reranker=Reranker(score_passages=prefer_payments, candidate_count=3),
    )

    assert [match.source_id for match in matches] == ["faq#payments"]
