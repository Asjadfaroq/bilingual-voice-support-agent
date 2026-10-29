import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from psycopg import AsyncConnection

from voice_agent.knowledge.ingest import TextEmbedder, sentence_transformers_installed

PassageScorer = Callable[[str, list[str]], list[float]]

QUERY_PREFIX = "query: "


@dataclass(frozen=True)
class Reranker:
    score_passages: PassageScorer
    candidate_count: int


@dataclass(frozen=True)
class KnowledgeMatch:
    source_id: str
    heading: str
    content: str
    similarity: float


def build_cross_encoder_scorer(model_name: str) -> PassageScorer:
    if not sentence_transformers_installed():
        from voice_agent.knowledge.onnx_models import build_onnx_passage_scorer

        return build_onnx_passage_scorer(model_name)
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    return lambda question, passages: model.predict(
        [(question, passage) for passage in passages]
    ).tolist()


def build_cross_encoder_reranker(model_name: str, candidate_count: int) -> Reranker:
    return Reranker(build_cross_encoder_scorer(model_name), candidate_count)


async def find_similar_chunks(
    connection: AsyncConnection,
    query_embedding: list[float],
    limit: int,
    min_similarity: float,
) -> list[KnowledgeMatch]:
    cursor = await connection.execute(
        "SELECT source_id, heading, content, 1 - (embedding <=> %(query)s::vector) AS similarity"
        " FROM knowledge_chunks"
        " WHERE 1 - (embedding <=> %(query)s::vector) >= %(min_similarity)s"
        " ORDER BY embedding <=> %(query)s::vector"
        " LIMIT %(limit)s",
        {"query": query_embedding, "min_similarity": min_similarity, "limit": limit},
    )
    return [KnowledgeMatch(*row) for row in await cursor.fetchall()]


async def rerank_matches(
    question: str, matches: list[KnowledgeMatch], score_passages: PassageScorer
) -> list[KnowledgeMatch]:
    scores = await asyncio.to_thread(score_passages, question, [match.content for match in matches])
    ranked = sorted(zip(scores, matches, strict=True), key=lambda pair: pair[0], reverse=True)
    return [match for _, match in ranked]


async def search_knowledge_base(
    question: str,
    *,
    connection: AsyncConnection,
    embed: TextEmbedder,
    top_k: int,
    min_similarity: float,
    reranker: Reranker | None = None,
) -> list[KnowledgeMatch]:
    [query_embedding] = await asyncio.to_thread(embed, [f"{QUERY_PREFIX}{question}"])
    if reranker is None:
        return await find_similar_chunks(connection, query_embedding, top_k, min_similarity)
    candidates = await find_similar_chunks(
        connection, query_embedding, max(top_k, reranker.candidate_count), min_similarity
    )
    reranked = await rerank_matches(question, candidates, reranker.score_passages)
    return reranked[:top_k]
