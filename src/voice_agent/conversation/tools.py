import logging
from dataclasses import dataclass

import psycopg
from livekit.agents import RunContext, function_tool
from psycopg_pool import AsyncConnectionPool

from voice_agent.knowledge import search
from voice_agent.knowledge.ingest import TextEmbedder

logger = logging.getLogger(__name__)

NO_KNOWLEDGE_BASE_MATCH = (
    "No knowledge base passage answers this question. Tell the caller you don't have that "
    "information. Do not guess."
)
KNOWLEDGE_BASE_UNAVAILABLE = (
    "The knowledge base is unavailable right now. Apologise and say a colleague can call "
    "them back. Do not guess."
)


@dataclass(frozen=True)
class CallResources:
    knowledge_pool: AsyncConnectionPool
    embed: TextEmbedder
    reranker: search.Reranker | None
    kb_top_k: int
    kb_min_similarity: float


def format_passages_for_llm(matches: list[search.KnowledgeMatch]) -> str:
    return "\n\n".join(f"[{match.source_id}] {match.heading}\n{match.content}" for match in matches)


async def look_up_knowledge(question: str, resources: CallResources) -> str:
    try:
        async with resources.knowledge_pool.connection() as connection:
            matches = await search.search_knowledge_base(
                question,
                connection=connection,
                embed=resources.embed,
                top_k=resources.kb_top_k,
                min_similarity=resources.kb_min_similarity,
                reranker=resources.reranker,
            )
    except psycopg.Error as error:
        logger.error("Knowledge base search failed: %s: %s", type(error).__name__, error)
        return KNOWLEDGE_BASE_UNAVAILABLE
    if not matches:
        return NO_KNOWLEDGE_BASE_MATCH
    return format_passages_for_llm(matches)


@function_tool
async def search_knowledge_base(context: RunContext[CallResources], question: str) -> str:
    """Look up facts about Larkspur Systems and its product Ledgerly: plans and prices,
    payment methods, e-invoicing, data export, free trials, support priorities and response
    times, office hours and public holidays.

    Args:
        question: The caller's question as a short standalone question in English.
    """
    return await look_up_knowledge(question, context.userdata)
