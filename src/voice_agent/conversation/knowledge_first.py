import logging
from collections.abc import AsyncGenerator
from typing import Literal

import psycopg
from livekit.agents.llm import ChatContext, ChatMessage

from voice_agent.conversation.call_state import CallResources, CallState
from voice_agent.conversation.instructions import (
    KNOWLEDGE_BASE_UNAVAILABLE,
    NO_KNOWLEDGE_BASE_MATCH,
    SUPPORT_INSTRUCTIONS_TEMPLATE,
)
from voice_agent.conversation.reply_filter import ReplyChunk, StartReply
from voice_agent.conversation.system_notes import with_system_notes
from voice_agent.knowledge import search

logger = logging.getLogger(__name__)

# "tool": the model calls search_knowledge_base; "every_turn": the knowledge base is searched
# with the caller's words before every generation and the result is added to it.
KnowledgeSearchMode = Literal["tool", "every_turn"]

KNOWLEDGE_TOPICS = (
    "Ledgerly, its plans, prices, payments, billing, data, support priorities or response "
    "times, or about Larkspur's offices and hours"
)
KNOWLEDGE_RULES: dict[KnowledgeSearchMode, str] = {
    "tool": f"- For any question about {KNOWLEDGE_TOPICS}, call search_knowledge_base first, "
    "even if you think you know the answer.",
    "every_turn": f"- Every caller turn comes with the knowledge base search result for it. "
    f"Answer any question about {KNOWLEDGE_TOPICS} from that result alone, even if you think "
    "you know the answer.",
}
SUPPORT_INSTRUCTIONS: dict[KnowledgeSearchMode, str] = {
    mode: SUPPORT_INSTRUCTIONS_TEMPLATE.format(knowledge_rule=rule)
    for mode, rule in KNOWLEDGE_RULES.items()
}
KNOWLEDGE_FOR_THIS_TURN = "Knowledge base search result for the caller's latest turn:\n{passages}"
# Most turns that find nothing are not questions ("yes", an email address), so a miss must not
# carry the search tool's "offer a ticket" instruction into a ticket or contact flow.
NOTHING_FOUND_THIS_TURN = "The knowledge base search for the caller's latest turn found nothing."
EARLIER_KNOWLEDGE = (
    "Knowledge base results for the caller's earlier turns, which your earlier answers were "
    "based on:\n{passages}"
)


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
    return "\n\n".join(f"[{match.source_id}] {match.heading}\n{match.content}" for match in matches)


def earlier_knowledge_notes(call: CallState, mode: KnowledgeSearchMode) -> list[str]:
    # Each turn's search result reaches only that turn's generation; without the earlier ones the
    # model judged its own correct earlier answers ungrounded and took them back. The search
    # tool's results stay in the history, so only searching before the model needs this.
    found = dict.fromkeys(
        passages
        for _, passages in call.knowledge_lookups
        if passages not in (NO_KNOWLEDGE_BASE_MATCH, KNOWLEDGE_BASE_UNAVAILABLE)
    )
    if mode != "every_turn" or not found:
        return []
    return [EARLIER_KNOWLEDGE.format(passages="\n\n".join(found))]


def searches_before_reply(chat_ctx: ChatContext, call: CallState) -> bool:
    # Only a new caller turn is searched: after a tool call the search would repeat and its
    # passages leaked into ticket answers, and an answer to a read-back is not a question.
    latest = chat_ctx.items[-1] if chat_ctx.items else None
    is_caller_turn = isinstance(latest, ChatMessage) and latest.role == "user"
    return is_caller_turn and call.pending_write is None


async def reply_with_knowledge(
    start_reply: StartReply, chat_ctx: ChatContext, question: str, resources: CallResources
) -> AsyncGenerator[ReplyChunk, None]:
    passages = await look_up_knowledge(question, resources)
    resources.call.knowledge_lookups.append((question, passages))
    note = (
        NOTHING_FOUND_THIS_TURN
        if passages == NO_KNOWLEDGE_BASE_MATCH
        else KNOWLEDGE_FOR_THIS_TURN.format(passages=passages)
    )
    async for chunk in start_reply(with_system_notes(chat_ctx, note)):
        yield chunk
