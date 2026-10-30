from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from psycopg_pool import AsyncConnectionPool

from voice_agent.config import Settings
from voice_agent.conversation.instructions import (
    CALLER_IDENTIFIED_NOTE,
    CALLER_LINE_NOT_ON_FILE_NOTE,
)
from voice_agent.crm.hubspot_client import HubSpotClient, HubSpotContact
from voice_agent.knowledge.ingest import TextEmbedder
from voice_agent.knowledge.search import Reranker
from voice_agent.pipeline.language import CallerLanguage

if TYPE_CHECKING:
    from voice_agent.conversation.confirmation import PendingWrite


@dataclass
class CallState:
    caller_phone: str | None = None
    # True once the caller's line was looked up, so "not on file" differs from "unknown".
    line_checked: bool = False
    contact: HubSpotContact | None = None
    language: CallerLanguage | None = None
    # Every language the caller spoke, in the order they first used it, for the call log.
    languages: list[CallerLanguage] = field(default_factory=list)
    ticket_ids: list[str] = field(default_factory=list)
    pending_write: PendingWrite | None = None
    # Searches made before the model (kb_search_mode "every_turn"): question and result, so a
    # transcript shows what each reply was grounded on.
    knowledge_lookups: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class CallResources:
    knowledge_pool: AsyncConnectionPool
    embed: TextEmbedder
    reranker: Reranker | None
    kb_top_k: int
    kb_min_similarity: float
    crm: HubSpotClient
    ticket_pipeline_id: str
    new_ticket_stage_id: str
    ticket_status_limit: int
    default_country_calling_code: str
    reply_language_min_letters: int
    call: CallState = field(default_factory=CallState)


def call_resources_for(
    settings: Settings,
    knowledge_pool: AsyncConnectionPool,
    embed: TextEmbedder,
    reranker: Reranker | None,
    crm: HubSpotClient,
    call: CallState,
) -> CallResources:
    return CallResources(
        knowledge_pool=knowledge_pool,
        embed=embed,
        reranker=reranker,
        kb_top_k=settings.kb_top_k,
        kb_min_similarity=settings.kb_min_similarity,
        crm=crm,
        ticket_pipeline_id=settings.hubspot_ticket_pipeline_id,
        new_ticket_stage_id=settings.hubspot_new_ticket_stage_id,
        ticket_status_limit=settings.ticket_status_limit,
        default_country_calling_code=settings.default_country_calling_code,
        reply_language_min_letters=settings.reply_language_min_letters,
        call=call,
    )


def caller_identity_note(call: CallState) -> str | None:
    if call.contact is not None:
        return CALLER_IDENTIFIED_NOTE.format(first_name=call.contact.first_name)
    return CALLER_LINE_NOT_ON_FILE_NOTE if call.line_checked else None


def caller_phone_from_metadata(room_metadata: str, key: str) -> str | None:
    # Playground rooms have no metadata; a phone gateway would put the caller's number here.
    try:
        metadata = json.loads(room_metadata) if room_metadata else {}
    except json.JSONDecodeError:
        return None
    phone = metadata.get(key) if isinstance(metadata, dict) else None
    return phone if isinstance(phone, str) and phone else None
