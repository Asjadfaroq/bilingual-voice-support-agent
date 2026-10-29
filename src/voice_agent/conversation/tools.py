import logging
from dataclasses import dataclass, field

import psycopg
from livekit.agents import RunContext, function_tool
from livekit.agents.telemetry import tracer
from psycopg_pool import AsyncConnectionPool

from voice_agent.conversation import instructions as say
from voice_agent.conversation.confirmation import (
    PendingWrite,
    PerformWrite,
    WriteKind,
    confirm_before_write,
    propose_write,
    spell_for_read_back,
)
from voice_agent.crm.hubspot_client import (
    HubSpotClient,
    HubSpotContact,
    HubSpotError,
    normalize_phone_number,
)
from voice_agent.knowledge import search
from voice_agent.knowledge.ingest import TextEmbedder

logger = logging.getLogger(__name__)


@dataclass
class CallState:
    caller_phone: str | None = None
    contact: HubSpotContact | None = None
    ticket_ids: list[str] = field(default_factory=list)
    pending_write: PendingWrite | None = None


@dataclass(frozen=True)
class CallResources:
    knowledge_pool: AsyncConnectionPool
    embed: TextEmbedder
    reranker: search.Reranker | None
    kb_top_k: int
    kb_min_similarity: float
    crm: HubSpotClient
    ticket_pipeline_id: str
    new_ticket_stage_id: str
    ticket_status_limit: int
    default_country_calling_code: str
    call: CallState = field(default_factory=CallState)


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
        return say.KNOWLEDGE_BASE_UNAVAILABLE
    if not matches:
        return say.NO_KNOWLEDGE_BASE_MATCH
    return "\n\n".join(f"[{match.source_id}] {match.heading}\n{match.content}" for match in matches)


@function_tool
async def search_knowledge_base(context: RunContext[CallResources], question: str) -> str:
    """Look up facts about Larkspur Systems and its product Ledgerly: plans and prices,
    payment methods, e-invoicing, data export, free trials, support priorities and response
    times, office hours and public holidays.

    Args:
        question: The caller's question as a short standalone question in English.
    """
    return await look_up_knowledge(question, context.userdata)


def ask_to_confirm(
    context: RunContext[CallResources], kind: WriteKind, read_back: str, perform: PerformWrite
) -> str:
    history = context.session.history
    context.userdata.call.pending_write = propose_write(kind, read_back, perform, history)
    return say.READ_BACK_AND_ASK.format(read_back=read_back)


@function_tool
async def find_or_create_contact(
    context: RunContext[CallResources],
    first_name: str,
    last_name: str,
    email: str | None = None,
    phone: str | None = None,
) -> str:
    """Find the caller's contact by phone or email. If none exists, prepare a new contact
    and get the read-back to confirm with the caller before anything is saved.

    Args:
        first_name: The caller's first name in Latin letters.
        last_name: The caller's last name in Latin letters.
        email: The caller's email address, if they gave one.
        phone: A phone number the caller gave; leave empty to use the line they call from.
    """
    resources = context.userdata
    phone_number = normalize_phone_number(
        phone or resources.call.caller_phone, resources.default_country_calling_code
    )
    if not phone_number and not email:
        return say.ASK_FOR_EMAIL_OR_PHONE
    try:
        contact = await resources.crm.find_contact(phone=phone_number, email=email)
    except HubSpotError as error:
        logger.error("Contact lookup failed: %s", error)
        return say.CRM_UNAVAILABLE
    if contact is not None:
        resources.call.contact = contact
        return say.CONTACT_FOUND.format(first_name=contact.first_name)

    async def create_contact() -> str:
        resources.call.contact = await resources.crm.create_contact(
            first_name=first_name, last_name=last_name, email=email, phone=phone_number
        )
        return say.CONTACT_CREATED

    details = [f"{first_name} {last_name}"]
    details += [f"email {spell_for_read_back(email)}"] if email else []
    details += [f"phone {spell_for_read_back(phone_number)}"] if phone_number else []
    read_back = "A new contact for " + ", ".join(details)
    return ask_to_confirm(context, "contact", read_back, create_contact)


@function_tool
async def record_caller_answer(context: RunContext[CallResources]) -> str:
    """Pass on the caller's answer to the read-back you just gave, whether it was yes, no or
    anything else. The request is saved only if the answer is a clear yes."""
    call = context.userdata.call
    pending = call.pending_write
    with tracer.start_as_current_span("confirmation_decision") as span:
        span.set_attribute("voice_agent.write.kind", pending.kind if pending else "none")
        try:
            result = await confirm_before_write(pending, context.session.history)
        except HubSpotError as error:
            logger.error("Saving a confirmed request failed: %s", error)
            span.set_attribute("voice_agent.confirmation.decision", "failed")
            call.pending_write = None
            return say.CRM_UNAVAILABLE
        span.set_attribute("voice_agent.confirmation.decision", result.decision)
        span.set_attribute("voice_agent.confirmation.caller_answer", result.caller_answer or "")
    if result.pending_resolved:
        call.pending_write = None
    return result.reply_for_llm
