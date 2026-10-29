import logging
from datetime import datetime
from typing import Literal

from livekit.agents import RunContext, function_tool

from voice_agent.conversation import instructions as say
from voice_agent.conversation.tools import CallResources, ask_to_confirm
from voice_agent.crm.hubspot_client import HubSpotError, HubSpotTicket

logger = logging.getLogger(__name__)

TicketCategory = Literal["billing", "bug", "account", "it", "question"]
TicketPriority = Literal["P1", "P2", "P3"]
HUBSPOT_CATEGORY = {"billing": "BILLING_ISSUE", "bug": "PRODUCT_ISSUE"}
HUBSPOT_PRIORITY = {"P1": "HIGH", "P2": "MEDIUM", "P3": "LOW"}
URGENT_PRIORITY = "URGENT"
GENERAL_CATEGORY = "GENERAL_INQUIRY"
TICKET_SUFFIX_DIGITS = 4


def ticket_properties(
    resources: CallResources, subject: str, description: str, priority: str, category: str
) -> dict[str, str]:
    return {
        "subject": subject,
        "content": description,
        "hs_pipeline": resources.ticket_pipeline_id,
        "hs_pipeline_stage": resources.new_ticket_stage_id,
        "hs_ticket_priority": priority,
        "hs_ticket_category": category,
    }


async def open_ticket(resources: CallResources, contact_id: str, properties: dict[str, str]) -> str:
    ticket_id = await resources.crm.create_ticket(contact_id=contact_id, properties=properties)
    resources.call.ticket_ids.append(ticket_id)
    return ticket_id[-TICKET_SUFFIX_DIGITS:]


@function_tool
async def create_ticket(
    context: RunContext[CallResources],
    category: TicketCategory,
    priority: TicketPriority,
    subject: str,
    description: str,
) -> str:
    """Prepare a support ticket for the identified caller and get the read-back to confirm
    with them; nothing is saved until they say yes.

    Args:
        category: billing, bug, account, it or question.
        priority: P1 if the caller cannot create or send invoices, P2 if a core feature is
            broken with a workaround, P3 for a minor issue or a question.
        subject: A short English subject, such as "Duplicate charge on invoice 4412".
        description: What the caller reported, in English.
    """
    resources = context.userdata
    contact = resources.call.contact
    if contact is None:
        return say.CONTACT_NEEDED
    properties = ticket_properties(
        resources,
        subject,
        description,
        HUBSPOT_PRIORITY[priority],
        HUBSPOT_CATEGORY.get(category, GENERAL_CATEGORY),
    )

    async def save_ticket() -> str:
        ending = await open_ticket(resources, contact.id, properties)
        return say.TICKET_CREATED.format(ending=ending)

    return ask_to_confirm(context, f"A {priority} {category} ticket: {subject}", save_ticket)


def describe_ticket(ticket: HubSpotTicket, stage_labels: dict[str, str]) -> str:
    updated = datetime.fromisoformat(ticket.last_modified.replace("Z", "+00:00"))
    status = stage_labels.get(ticket.stage_id, ticket.stage_id)
    return (
        f"Ticket ending {ticket.id[-TICKET_SUFFIX_DIGITS:]}: {ticket.subject}. "
        f"Status: {status}. Last updated {updated:%-d %B %Y}."
    )


@function_tool
async def get_ticket_status(
    context: RunContext[CallResources], ticket_number_ending: str | None = None
) -> str:
    """Look up the status and last update of the identified caller's own tickets.

    Args:
        ticket_number_ending: The last digits of the ticket number, if the caller gave them.
    """
    resources = context.userdata
    contact = resources.call.contact
    if contact is None:
        return say.CONTACT_NEEDED
    try:
        tickets = await resources.crm.tickets_for_contact(contact.id)
        stage_labels = await resources.crm.ticket_stage_labels(resources.ticket_pipeline_id)
    except HubSpotError as error:
        logger.error("Ticket status lookup failed: %s", error)
        return say.CRM_UNAVAILABLE
    if ticket_number_ending:
        tickets = [ticket for ticket in tickets if ticket.id.endswith(ticket_number_ending)]
        if not tickets:
            return say.NO_MATCHING_TICKET
    if not tickets:
        return say.NO_TICKETS
    newest_first = sorted(tickets, key=lambda ticket: ticket.last_modified, reverse=True)
    limited = newest_first[: resources.ticket_status_limit]
    return "\n".join(describe_ticket(ticket, stage_labels) for ticket in limited)


@function_tool
async def request_human_callback(context: RunContext[CallResources], reason: str) -> str:
    """Prepare an urgent callback request so a colleague calls the caller back, and get
    the read-back to confirm with them. Use it when the caller asks for a human.

    Args:
        reason: Why the caller wants a colleague to call back, in English.
    """
    resources = context.userdata
    contact = resources.call.contact
    if contact is None:
        return say.CONTACT_NEEDED
    properties = ticket_properties(
        resources, f"Callback requested: {reason}", reason, URGENT_PRIORITY, GENERAL_CATEGORY
    )

    async def save_callback() -> str:
        await open_ticket(resources, contact.id, properties)
        return say.CALLBACK_CREATED

    return ask_to_confirm(
        context, f"An urgent callback from a colleague about: {reason}", save_callback
    )
