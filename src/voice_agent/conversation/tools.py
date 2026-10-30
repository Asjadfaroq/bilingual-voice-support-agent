import logging

from livekit.agents import RunContext, function_tool
from livekit.agents.telemetry import tracer

from voice_agent.conversation import instructions as say
from voice_agent.conversation.call_state import CallResources, CallState
from voice_agent.conversation.confirmation import (
    PerformWrite,
    WriteKind,
    confirm_before_write,
    propose_write,
    spell_for_read_back,
)
from voice_agent.conversation.knowledge_first import look_up_knowledge
from voice_agent.crm.hubspot_client import HubSpotError, normalize_phone_number
from voice_agent.pipeline.language import LANGUAGE_NAMES, CallerLanguage, phrase_in_language

logger = logging.getLogger(__name__)


@function_tool
async def search_knowledge_base(context: RunContext[CallResources], question: str) -> str:
    """Look up facts about Larkspur Systems and its product Ledgerly: plans and prices,
    payment methods, e-invoicing, data export, free trials, support priorities and response
    times, office hours and public holidays.

    Args:
        question: The caller's question as a short standalone question in English.
    """
    return await look_up_knowledge(question, context.userdata)


def caller_language(call: CallState) -> CallerLanguage:
    return call.language or "en"


def say_read_back(
    context: RunContext[CallResources],
    kind: WriteKind,
    read_back: str,
    perform: PerformWrite,
    spoken_phrase: str = "",
) -> str | None:
    # Spoken here rather than by the LLM, so the caller hears exactly what will be saved; the
    # model's phrase inside it must already be in the caller's language.
    resources = context.userdata
    language = caller_language(resources.call)
    if not phrase_in_language(spoken_phrase, language, resources.reply_language_min_letters):
        return say.PHRASE_IN_OTHER_LANGUAGE.format(language=LANGUAGE_NAMES[language])
    history = context.session.history
    resources.call.pending_write = propose_write(kind, read_back, perform, history)
    context.session.say(read_back)
    return None


async def identify_caller_by_line(resources: CallResources) -> None:
    call = resources.call
    phone = normalize_phone_number(call.caller_phone, resources.default_country_calling_code)
    if phone is None or call.contact is not None:
        return
    try:
        call.contact = await resources.crm.find_contact(phone=phone, email=None)
    except HubSpotError as error:
        logger.error("Looking up the caller's line failed: %s", error)
        return
    call.line_checked = True


@function_tool
async def find_or_create_contact(
    context: RunContext[CallResources],
    first_name: str,
    last_name: str,
    email: str | None = None,
    phone: str | None = None,
) -> str | None:
    """Find the caller's contact by email or phone. If none exists, prepare a new contact;
    the tool then reads it back to the caller itself, and nothing is saved before a yes.

    Args:
        first_name: The caller's first name in Latin letters.
        last_name: The caller's last name in Latin letters.
        email: The caller's email address, if they gave one.
        phone: A phone number the caller gave; leave empty to use the line they call from.
    """
    resources = context.userdata
    if resources.call.contact is not None:
        return say.CONTACT_FOUND.format(first_name=resources.call.contact.first_name)
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

    language = caller_language(resources.call)
    labels = say.DETAIL_LABELS[language]
    details = [f"{first_name} {last_name}"]
    details += [f"{labels['email']} {spell_for_read_back(email)}"] if email else []
    details += [f"{labels['phone']} {spell_for_read_back(phone_number)}"] if phone_number else []
    separator = "، " if language == "ar" else ", "
    read_back = say.READ_BACKS["contact"][language].format(details=separator.join(details))
    say_read_back(context, "contact", read_back, create_contact)
    return None


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
