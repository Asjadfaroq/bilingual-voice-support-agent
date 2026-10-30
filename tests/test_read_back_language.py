from typing import Any

import respx

from voice_agent.conversation import instructions as say
from voice_agent.conversation.ticket_tools import create_ticket, request_human_callback
from voice_agent.crm.hubspot_client import HubSpotContact

SARA = HubSpotContact(id="7", first_name="Sara")


async def propose_arabic_ticket(call_context: Any, spoken_subject: str) -> str | None:
    call_context.userdata.call.contact = SARA
    call_context.userdata.call.language = "ar"
    call_context.caller_says("التحويل البنكي لفاتورة رقم 7781 ما انعكس، افتحوا لي تذكرة.")
    return await create_ticket(
        call_context,
        category="billing",
        priority="P2",
        subject="Bank transfer for invoice 7781 not reconciled",
        description="A bank transfer for invoice 7781 has not been reconciled.",
        spoken_subject=spoken_subject,
    )


async def test_an_english_subject_in_an_arabic_call_is_sent_back_and_never_read_out(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    tickets = hubspot_api.post("/crm/v3/objects/tickets")

    reply = await propose_arabic_ticket(
        call_context, "a bank transfer for invoice 7781 that has not been reconciled"
    )

    assert reply == say.PHRASE_IN_OTHER_LANGUAGE.format(language="Arabic")
    assert call_context.session.spoken == []
    assert call_context.userdata.call.pending_write is None
    assert not tickets.called


async def test_an_arabic_subject_keeps_invoice_numbers_and_product_names_and_is_read_back(
    call_context: Any,
) -> None:
    reply = await propose_arabic_ticket(call_context, "تحويل بنكي لفاتورة رقم 7781 في Ledgerly")

    assert reply is None
    [read_back] = call_context.session.spoken
    assert "تحويل بنكي لفاتورة رقم 7781 في Ledgerly" in read_back
    assert call_context.userdata.call.pending_write is not None


async def test_a_callback_reason_in_the_wrong_language_is_sent_back_too(call_context: Any) -> None:
    call_context.userdata.call.contact = SARA
    call_context.userdata.call.language = "ar"

    reply = await request_human_callback(
        call_context,
        reason="Invoices cannot be sent",
        spoken_reason="your invoices failing to send",
    )

    assert reply == say.PHRASE_IN_OTHER_LANGUAGE.format(language="Arabic")
    assert call_context.session.spoken == []
