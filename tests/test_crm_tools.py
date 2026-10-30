import json
from typing import Any

import httpx
import respx

from voice_agent.conversation import instructions as say
from voice_agent.conversation.confirmation import NOT_CONFIRMED, WAITING_FOR_ANSWER
from voice_agent.conversation.ticket_tools import (
    create_ticket,
    request_human_callback,
)
from voice_agent.conversation.tools import (
    record_caller_answer,
)
from voice_agent.crm.hubspot_client import HubSpotContact

SARA = HubSpotContact(id="7", first_name="Sara")
NO_CONTACTS = {"results": []}


async def propose_billing_ticket(call_context: Any) -> None:
    call_context.userdata.call.contact = SARA
    call_context.caller_says("I was charged twice for invoice 4412, please open a ticket.")
    return await create_ticket(
        call_context,
        category="billing",
        priority="P2",
        subject="Duplicate charge on invoice 4412",
        description="Charged twice for invoice 4412.",
        spoken_subject="a duplicate charge on invoice 4412",
    )


async def test_ticket_is_read_back_and_nothing_is_saved_before_the_caller_answers(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    tickets = hubspot_api.post("/crm/v3/objects/tickets")

    await propose_billing_ticket(call_context)
    premature_save = await record_caller_answer(call_context)

    assert call_context.session.spoken == [
        "I'll open a P2 billing ticket about a duplicate charge on invoice 4412. Shall I open it?"
    ]
    assert premature_save == WAITING_FOR_ANSWER
    assert not tickets.called


async def test_confirmed_ticket_is_created_for_the_caller_after_a_spoken_yes(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    tickets = hubspot_api.post("/crm/v3/objects/tickets").mock(
        return_value=httpx.Response(201, json={"id": "444586958010"})
    )
    await propose_billing_ticket(call_context)
    call_context.caller_says("Yes, please open it.")

    reply = await record_caller_answer(call_context)

    sent = json.loads(tickets.calls.last.request.content)
    assert sent["properties"]["hs_ticket_priority"] == "MEDIUM"
    assert sent["properties"]["hs_ticket_category"] == "BILLING_ISSUE"
    assert sent["associations"][0]["to"]["id"] == SARA.id
    assert reply == say.TICKET_CREATED.format(ending="8010")
    assert call_context.userdata.call.ticket_ids == ["444586958010"]


async def test_declined_confirmation_creates_no_ticket(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    tickets = hubspot_api.post("/crm/v3/objects/tickets")
    await propose_billing_ticket(call_context)
    call_context.caller_says("No, don't open that.")

    reply = await record_caller_answer(call_context)

    assert reply == NOT_CONFIRMED
    assert not tickets.called
    assert call_context.userdata.call.pending_write is None


async def test_a_later_yes_cannot_approve_an_old_read_back(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    tickets = hubspot_api.post("/crm/v3/objects/tickets")
    await propose_billing_ticket(call_context)
    call_context.caller_says("How long will it take?")
    call_context.caller_says("Yes.")

    assert await record_caller_answer(call_context) == NOT_CONFIRMED
    assert not tickets.called


async def test_confirmed_human_callback_is_an_urgent_ticket_and_ends_the_call(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    tickets = hubspot_api.post("/crm/v3/objects/tickets").mock(
        return_value=httpx.Response(201, json={"id": "901"})
    )
    call_context.userdata.call.contact = SARA
    call_context.caller_says("I want to talk to a human.")
    call_context.userdata.call.language = "ar"
    await request_human_callback(
        call_context, reason="Caller asked for a human", spoken_reason="طلبك التحدث مع موظف"
    )
    assert call_context.session.spoken[-1].startswith("سأطلب من أحد زملائنا الاتصال بك")
    call_context.caller_says("نعم")

    reply = await record_caller_answer(call_context)

    assert (
        json.loads(tickets.calls.last.request.content)["properties"]["hs_ticket_priority"]
        == "URGENT"
    )
    assert reply == say.CALLBACK_CREATED


async def test_crm_outage_keeps_the_caller_talking_and_saves_nothing(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    hubspot_api.post("/crm/v3/objects/tickets").mock(return_value=httpx.Response(503))
    await propose_billing_ticket(call_context)
    call_context.caller_says("Yes")

    assert await record_caller_answer(call_context) == say.CRM_UNAVAILABLE
    assert call_context.userdata.call.ticket_ids == []
    assert call_context.userdata.call.pending_write is None


def test_a_saved_ticket_is_confirmed_by_its_number_without_remarks_on_urgency() -> None:
    instruction = say.TICKET_CREATED.format(ending="8010")

    assert "ends in 8010" in instruction
    assert "anything else" in instruction
    assert "say nothing about its priority or how soon it will be handled" in instruction


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
