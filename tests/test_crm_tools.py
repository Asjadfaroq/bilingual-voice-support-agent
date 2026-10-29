import json
from typing import Any

import httpx
import respx

from voice_agent.conversation import instructions as say
from voice_agent.conversation.confirmation import NOT_CONFIRMED, WAITING_FOR_ANSWER
from voice_agent.conversation.ticket_tools import (
    create_ticket,
    get_ticket_status,
    request_human_callback,
)
from voice_agent.conversation.tools import find_or_create_contact, save_confirmed_request
from voice_agent.crm.hubspot_client import HubSpotContact

SARA = HubSpotContact(id="7", first_name="Sara")
NO_CONTACTS = {"results": []}


def ticket_json(ticket_id: str, subject: str) -> dict[str, Any]:
    properties = {
        "subject": subject,
        "hs_pipeline_stage": "3",
        "hs_lastmodifieddate": "2026-10-29T09:00:00Z",
    }
    return {"id": ticket_id, "properties": properties}


async def propose_billing_ticket(call_context: Any) -> str:
    call_context.userdata.call.contact = SARA
    call_context.caller_says("I was charged twice for invoice 4412, please open a ticket.")
    return await create_ticket(
        call_context,
        category="billing",
        priority="P2",
        subject="Duplicate charge on invoice 4412",
        description="Charged twice for invoice 4412.",
    )


async def test_ticket_is_read_back_and_nothing_is_saved_before_the_caller_answers(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    tickets = hubspot_api.post("/crm/v3/objects/tickets")

    read_back = await propose_billing_ticket(call_context)
    premature_save = await save_confirmed_request(call_context)

    assert "A P2 billing ticket: Duplicate charge on invoice 4412" in read_back
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

    reply = await save_confirmed_request(call_context)

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

    reply = await save_confirmed_request(call_context)

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

    assert await save_confirmed_request(call_context) == NOT_CONFIRMED
    assert not tickets.called


async def test_new_contact_spelling_is_read_back_before_it_is_created(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    hubspot_api.post("/crm/v3/objects/contacts/search").mock(
        return_value=httpx.Response(200, json=NO_CONTACTS)
    )
    creation = hubspot_api.post("/crm/v3/objects/contacts").mock(
        return_value=httpx.Response(201, json={"id": "7"})
    )
    call_context.caller_says("I'm Sara Test, sara@ex.com")

    read_back = await find_or_create_contact(
        call_context, first_name="Sara", last_name="Test", email="sara@ex.com"
    )
    assert "s a r a at e x dot c o m" in read_back and not creation.called
    call_context.caller_says("Yes, that's right")
    await save_confirmed_request(call_context)

    assert json.loads(creation.calls.last.request.content)["properties"]["email"] == "sara@ex.com"
    assert call_context.userdata.call.contact == HubSpotContact(id="7", first_name="Sara")


async def test_caller_is_found_by_the_phone_number_from_the_room_metadata(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    search = hubspot_api.post("/crm/v3/objects/contacts/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"id": "7", "properties": {"firstname": "Sara"}}]}
        )
    )
    call_context.userdata.call.caller_phone = "0551234567"

    reply = await find_or_create_contact(call_context, first_name="Sara", last_name="Test")

    filters = json.loads(search.calls.last.request.content)["filterGroups"]
    assert filters[0]["filters"][0]["value"] == "+966551234567"
    assert reply == say.CONTACT_FOUND.format(first_name="Sara")


async def test_ticket_status_is_shared_only_for_the_callers_own_tickets(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    call_context.userdata.call.contact = SARA
    hubspot_api.get("/crm/v4/objects/contacts/7/associations/tickets").mock(
        return_value=httpx.Response(200, json={"results": [{"toObjectId": 444586958010}]})
    )
    hubspot_api.post("/crm/v3/objects/tickets/batch/read").mock(
        return_value=httpx.Response(
            200, json={"results": [ticket_json("444586958010", "Duplicate charge")]}
        )
    )
    hubspot_api.get("/crm/v3/pipelines/tickets/0").mock(
        return_value=httpx.Response(200, json={"stages": [{"id": "3", "label": "Waiting on us"}]})
    )

    own = await get_ticket_status(call_context, ticket_number_ending="8010")
    someone_elses = await get_ticket_status(call_context, ticket_number_ending="1234")

    assert own == (
        "Ticket ending 8010: Duplicate charge. Status: Waiting on us. Last updated 29 October 2026."
    )
    assert someone_elses == say.NO_MATCHING_TICKET


async def test_ticket_tools_require_an_identified_caller(call_context: Any) -> None:
    assert await get_ticket_status(call_context) == say.CONTACT_NEEDED
    assert await request_human_callback(call_context, reason="Wants a human") == say.CONTACT_NEEDED


async def test_confirmed_human_callback_is_an_urgent_ticket_and_ends_the_call(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    tickets = hubspot_api.post("/crm/v3/objects/tickets").mock(
        return_value=httpx.Response(201, json={"id": "901"})
    )
    call_context.userdata.call.contact = SARA
    call_context.caller_says("I want to talk to a human.")
    await request_human_callback(call_context, reason="Caller asked for a human")
    call_context.caller_says("نعم")

    reply = await save_confirmed_request(call_context)

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

    assert await save_confirmed_request(call_context) == say.CRM_UNAVAILABLE
    assert call_context.userdata.call.ticket_ids == []
    assert call_context.userdata.call.pending_write is None
