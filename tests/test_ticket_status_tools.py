from typing import Any

import httpx
import respx

from voice_agent.conversation import instructions as say
from voice_agent.conversation.ticket_tools import get_ticket_status, request_human_callback
from voice_agent.crm.hubspot_client import HubSpotContact

SARA = HubSpotContact(id="7", first_name="Sara")


def ticket_json(ticket_id: str, subject: str) -> dict[str, Any]:
    properties = {
        "subject": subject,
        "hs_pipeline_stage": "3",
        "hs_lastmodifieddate": "2026-10-29T09:00:00Z",
    }
    return {"id": ticket_id, "properties": properties}


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
    callback = await request_human_callback(
        call_context, reason="Wants a human", spoken_reason="talking to a person"
    )
    assert callback == say.CONTACT_NEEDED
