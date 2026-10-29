from typing import Any

import httpx
import pytest
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


def mock_sara_owning_ticket_8010(hubspot_api: respx.MockRouter) -> None:
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


async def test_the_callers_own_ticket_status_is_read_back(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    call_context.userdata.call.contact = SARA
    mock_sara_owning_ticket_8010(hubspot_api)

    own = await get_ticket_status(call_context, ticket_number_ending="8010")

    assert own == (
        "Ticket ending 8010: Duplicate charge. Status: Waiting on us. Last updated 29 October 2026."
    )


@pytest.mark.parametrize("language", ["en", "ar"])
async def test_someone_elses_ticket_and_a_missing_ticket_get_the_same_fixed_reply(
    call_context: Any, hubspot_api: respx.MockRouter, language: str
) -> None:
    # 5577 belongs to another customer, 9999 does not exist; the caller must not tell them apart.
    call_context.userdata.call.contact = SARA
    call_context.userdata.call.language = language
    mock_sara_owning_ticket_8010(hubspot_api)

    someone_elses = await get_ticket_status(call_context, ticket_number_ending="5577")
    missing = await get_ticket_status(call_context, ticket_number_ending="9999")

    assert someone_elses is None and missing is None
    assert call_context.session.spoken == [say.NO_MATCHING_TICKET[language]] * 2


async def test_ticket_tools_require_an_identified_caller(call_context: Any) -> None:
    assert await get_ticket_status(call_context) == say.CONTACT_NEEDED
    callback = await request_human_callback(
        call_context, reason="Wants a human", spoken_reason="talking to a person"
    )
    assert callback == say.CONTACT_NEEDED
