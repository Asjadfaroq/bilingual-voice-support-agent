import json
from typing import Any

import httpx
import respx

from voice_agent.conversation import instructions as say
from voice_agent.conversation.call_state import caller_identity_note
from voice_agent.conversation.tools import (
    find_or_create_contact,
    identify_caller_by_line,
    record_caller_answer,
)
from voice_agent.crm.hubspot_client import HubSpotContact

SARA = HubSpotContact(id="7", first_name="Sara")
NO_CONTACTS = {"results": []}


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

    await find_or_create_contact(
        call_context, first_name="Sara", last_name="Test", email="sara@ex.com"
    )
    [read_back] = call_context.session.spoken
    assert "email s a r a at e x dot c o m" in read_back and not creation.called
    call_context.caller_says("Yes, that's right")
    await record_caller_answer(call_context)

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


async def test_caller_is_identified_from_their_line_before_they_are_asked_anything(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    hubspot_api.post("/crm/v3/objects/contacts/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"id": "7", "properties": {"firstname": "Sara"}}]}
        )
    )
    call_context.userdata.call.caller_phone = "0551234567"

    await identify_caller_by_line(call_context.userdata)

    assert call_context.userdata.call.contact == SARA
    assert call_context.userdata.call.line_checked


async def test_a_line_not_on_file_is_remembered_so_the_agent_asks_for_an_email(
    call_context: Any, hubspot_api: respx.MockRouter
) -> None:
    hubspot_api.post("/crm/v3/objects/contacts/search").mock(
        return_value=httpx.Response(200, json=NO_CONTACTS)
    )
    call_context.userdata.call.caller_phone = "0551234567"

    await identify_caller_by_line(call_context.userdata)

    assert caller_identity_note(call_context.userdata.call) == say.CALLER_LINE_NOT_ON_FILE_NOTE
