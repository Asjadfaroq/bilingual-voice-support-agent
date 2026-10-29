import json

import httpx
import pytest
import respx

from voice_agent.crm.hubspot_client import (
    CALL_TO_CONTACT,
    CALL_TO_TICKET,
    TICKET_TO_CONTACT,
    HubSpotClient,
    HubSpotError,
    normalize_phone_number,
)


@pytest.mark.parametrize(
    "raw_phone",
    ["0551234567", "+966 55 123 4567", "00966551234567", "٠٥٥١٢٣٤٥٦٧", "(055) 123-4567"],
)
def test_saudi_phone_numbers_are_normalised_to_e164(raw_phone: str) -> None:
    assert normalize_phone_number(raw_phone, "966") == "+966551234567"


async def test_server_error_is_retried_before_succeeding(
    hubspot_client: HubSpotClient, hubspot_api: respx.MockRouter
) -> None:
    route = hubspot_api.get("/crm/v3/pipelines/tickets/0").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(200, json={"stages": [{"id": "1", "label": "New"}]}),
        ]
    )

    assert await hubspot_client.ticket_stage_labels("0") == {"1": "New"}
    assert route.call_count == 2


async def test_rate_limit_and_network_failures_give_up_with_a_clear_error(
    hubspot_client: HubSpotClient, hubspot_api: respx.MockRouter
) -> None:
    hubspot_api.get("/crm/v3/pipelines/tickets/0").mock(
        side_effect=[httpx.Response(429), httpx.ConnectError("connection refused")]
    )

    with pytest.raises(
        HubSpotError, match=r"GET /crm/v3/pipelines/tickets/0 failed after 2 attempts: ConnectError"
    ):
        await hubspot_client.ticket_stage_labels("0")


async def test_client_error_is_not_retried_and_quotes_hubspots_message(
    hubspot_client: HubSpotClient, hubspot_api: respx.MockRouter
) -> None:
    route = hubspot_api.post("/crm/v3/objects/tickets").mock(
        return_value=httpx.Response(400, json={"message": "Property values were not valid"})
    )

    with pytest.raises(HubSpotError, match="HTTP 400: Property values were not valid"):
        await hubspot_client.create_ticket(contact_id="7", properties={"subject": "x"})
    assert route.call_count == 1


async def test_contact_is_searched_by_phone_or_email(
    hubspot_client: HubSpotClient, hubspot_api: respx.MockRouter
) -> None:
    route = hubspot_api.post("/crm/v3/objects/contacts/search").mock(
        return_value=httpx.Response(
            200, json={"results": [{"id": "7", "properties": {"firstname": "Sara"}}]}
        )
    )

    contact = await hubspot_client.find_contact(phone="+966551234567", email="sara@example.com")

    sent = json.loads(route.calls.last.request.content)
    assert [group["filters"][0]["propertyName"] for group in sent["filterGroups"]] == [
        "phone",
        "email",
    ]
    assert contact is not None and contact.id == "7" and contact.first_name == "Sara"


async def test_ticket_is_created_associated_with_the_contact(
    hubspot_client: HubSpotClient, hubspot_api: respx.MockRouter
) -> None:
    route = hubspot_api.post("/crm/v3/objects/tickets").mock(
        return_value=httpx.Response(201, json={"id": "901"})
    )

    assert await hubspot_client.create_ticket(contact_id="7", properties={"subject": "x"}) == "901"
    [link] = json.loads(route.calls.last.request.content)["associations"]
    assert link["to"]["id"] == "7"
    assert link["types"][0]["associationTypeId"] == TICKET_TO_CONTACT


async def test_call_is_associated_with_the_contact_and_every_ticket(
    hubspot_client: HubSpotClient, hubspot_api: respx.MockRouter
) -> None:
    route = hubspot_api.post("/crm/v3/objects/calls").mock(
        return_value=httpx.Response(201, json={"id": "55"})
    )

    await hubspot_client.create_call(properties={}, contact_id="7", ticket_ids=["901", "902"])

    links = json.loads(route.calls.last.request.content)["associations"]
    assert [(link["to"]["id"], link["types"][0]["associationTypeId"]) for link in links] == [
        ("7", CALL_TO_CONTACT),
        ("901", CALL_TO_TICKET),
        ("902", CALL_TO_TICKET),
    ]


async def test_tickets_for_a_contact_come_from_its_associations(
    hubspot_client: HubSpotClient, hubspot_api: respx.MockRouter
) -> None:
    hubspot_api.get("/crm/v4/objects/contacts/7/associations/tickets").mock(
        return_value=httpx.Response(200, json={"results": [{"toObjectId": 901}]})
    )
    batch = hubspot_api.post("/crm/v3/objects/tickets/batch/read").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "901",
                        "properties": {
                            "subject": "Duplicate charge",
                            "hs_pipeline_stage": "1",
                            "hs_lastmodifieddate": "2026-10-29T10:00:00Z",
                        },
                    }
                ]
            },
        )
    )

    [ticket] = await hubspot_client.tickets_for_contact("7")

    assert json.loads(batch.calls.last.request.content)["inputs"] == [{"id": "901"}]
    assert (ticket.id, ticket.subject, ticket.stage_id) == ("901", "Duplicate charge", "1")
