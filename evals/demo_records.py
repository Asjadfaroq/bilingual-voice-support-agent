import httpx

from voice_agent.config import Settings

CALL_TITLE_PREFIX = "Voice support call"
HUBSPOT_RECORD_PAGES = {"contacts": "0-1", "tickets": "0-5", "calls": "0-48"}


async def logged_call_links(settings: Settings, since_epoch_ms: int) -> list[str] | None:
    """Links to the newest call the agent logged since the demo started, and to the contact and
    tickets linked to it. Found through the call rather than the caller's email, which the
    speech-to-text does not always hear the same way."""
    headers = {"Authorization": f"Bearer {settings.hubspot_access_token.get_secret_value()}"}
    async with httpx.AsyncClient(
        base_url=settings.hubspot_base_url,
        headers=headers,
        timeout=settings.provider_timeout_seconds,
    ) as hubspot:
        filters = [
            {"propertyName": "hs_createdate", "operator": "GTE", "value": str(since_epoch_ms)},
            {
                "propertyName": "hs_call_title",
                "operator": "CONTAINS_TOKEN",
                "value": CALL_TITLE_PREFIX,
            },
        ]
        search = {
            "filterGroups": [{"filters": filters}],
            "sorts": [{"propertyName": "hs_createdate", "direction": "DESCENDING"}],
            "limit": 1,
        }
        found = (await hubspot.post("/crm/v3/objects/calls/search", json=search)).json()
        if not found.get("results"):
            return None
        call_id = found["results"][0]["id"]
        call = await hubspot.get(
            f"/crm/v3/objects/calls/{call_id}", params={"associations": "contacts,tickets"}
        )
        linked = {
            kind: [record["id"] for record in value["results"]]
            for kind, value in call.json().get("associations", {}).items()
        }
        portal = (await hubspot.get("/account-info/v3/details")).json()
    base = f"https://{portal['uiDomain']}/contacts/{portal['portalId']}/record"
    records = [("calls", call_id)] + [
        (kind, record_id) for kind in ("contacts", "tickets") for record_id in linked.get(kind, [])
    ]
    return [f"{base}/{HUBSPOT_RECORD_PAGES[kind]}/{record_id}" for kind, record_id in records]
