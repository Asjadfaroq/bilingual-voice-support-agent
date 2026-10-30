import httpx

from voice_agent.config import Settings


async def logged_call_links(settings: Settings, email: str, since: str) -> list[str] | None:
    headers = {"Authorization": f"Bearer {settings.hubspot_access_token.get_secret_value()}"}
    async with httpx.AsyncClient(
        base_url=settings.hubspot_base_url,
        headers=headers,
        timeout=settings.provider_timeout_seconds,
    ) as hubspot:
        by_email = {"propertyName": "email", "operator": "EQ", "value": email}
        search = {"filterGroups": [{"filters": [by_email]}]}
        found = (await hubspot.post("/crm/v3/objects/contacts/search", json=search)).json()
        if not found.get("results"):
            return None
        contact_id = found["results"][0]["id"]
        contact = await hubspot.get(
            f"/crm/v3/objects/contacts/{contact_id}", params={"associations": "tickets,calls"}
        )
        linked = {
            kind: [record["id"] for record in value["results"]]
            for kind, value in contact.json().get("associations", {}).items()
        }
        calls = [
            (await hubspot.get(f"/crm/v3/objects/calls/{i}")).json()
            for i in linked.get("calls", [])
        ]
        if not any(call["createdAt"] >= since for call in calls):
            return None
        portal = (await hubspot.get("/account-info/v3/details")).json()
    base = f"https://{portal['uiDomain']}/contacts/{portal['portalId']}/record"
    tickets = [f"{base}/0-5/{ticket_id}" for ticket_id in linked.get("tickets", [])]
    return [f"{base}/0-1/{contact_id}", *tickets, *(f"{base}/0-48/{call['id']}" for call in calls)]
