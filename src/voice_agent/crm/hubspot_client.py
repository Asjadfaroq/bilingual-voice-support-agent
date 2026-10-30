import asyncio
import re
from dataclasses import dataclass
from typing import Any

import httpx

from voice_agent.pipeline.language import ARABIC_INDIC_DIGITS

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
TICKET_TO_CONTACT = 16
CALL_TO_CONTACT = 194
CALL_TO_TICKET = 220
CONTACT_PROPERTIES = ["firstname", "lastname", "email", "phone"]
TICKET_PROPERTIES = ["subject", "hs_pipeline_stage", "hs_lastmodifieddate"]


class HubSpotError(Exception):
    pass


@dataclass(frozen=True)
class HubSpotContact:
    id: str
    first_name: str


@dataclass(frozen=True)
class HubSpotTicket:
    id: str
    subject: str
    stage_id: str
    last_modified: str


def normalize_phone_number(raw_phone: str | None, default_country_code: str) -> str | None:
    if not raw_phone:
        return None
    digits = re.sub(r"[^\d+]", "", raw_phone.translate(ARABIC_INDIC_DIGITS))
    if digits.startswith("+"):
        return digits
    if digits.startswith("00"):
        return f"+{digits[2:]}"
    if digits.startswith("0"):
        return f"+{default_country_code}{digits[1:]}"
    return f"+{digits}"


def error_message(response: httpx.Response) -> str:
    if "json" in response.headers.get("content-type", ""):
        return str(response.json().get("message", response.text))
    return response.text


def association(to_id: str, type_id: int) -> dict[str, Any]:
    category = {"associationCategory": "HUBSPOT_DEFINED", "associationTypeId": type_id}
    return {"to": {"id": to_id}, "types": [category]}


class HubSpotClient:
    def __init__(self, http: httpx.AsyncClient, max_attempts: int, retry_backoff_seconds: float):
        self._http = http
        self._max_attempts = max_attempts
        self._retry_backoff_seconds = retry_backoff_seconds

    async def aclose(self) -> None:
        await self._http.aclose()

    async def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        failure = ""
        for attempt in range(self._max_attempts):
            if attempt:
                await asyncio.sleep(self._retry_backoff_seconds * 2 ** (attempt - 1))
            try:
                response = await self._http.request(method, path, json=body)
            except httpx.TransportError as error:
                failure = f"{type(error).__name__}: {error}"
                continue
            if response.status_code in RETRYABLE_STATUS_CODES:
                failure = f"HTTP {response.status_code}"
                continue
            if response.is_error:
                raise HubSpotError(
                    f"HubSpot {method} {path} failed: HTTP {response.status_code}: "
                    f"{error_message(response)}"
                )
            return response.json() if response.content else None
        raise HubSpotError(
            f"HubSpot {method} {path} failed after {self._max_attempts} attempts: {failure}"
        )

    async def find_contact(self, *, phone: str | None, email: str | None) -> HubSpotContact | None:
        filter_groups = [
            {"filters": [{"propertyName": name, "operator": "EQ", "value": value}]}
            for name, value in (("phone", phone), ("email", email))
            if value
        ]
        if not filter_groups:
            return None
        body = {"filterGroups": filter_groups, "properties": CONTACT_PROPERTIES, "limit": 1}
        found = await self.request("POST", "/crm/v3/objects/contacts/search", body)
        if not found["results"]:
            return None
        match = found["results"][0]
        return HubSpotContact(id=match["id"], first_name=match["properties"].get("firstname") or "")

    async def create_contact(
        self, *, first_name: str, last_name: str, email: str | None, phone: str | None
    ) -> HubSpotContact:
        properties = {
            "firstname": first_name,
            "lastname": last_name,
            "email": email,
            "phone": phone,
        }
        body = {"properties": {name: value for name, value in properties.items() if value}}
        created = await self.request("POST", "/crm/v3/objects/contacts", body)
        return HubSpotContact(id=created["id"], first_name=first_name)

    async def create_ticket(self, *, contact_id: str, properties: dict[str, str]) -> str:
        body = {
            "properties": properties,
            "associations": [association(contact_id, TICKET_TO_CONTACT)],
        }
        created = await self.request("POST", "/crm/v3/objects/tickets", body)
        return str(created["id"])

    async def tickets_for_contact(self, contact_id: str) -> list[HubSpotTicket]:
        linked = await self.request(
            "GET", f"/crm/v4/objects/contacts/{contact_id}/associations/tickets"
        )
        ticket_ids = [{"id": str(link["toObjectId"])} for link in linked["results"]]
        if not ticket_ids:
            return []
        body = {"inputs": ticket_ids, "properties": TICKET_PROPERTIES}
        tickets = await self.request("POST", "/crm/v3/objects/tickets/batch/read", body)
        return [
            HubSpotTicket(
                id=ticket["id"],
                subject=ticket["properties"].get("subject") or "",
                stage_id=ticket["properties"].get("hs_pipeline_stage") or "",
                last_modified=ticket["properties"].get("hs_lastmodifieddate") or "",
            )
            for ticket in tickets["results"]
        ]

    async def ticket_stage_labels(self, pipeline_id: str) -> dict[str, str]:
        pipeline = await self.request("GET", f"/crm/v3/pipelines/tickets/{pipeline_id}")
        return {stage["id"]: stage["label"] for stage in pipeline["stages"]}

    async def create_call(
        self, *, properties: dict[str, str], contact_id: str | None, ticket_ids: list[str]
    ) -> str:
        associations = [association(contact_id, CALL_TO_CONTACT)] if contact_id else []
        associations += [association(ticket_id, CALL_TO_TICKET) for ticket_id in ticket_ids]
        body = {"properties": properties, "associations": associations}
        created = await self.request("POST", "/crm/v3/objects/calls", body)
        return str(created["id"])
