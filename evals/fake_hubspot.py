import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import httpx

from voice_agent.crm.hubspot_client import HubSpotClient

FAKE_BASE_URL = "https://hubspot.fake"
SUPPORT_STAGES = [
    {"id": "1", "label": "New"},
    {"id": "2", "label": "Waiting on contact"},
    {"id": "3", "label": "Waiting on us"},
    {"id": "4", "label": "Closed"},
]
Route = Callable[[re.Match[str], dict[str, Any]], httpx.Response]


@dataclass(frozen=True)
class RecordedWrite:
    object_type: str
    caller_turn: int
    properties: dict[str, Any]


@dataclass
class FakeHubSpot:
    """An in-memory HubSpot that answers the client's requests and records every write."""

    contacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    tickets: dict[str, dict[str, Any]] = field(default_factory=dict)
    ticket_contact: dict[str, str] = field(default_factory=dict)
    writes: list[RecordedWrite] = field(default_factory=list)
    caller_turn: int = 0
    _next_id: count = field(default_factory=lambda: count(900000000001))

    def add_contact(self, contact_id: str, properties: dict[str, Any]) -> None:
        self.contacts[contact_id] = properties

    def add_ticket(self, ticket_id: str, contact_id: str, properties: dict[str, Any]) -> None:
        self.tickets[ticket_id] = properties
        self.ticket_contact[ticket_id] = contact_id

    def client(self) -> HubSpotClient:
        http = httpx.AsyncClient(base_url=FAKE_BASE_URL, transport=httpx.MockTransport(self.handle))
        return HubSpotClient(http, max_attempts=1, retry_backoff_seconds=0)

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        for method, pattern, route in self._routes():
            match = re.fullmatch(pattern, request.url.path)
            if request.method == method and match:
                return route(match, body)
        return httpx.Response(404, json={"message": f"Fake HubSpot has no {request.url.path}"})

    def _routes(self) -> list[tuple[str, str, Route]]:
        return [
            ("POST", r"/crm/v3/objects/contacts/search", self._search_contacts),
            ("POST", r"/crm/v3/objects/contacts", self._create_contact),
            ("POST", r"/crm/v3/objects/tickets", self._create_ticket),
            ("POST", r"/crm/v3/objects/tickets/batch/read", self._read_tickets),
            ("GET", r"/crm/v4/objects/contacts/(\w+)/associations/tickets", self._contact_tickets),
            ("GET", r"/crm/v3/pipelines/tickets/\w+", self._pipeline),
            ("POST", r"/crm/v3/objects/calls", self._create_call),
        ]

    def _search_contacts(self, match: re.Match[str], body: dict[str, Any]) -> httpx.Response:
        wanted = [group["filters"][0] for group in body["filterGroups"]]
        found = [
            {"id": contact_id, "properties": properties}
            for contact_id, properties in self.contacts.items()
            if any(properties.get(f["propertyName"]) == f["value"] for f in wanted)
        ]
        return httpx.Response(200, json={"results": found[:1]})

    def _record(self, object_type: str, properties: dict[str, Any]) -> str:
        self.writes.append(RecordedWrite(object_type, self.caller_turn, properties))
        return str(next(self._next_id))

    def _create_contact(self, match: re.Match[str], body: dict[str, Any]) -> httpx.Response:
        contact_id = self._record("contact", body["properties"])
        self.add_contact(contact_id, body["properties"])
        return httpx.Response(201, json={"id": contact_id})

    def _create_ticket(self, match: re.Match[str], body: dict[str, Any]) -> httpx.Response:
        ticket_id = self._record("ticket", body["properties"])
        properties = {**body["properties"], "hs_lastmodifieddate": "2026-10-29T12:00:00Z"}
        self.add_ticket(ticket_id, body["associations"][0]["to"]["id"], properties)
        return httpx.Response(201, json={"id": ticket_id})

    def _create_call(self, match: re.Match[str], body: dict[str, Any]) -> httpx.Response:
        return httpx.Response(201, json={"id": self._record("call", body["properties"])})

    def _contact_tickets(self, match: re.Match[str], body: dict[str, Any]) -> httpx.Response:
        owned = [t for t, owner in self.ticket_contact.items() if owner == match.group(1)]
        return httpx.Response(200, json={"results": [{"toObjectId": int(t)} for t in owned]})

    def _read_tickets(self, match: re.Match[str], body: dict[str, Any]) -> httpx.Response:
        found = [
            {"id": item["id"], "properties": self.tickets[item["id"]]}
            for item in body["inputs"]
            if item["id"] in self.tickets
        ]
        return httpx.Response(200, json={"results": found})

    def _pipeline(self, match: re.Match[str], body: dict[str, Any]) -> httpx.Response:
        return httpx.Response(200, json={"stages": SUPPORT_STAGES})
