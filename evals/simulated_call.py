import json
from dataclasses import dataclass, field
from typing import Any

from fake_hubspot import FakeHubSpot, RecordedWrite
from livekit.agents import AgentSession, MetricsCollectedEvent
from livekit.agents.llm import ChatContext
from livekit.agents.metrics import LLMMetrics
from psycopg_pool import AsyncConnectionPool

from voice_agent.config import Settings
from voice_agent.conversation.confirmation import answer_to_read_back
from voice_agent.conversation.support_agent import LanguageSettings, SupportAgent
from voice_agent.conversation.tools import CallResources, CallState
from voice_agent.knowledge.ingest import TextEmbedder
from voice_agent.pipeline.components import build_llm

DEFAULT_YES = {"en": "Yes, that's right.", "ar": "نعم، صحيح."}


@dataclass(frozen=True)
class Scenario:
    id: str
    language: str
    category: str
    turns: list[str]
    expect: dict[str, Any]
    judge_notes: str
    answers: dict[str, str] = field(default_factory=dict)
    details: str | None = None
    setup: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    output: str
    caller_turn: int


@dataclass
class CallOutcome:
    scenario: Scenario
    caller_lines: list[str] = field(default_factory=list)
    agent_lines: list[tuple[int, str]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    writes: list[RecordedWrite] = field(default_factory=list)
    details_turn: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class SimulatedCaller:
    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._turns = list(scenario.turns)
        self.details_given = False

    def next_line(self, call: CallState, history: ChatContext, last_agent_text: str) -> str | None:
        # Answers the read-back actually pending, so the call follows the agent's real flow.
        pending = call.pending_write
        if pending is not None and answer_to_read_back(history, pending) is None:
            return self._scenario.answers.get(pending.kind, DEFAULT_YES[self._scenario.language])
        asked_question = last_agent_text.rstrip().endswith(("?", "؟"))
        if self._scenario.details and not self.details_given and asked_question:
            self.details_given = True
            return self._scenario.details
        return self._turns.pop(0) if self._turns else None


def seed_fake_hubspot(setup: dict[str, Any]) -> FakeHubSpot:
    hubspot = FakeHubSpot()
    for contact in setup.get("contacts", []):
        hubspot.add_contact(contact["id"], contact["properties"])
    for ticket in setup.get("tickets", []):
        hubspot.add_ticket(ticket["id"], ticket["contact_id"], ticket["properties"])
    return hubspot


async def run_simulated_call(
    scenario: Scenario,
    settings: Settings,
    knowledge_pool: AsyncConnectionPool,
    embed: TextEmbedder,
) -> CallOutcome:
    hubspot = seed_fake_hubspot(scenario.setup)
    crm = hubspot.client()
    call = CallState(caller_phone=scenario.setup.get("caller_phone"))
    resources = CallResources(
        knowledge_pool, embed, None, settings.kb_top_k, settings.kb_min_similarity, crm,
        settings.hubspot_ticket_pipeline_id, settings.hubspot_new_ticket_stage_id,
        settings.ticket_status_limit, settings.default_country_calling_code, call,
    )  # fmt: skip
    agent = SupportAgent(
        LanguageSettings(settings.tts_voice_id, {}, settings.script_dominance_threshold), None
    )
    outcome = CallOutcome(scenario)

    def count_tokens(event: MetricsCollectedEvent) -> None:
        if isinstance(event.metrics, LLMMetrics):
            outcome.input_tokens += event.metrics.prompt_tokens
            outcome.output_tokens += event.metrics.completion_tokens

    session = AgentSession[CallResources](llm=build_llm(settings), userdata=resources)
    session.on("metrics_collected", count_tokens)
    await session.start(agent)
    caller = SimulatedCaller(scenario)
    last_agent_text = ""
    for turn in range(settings.eval_max_caller_turns):
        line = caller.next_line(call, session.history, last_agent_text)
        if line is None:
            break
        if line == scenario.details:
            outcome.details_turn = turn
        hubspot.caller_turn = turn
        outcome.caller_lines.append(line)
        result = await session.run(user_input=line)
        outputs = {
            e.item.call_id: e.item.output for e in result.events if e.type == "function_call_output"
        }
        for event in result.events:
            if event.type == "message" and event.item.text_content:
                last_agent_text = event.item.text_content
                outcome.agent_lines.append((turn, last_agent_text))
            elif event.type == "function_call":
                arguments = json.loads(event.item.arguments or "{}")
                output = outputs.get(event.item.call_id, "")
                outcome.tool_calls.append(ToolCall(event.item.name, arguments, output, turn))
        if any(tool.name == "end_call" for tool in outcome.tool_calls):
            break
    await session.aclose()
    await crm.aclose()
    outcome.writes = list(hubspot.writes)
    return outcome
