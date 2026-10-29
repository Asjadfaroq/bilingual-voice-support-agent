import json
from dataclasses import dataclass, field
from typing import Any

from fake_hubspot import FakeHubSpot, RecordedWrite
from livekit.agents import AgentSession, MetricsCollectedEvent
from livekit.agents.llm import ChatMessage, FunctionCall, FunctionCallOutput
from livekit.agents.metrics import LLMMetrics
from psycopg_pool import AsyncConnectionPool
from simulated_caller import Scenario, SimulatedCaller

from voice_agent.config import Settings
from voice_agent.conversation.call_state import CallResources, CallState
from voice_agent.conversation.instructions import GREETING
from voice_agent.conversation.support_agent import SupportAgent
from voice_agent.conversation.tools import identify_caller_by_line
from voice_agent.crm.hubspot_client import HubSpotClient
from voice_agent.knowledge.ingest import TextEmbedder
from voice_agent.pipeline.components import build_llm
from voice_agent.pipeline.language import LanguageSettings


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
    identified_contact_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def record_turn(outcome: CallOutcome, items: list[Any], turn: int) -> str | None:
    outputs = {item.call_id: item.output for item in items if isinstance(item, FunctionCallOutput)}
    last_agent_text = None
    for item in items:
        if isinstance(item, ChatMessage) and item.role == "assistant" and item.text_content:
            if item.text_content != GREETING:
                last_agent_text = item.text_content
                outcome.agent_lines.append((turn, item.text_content))
        elif isinstance(item, FunctionCall):
            arguments = json.loads(item.arguments or "{}")
            outcome.tool_calls.append(
                ToolCall(item.name, arguments, outputs.get(item.call_id, ""), turn)
            )
    return last_agent_text


def seed_fake_hubspot(setup: dict[str, Any]) -> FakeHubSpot:
    hubspot = FakeHubSpot()
    for contact in setup.get("contacts", []):
        hubspot.add_contact(contact["id"], contact["properties"])
    for ticket in setup.get("tickets", []):
        hubspot.add_ticket(ticket["id"], ticket["contact_id"], ticket["properties"])
    return hubspot


def build_call_resources(
    settings: Settings,
    knowledge_pool: AsyncConnectionPool,
    embed: TextEmbedder,
    crm: HubSpotClient,
    call: CallState,
) -> CallResources:
    return CallResources(
        knowledge_pool, embed, None, settings.kb_top_k, settings.kb_min_similarity, crm,
        settings.hubspot_ticket_pipeline_id, settings.hubspot_new_ticket_stage_id,
        settings.ticket_status_limit, settings.default_country_calling_code, call,
    )  # fmt: skip


def text_mode_language_settings(settings: Settings) -> LanguageSettings:
    return LanguageSettings(
        settings.tts_voice_id, {}, settings.script_dominance_threshold,
        settings.reply_language_min_letters,
    )  # fmt: skip


async def run_simulated_call(
    scenario: Scenario,
    settings: Settings,
    knowledge_pool: AsyncConnectionPool,
    embed: TextEmbedder,
) -> CallOutcome:
    hubspot = seed_fake_hubspot(scenario.setup)
    crm = hubspot.client()
    call = CallState(caller_phone=scenario.setup.get("caller_phone"))
    resources = build_call_resources(settings, knowledge_pool, embed, crm, call)
    agent = SupportAgent(
        text_mode_language_settings(settings), None, call, kb_search_mode=settings.kb_search_mode
    )
    await identify_caller_by_line(resources)
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
        if line in scenario.clarifications and outcome.details_turn is None:
            outcome.details_turn = turn
        hubspot.caller_turn = turn
        outcome.caller_lines.append(line)
        seen, searched = len(session.history.items), len(call.knowledge_lookups)
        await session.run(user_input=line)
        # Read-backs are said by the tools themselves, so the turn is read from the history.
        while session.current_speech is not None:
            await session.current_speech.wait_for_playout()
        last_agent_text = (
            record_turn(outcome, session.history.items[seen:], turn) or last_agent_text
        )
        # Searching before the model is the same knowledge base search, shown as the tool call.
        outcome.tool_calls += [
            ToolCall("search_knowledge_base", {"question": question}, passages, turn)
            for question, passages in call.knowledge_lookups[searched:]
        ]
        if any(tool.name == "end_call" for tool in outcome.tool_calls):
            break
    await session.aclose()
    await crm.aclose()
    outcome.writes = list(hubspot.writes)
    outcome.identified_contact_id = call.contact.id if call.contact else None
    return outcome
