import re
from dataclasses import dataclass, field
from typing import Any

from livekit.agents.llm import ChatContext

from voice_agent.conversation.call_state import CallState
from voice_agent.conversation.confirmation import answer_to_read_back

DEFAULT_YES = {"en": "Yes, that's right.", "ar": "نعم، صحيح."}
CLOSING_LINE = {"en": "No, that's all, thank you.", "ar": "لا، هذا كل شيء، شكرًا."}
ANYTHING_ELSE = re.compile(
    r"anything else|something else|شيء آخر|أي شيء|أي مساعدة أخرى", re.IGNORECASE
)


@dataclass(frozen=True)
class Scenario:
    id: str
    language: str
    category: str
    turns: list[str]
    expect: dict[str, Any]
    judge_notes: str
    answers: dict[str, str] = field(default_factory=dict)
    # Answers to the agent's clarifying questions, given in order, one per question.
    clarifications: list[str] = field(default_factory=list)
    setup: dict[str, Any] = field(default_factory=dict)


class SimulatedCaller:
    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._turns = list(scenario.turns)
        self._clarifications = list(scenario.clarifications)
        self.finished = False

    def next_line(self, call: CallState, history: ChatContext, last_agent_text: str) -> str | None:
        # Answers the read-back actually pending, so the call follows the agent's real flow.
        pending = call.pending_write
        language = self._scenario.language
        if pending is not None and answer_to_read_back(history, pending) is None:
            return self._scenario.answers.get(pending.kind, DEFAULT_YES[language])
        if self.finished:
            return None
        asked_question = last_agent_text.rstrip().endswith(("?", "؟"))
        if asked_question and ANYTHING_ELSE.search(last_agent_text) and not self._turns:
            self.finished = True
            return CLOSING_LINE[language]
        if asked_question and self._clarifications:
            return self._clarifications.pop(0)
        return self._turns.pop(0) if self._turns else None
