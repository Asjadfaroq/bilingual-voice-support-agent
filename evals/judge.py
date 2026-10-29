import json
from dataclasses import dataclass

import anthropic
from simulated_call import CallOutcome

from voice_agent.config import Settings

RUBRIC = """\
You grade one support call handled by a voice agent for Larkspur Systems (product: Ledgerly).
The caller was simulated; HubSpot was a test double. Score three things from 1 to 5.

correctness: Did the agent do what this scenario requires (see Expected behaviour), in the
right order, without inventing facts or saving anything the caller did not clearly approve?
5 = fully right; 3 = partly right or one notable slip; 1 = wrong or harmful.

grounding: Is every product, price, policy or ticket fact the agent stated backed by a tool
output in the transcript? Saying "I don't have that information" is grounded. 5 = all
grounded; 3 = one unsupported detail; 1 = invented facts.

spoken_style: Would this sound natural when spoken by a phone agent: short sentences, at most
one question per turn, no lists or symbols, numbers and prices said as words, the caller's
language? 5 = natural; 3 = noticeably awkward; 1 = unusable as speech.

Expected behaviour for this scenario:
{judge_notes}

Transcript (tool calls and their outputs are shown in brackets):
{transcript}
"""

SCORES_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer"},
        "grounding": {"type": "integer"},
        "spoken_style": {"type": "integer"},
        "reasons": {"type": "string"},
    },
    "required": ["correctness", "grounding", "spoken_style", "reasons"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class JudgeVerdict:
    correctness: int
    grounding: int
    spoken_style: int
    reasons: str
    input_tokens: int
    output_tokens: int

    def passes(self, pass_score: int) -> bool:
        return min(self.correctness, self.grounding, self.spoken_style) >= pass_score


def format_transcript(outcome: CallOutcome) -> str:
    lines = []
    for turn, caller_line in enumerate(outcome.caller_lines):
        lines.append(f"Caller: {caller_line}")
        lines += [
            f"[tool {tool.name}({json.dumps(tool.arguments, ensure_ascii=False)}) -> {tool.output}]"
            for tool in outcome.tool_calls
            if tool.caller_turn == turn
        ]
        lines += [
            f"Agent: {text}" for agent_turn, text in outcome.agent_lines if agent_turn == turn
        ]
    return "\n".join(lines)


async def judge_call(
    client: anthropic.AsyncAnthropic, settings: Settings, outcome: CallOutcome
) -> JudgeVerdict:
    prompt = RUBRIC.format(
        judge_notes=outcome.scenario.judge_notes, transcript=format_transcript(outcome)
    )
    response = await client.messages.create(
        model=settings.judge_model,
        max_tokens=settings.judge_max_tokens,
        output_config={
            "effort": settings.judge_effort,
            "format": {"type": "json_schema", "schema": SCORES_SCHEMA},
        },
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"Judge stopped with {response.stop_reason} for {outcome.scenario.id}")
    text = "".join(block.text for block in response.content if block.type == "text")
    scores = json.loads(text)
    return JudgeVerdict(
        correctness=scores["correctness"],
        grounding=scores["grounding"],
        spoken_style=scores["spoken_style"],
        reasons=scores["reasons"],
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
