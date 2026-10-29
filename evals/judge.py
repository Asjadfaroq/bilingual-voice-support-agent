import json
from dataclasses import dataclass

import anthropic
from checks import expected_reply_language
from simulated_call import CallOutcome

from voice_agent.config import Settings
from voice_agent.pipeline.spoken_text import to_spoken_text

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

How this agent saves things: find_or_create_contact (for a new contact), create_ticket and
request_human_callback never save anything. They say a read-back to the caller themselves and
return nothing; only record_caller_answer saves, and only after the caller's clear yes.

Expected behaviour for this scenario:
{judge_notes}

Transcript (tool calls and their outputs are shown in brackets; each Agent line is exactly
the text the voice was given to speak, after numbers were turned into words):
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


def format_transcript(outcome: CallOutcome, min_letters: int) -> str:
    lines = []
    for turn, caller_line in enumerate(outcome.caller_lines):
        lines.append(f"Caller: {caller_line}")
        lines += [
            f"[tool {tool.name}({json.dumps(tool.arguments, ensure_ascii=False)}) -> "
            f"{tool.output or '(returned nothing)'}]"
            for tool in outcome.tool_calls
            if tool.caller_turn == turn
        ]
        language = expected_reply_language(outcome, turn, min_letters)
        lines += [
            f"Agent: {to_spoken_text(text, language)}"
            for agent_turn, text in outcome.agent_lines
            if agent_turn == turn
        ]
    return "\n".join(lines)


async def judge_call(
    client: anthropic.AsyncAnthropic, settings: Settings, outcome: CallOutcome
) -> JudgeVerdict:
    prompt = RUBRIC.format(
        judge_notes=outcome.scenario.judge_notes,
        transcript=format_transcript(outcome, settings.reply_language_min_letters),
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
