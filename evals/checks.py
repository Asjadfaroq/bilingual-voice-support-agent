import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from simulated_call import CallOutcome

from voice_agent.conversation.confirmation import is_explicit_yes, spell_for_read_back
from voice_agent.pipeline.language import language_from_script
from voice_agent.privacy.pii import DIGIT

# A reply is in the language most of its letters are written in, so a spelled email in an
# Arabic reply still counts as Arabic.
MAJORITY_SCRIPT = 0.5
MARKDOWN = re.compile(r"\*\*|__|`|^\s*#|^\s*[-*•]\s|^\s*\d+\.\s", re.MULTILINE)
URL = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|sa|io|net|org)\b", re.IGNORECASE)
ANY_DIGIT = re.compile(DIGIT)
# English replies may say identifiers ("invoice 4412"), but prices, times and durations in words.
DIGIT_AMOUNT = re.compile(
    r"\d[\d,.]*\s*(?:riyals?|sar|hours?|days?|minutes?|am|pm|%|percent|users?|invoices?)\b"
    r"|\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str = ""


def result(name: str, problems: list[str]) -> CheckResult:
    return CheckResult(name, not problems, "; ".join(problems))


def check_no_write_without_yes(outcome: CallOutcome) -> CheckResult:
    problems = []
    for write in outcome.writes:
        answer = outcome.caller_lines[write.caller_turn]
        saved_this_turn = any(
            tool.name == "record_caller_answer" and tool.caller_turn == write.caller_turn
            for tool in outcome.tool_calls
        )
        if not (is_explicit_yes(answer) and saved_this_turn):
            problems.append(f"{write.object_type} written after caller said {answer!r}")
    return result("no_write_without_yes", problems)


def expected_reply_language(outcome: CallOutcome, turn: int) -> str:
    caller_line = outcome.caller_lines[turn]
    return language_from_script(caller_line, MAJORITY_SCRIPT) or outcome.scenario.language


def check_reply_language(outcome: CallOutcome) -> CheckResult:
    problems = []
    for turn, text in outcome.agent_lines:
        expected = expected_reply_language(outcome, turn)
        actual = language_from_script(text, MAJORITY_SCRIPT)
        if actual != expected:
            problems.append(f"turn {turn}: expected {expected}, got {actual}: {text[:80]!r}")
    return result("reply_language", problems)


def check_spoken_format(outcome: CallOutcome) -> CheckResult:
    problems = [
        f"turn {turn}: {pattern_name} in {text[:80]!r}"
        for turn, text in outcome.agent_lines
        for pattern_name, pattern in (("markdown", MARKDOWN), ("url", URL))
        if pattern.search(text)
    ]
    return result("no_markdown_or_urls", problems)


def check_numbers_as_words(outcome: CallOutcome) -> CheckResult:
    problems = []
    for turn, text in outcome.agent_lines:
        arabic = expected_reply_language(outcome, turn) == "ar"
        found = (ANY_DIGIT if arabic else DIGIT_AMOUNT).search(text)
        if found:
            problems.append(f"turn {turn}: digits {found.group(0)!r} in {text[:80]!r}")
    return result("numbers_as_words", problems)


def arguments_match(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(
        actual.get(name) in wanted if isinstance(wanted, list) else actual.get(name) == wanted
        for name, wanted in expected.items()
    )


def check_tools(outcome: CallOutcome, expect: dict[str, Any]) -> CheckResult:
    problems = []
    for wanted in expect.get("tools_called", []):
        calls = [tool for tool in outcome.tool_calls if tool.name == wanted["name"]]
        if not any(arguments_match(tool.arguments, wanted.get("args", {})) for tool in calls):
            seen = [tool.arguments for tool in calls]
            problems.append(f"expected {wanted['name']} with {wanted.get('args', {})}, saw {seen}")
    called = {tool.name for tool in outcome.tool_calls}
    problems += [f"called {name}" for name in expect.get("tools_not_called", []) if name in called]
    return result("tools_and_arguments", problems)


def check_writes(outcome: CallOutcome, expect: dict[str, Any]) -> CheckResult:
    counts = Counter(write.object_type for write in outcome.writes)
    problems = [
        f"expected {wanted} {object_type} writes, got {counts[object_type]}"
        for object_type, wanted in expect.get("writes", {}).items()
        if counts[object_type] != wanted
    ]
    wanted = expect.get("ticket_properties", {})
    tickets = [write.properties for write in outcome.writes if write.object_type == "ticket"]
    if wanted and not any(arguments_match(ticket, wanted) for ticket in tickets):
        problems.append(f"no ticket with {wanted}, tickets were {tickets}")
    return result("writes", problems)


def spelled_pattern(email: str) -> re.Pattern[str]:
    tokens = spell_for_read_back(email.split("@")[0]).split()
    return re.compile(r"[\s,]+".join(re.escape(token) for token in tokens), re.IGNORECASE)


def check_scenario_specifics(outcome: CallOutcome, expect: dict[str, Any]) -> CheckResult:
    agent_text = "\n".join(text for _, text in outcome.agent_lines)
    problems = [
        f"said {phrase!r}" for phrase in expect.get("must_not_say", []) if phrase in agent_text
    ]
    email = expect.get("spelled_email")
    if email and not spelled_pattern(email).search(agent_text):
        problems.append(f"{email} was not spelled out before saving")
    if expect.get("ticket_after_details"):
        tickets = [tool.caller_turn for tool in outcome.tool_calls if tool.name == "create_ticket"]
        if outcome.details_turn is None:
            problems.append("the agent never asked for the caller's details")
        elif tickets and tickets[0] < outcome.details_turn:
            problems.append(
                f"ticket proposed in turn {tickets[0]}, before the details in turn "
                f"{outcome.details_turn}"
            )
    return result("scenario_specifics", problems)


def run_checks(outcome: CallOutcome) -> list[CheckResult]:
    expect = outcome.scenario.expect
    return [
        check_no_write_without_yes(outcome),
        check_reply_language(outcome),
        check_spoken_format(outcome),
        check_numbers_as_words(outcome),
        check_tools(outcome, expect),
        check_writes(outcome, expect),
        check_scenario_specifics(outcome, expect),
    ]
