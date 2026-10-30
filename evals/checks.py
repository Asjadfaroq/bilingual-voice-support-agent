import re
from dataclasses import dataclass

from simulated_call import CallOutcome

from voice_agent.conversation.confirmation import is_explicit_yes
from voice_agent.pipeline.language import prose_language
from voice_agent.pipeline.spoken_text import to_spoken_text
from voice_agent.privacy.pii import DIGIT

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


def check_result(name: str, problems: list[str]) -> CheckResult:
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
    return check_result("no_write_without_yes", problems)


def expected_reply_language(outcome: CallOutcome, turn: int, min_letters: int) -> str:
    # Emails, ticket ids and product names say nothing about the language a line is in.
    return prose_language(outcome.caller_lines[turn], min_letters) or outcome.scenario.language


def check_reply_language(outcome: CallOutcome, min_letters: int) -> CheckResult:
    problems = []
    for turn, text in outcome.agent_lines:
        expected = expected_reply_language(outcome, turn, min_letters)
        actual = prose_language(text, min_letters)
        if actual is not None and actual != expected:
            problems.append(f"turn {turn}: expected {expected}, got {actual}: {text[:80]!r}")
    return check_result("reply_language", problems)


def check_spoken_format(outcome: CallOutcome) -> CheckResult:
    problems = [
        f"turn {turn}: {pattern_name} in {text[:80]!r}"
        for turn, text in outcome.agent_lines
        for pattern_name, pattern in (("markdown", MARKDOWN), ("url", URL))
        if pattern.search(text)
    ]
    return check_result("no_markdown_or_urls", problems)


def check_numbers_as_words(outcome: CallOutcome, min_letters: int) -> CheckResult:
    # Checked on what the TTS is given, after the spoken text layer has run.
    problems = []
    for turn, text in outcome.agent_lines:
        language = expected_reply_language(outcome, turn, min_letters)
        spoken = to_spoken_text(text, language)
        found = (ANY_DIGIT if language == "ar" else DIGIT_AMOUNT).search(spoken)
        if found:
            problems.append(f"turn {turn}: digits {found.group(0)!r} in {spoken[:80]!r}")
    return check_result("numbers_as_words", problems)


def check_one_question_per_turn(outcome: CallOutcome) -> CheckResult:
    problems = []
    for turn in range(len(outcome.caller_lines)):
        said = " ".join(text for agent_turn, text in outcome.agent_lines if agent_turn == turn)
        if (questions := said.count("?") + said.count("؟")) > 1:
            problems.append(f"turn {turn}: {questions} questions in {said[:120]!r}")
    return check_result("one_question_per_turn", problems)
