from typing import Any

from checks import (
    check_no_write_without_yes,
    check_numbers_as_words,
    check_reply_language,
    check_scenario_specifics,
    check_spoken_format,
    check_tools,
    check_writes,
)
from fake_hubspot import FakeHubSpot, RecordedWrite
from simulated_call import CallOutcome, Scenario, ToolCall


def outcome_with(
    caller_lines: list[str],
    agent_lines: list[tuple[int, str]],
    tool_calls: list[ToolCall] | None = None,
    writes: list[RecordedWrite] | None = None,
    language: str = "en",
) -> CallOutcome:
    scenario = Scenario(
        id="s", language=language, category="c", turns=[], expect={}, judge_notes=""
    )
    return CallOutcome(
        scenario,
        caller_lines=caller_lines,
        agent_lines=agent_lines,
        tool_calls=tool_calls or [],
        writes=writes or [],
    )


def save(turn: int) -> ToolCall:
    return ToolCall("record_caller_answer", {}, "", turn)


def test_write_in_the_turn_the_caller_said_yes_passes() -> None:
    outcome = outcome_with(
        ["open a ticket", "Yes, please"], [], [save(1)], [RecordedWrite("ticket", 1, {})]
    )

    assert check_no_write_without_yes(outcome).passed


def test_write_after_a_no_or_without_the_save_step_fails() -> None:
    after_no = outcome_with(["open it", "No"], [], [save(1)], [RecordedWrite("ticket", 1, {})])
    without_save = outcome_with(["open it", "Yes"], [], [], [RecordedWrite("ticket", 1, {})])

    assert not check_no_write_without_yes(after_no).passed
    assert not check_no_write_without_yes(without_save).passed


def test_reply_must_be_in_the_language_of_the_caller_turn_it_answers() -> None:
    switched = outcome_with(
        ["How much is Growth?", "وكم سعر المبتدئ؟"],
        [(0, "Growth is one hundred forty-nine riyals."), (1, "Starter is forty-nine riyals.")],
    )

    check = check_reply_language(switched)

    assert not check.passed and "turn 1: expected ar, got en" in check.reason


def test_arabic_reply_with_a_spelled_email_is_still_arabic() -> None:
    outcome = outcome_with(
        ["سجلني"], [(0, "سأسجلك بالبريد k h a l e d at e x a m p l e dot c o m، هل هذا صحيح؟")]
    )

    assert check_reply_language(outcome).passed


def test_markdown_and_web_addresses_are_not_speakable() -> None:
    outcome = outcome_with(["hi"], [(0, "**Plans:**\n- Starter"), (0, "See ledgerly.com")])

    assert not check_spoken_format(outcome).passed


def test_arabic_digits_and_english_prices_in_digits_fail_but_identifiers_pass() -> None:
    arabic = outcome_with(["كم السعر؟"], [(0, "السعر 49 ريالًا")], language="ar")
    english_price = outcome_with(["price?"], [(0, "It costs 149 riyals a month.")])
    english_invoice = outcome_with(["ticket"], [(0, "I'll note invoice 4412.")])

    assert not check_numbers_as_words(arabic).passed
    assert not check_numbers_as_words(english_price).passed
    assert check_numbers_as_words(english_invoice).passed


def test_expected_tool_arguments_accept_any_listed_value() -> None:
    outcome = outcome_with(["hi"], [], [ToolCall("create_ticket", {"category": "bug"}, "", 0)])
    expect: dict[str, Any] = {
        "tools_called": [{"name": "create_ticket", "args": {"category": ["bug", "question"]}}],
        "tools_not_called": ["request_human_callback"],
    }

    assert check_tools(outcome, expect).passed


def test_write_counts_and_ticket_properties_are_compared_exactly() -> None:
    outcome = outcome_with(
        ["x"], [], writes=[RecordedWrite("ticket", 0, {"hs_ticket_priority": "HIGH"})]
    )

    check = check_writes(
        outcome, {"writes": {"ticket": 1}, "ticket_properties": {"hs_ticket_priority": "URGENT"}}
    )

    assert not check.passed and "no ticket with" in check.reason


def test_spelled_email_and_ticket_after_details_are_checked() -> None:
    outcome = outcome_with(
        ["I need a ticket", "I'm Lina, lina.k@example.com, export fails"],
        [(1, "I have l, i, n, a, dot, k at example dot com. Is that right?")],
        [ToolCall("create_ticket", {}, "", 1)],
    )
    outcome.details_turn = 1

    assert check_scenario_specifics(
        outcome, {"spelled_email": "lina.k@example.com", "ticket_after_details": True}
    ).passed


def test_fake_hubspot_records_writes_with_the_caller_turn() -> None:
    hubspot = FakeHubSpot()
    hubspot.caller_turn = 2

    hubspot._create_contact(None, {"properties": {"email": "a@b.c"}})  # type: ignore[arg-type]

    assert hubspot.writes == [RecordedWrite("contact", 2, {"email": "a@b.c"})]
