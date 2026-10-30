import re
from collections import Counter
from typing import Any

from checks import CheckResult, check_result
from simulated_call import CallOutcome

from voice_agent.conversation.confirmation import spell_for_read_back


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
    return check_result("tools_and_arguments", problems)


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
    return check_result("writes", problems)


def spelled_pattern(email: str) -> re.Pattern[str]:
    tokens = spell_for_read_back(email.split("@")[0]).split()
    return re.compile(r"[\s,]+".join(re.escape(token) for token in tokens), re.IGNORECASE)


def check_scenario_specifics(outcome: CallOutcome, expect: dict[str, Any]) -> CheckResult:
    agent_text = "\n".join(text for _, text in outcome.agent_lines)
    problems = [
        f"said {phrase!r}" for phrase in expect.get("must_not_say", []) if phrase in agent_text
    ]
    if (
        contact_id := expect.get("identified_contact")
    ) and outcome.identified_contact_id != contact_id:
        problems.append(
            f"caller identified as {outcome.identified_contact_id}, expected {contact_id}"
        )
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
    return check_result("scenario_specifics", problems)
