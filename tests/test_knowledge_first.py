from collections.abc import AsyncGenerator
from typing import Any, cast

import pytest
from livekit.agents.llm import ChatContext, FunctionCallOutput

from voice_agent.conversation import knowledge_first
from voice_agent.conversation.call_state import CallState
from voice_agent.conversation.confirmation import PendingWrite
from voice_agent.conversation.instructions import (
    KNOWLEDGE_BASE_UNAVAILABLE,
    NO_KNOWLEDGE_BASE_MATCH,
)
from voice_agent.conversation.support_agent import SupportAgent
from voice_agent.conversation.tools import search_knowledge_base
from voice_agent.pipeline.language import LanguageSettings

LANGUAGE_SETTINGS = LanguageSettings("sarah", {}, 0.8, 6)
GROWTH_PASSAGE = "[ledgerly-product-faq#plans] Plans\nGrowth is 149 SAR a month."


def test_the_search_tool_is_offered_only_when_the_model_searches_itself() -> None:
    tool_mode = SupportAgent(LANGUAGE_SETTINGS, None, CallState(), kb_search_mode="tool")
    every_turn = SupportAgent(LANGUAGE_SETTINGS, None, CallState(), kb_search_mode="every_turn")

    assert search_knowledge_base in tool_mode.tools
    assert search_knowledge_base not in every_turn.tools


async def test_searching_every_turn_grounds_the_reply_on_the_callers_own_words(
    call_context: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    searched: list[str] = []

    async def look_up_knowledge(question: str, resources: Any) -> str:
        searched.append(question)
        return GROWTH_PASSAGE

    monkeypatch.setattr(knowledge_first, "look_up_knowledge", look_up_knowledge)
    contexts: list[ChatContext] = []

    async def start_reply(chat_ctx: ChatContext) -> AsyncGenerator[str, None]:
        contexts.append(chat_ctx)
        yield "Growth is 149 SAR a month."

    resources = call_context.userdata
    reply = knowledge_first.reply_with_knowledge(
        start_reply, ChatContext.empty(), "How much is Growth?", resources
    )

    assert [chunk async for chunk in reply] == ["Growth is 149 SAR a month."]
    assert searched == ["How much is Growth?"]
    assert GROWTH_PASSAGE in (contexts[0].items[-1].text_content or "")
    assert resources.call.knowledge_lookups == [("How much is Growth?", GROWTH_PASSAGE)]


async def test_a_turn_the_knowledge_base_has_nothing_for_is_not_told_to_offer_a_ticket(
    call_context: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def look_up_knowledge(question: str, resources: Any) -> str:
        return NO_KNOWLEDGE_BASE_MATCH

    monkeypatch.setattr(knowledge_first, "look_up_knowledge", look_up_knowledge)
    contexts: list[ChatContext] = []

    async def start_reply(chat_ctx: ChatContext) -> AsyncGenerator[str, None]:
        contexts.append(chat_ctx)
        yield "Saved."

    reply = knowledge_first.reply_with_knowledge(
        start_reply, ChatContext.empty(), "yes", call_context.userdata
    )

    assert [chunk async for chunk in reply] == ["Saved."]
    assert contexts[0].items[-1].text_content == knowledge_first.NOTHING_FOUND_THIS_TURN


def test_only_a_new_caller_turn_without_a_pending_read_back_is_searched() -> None:
    caller_turn = ChatContext.empty()
    caller_turn.add_message(role="user", content="How much is Growth?")
    after_tool = caller_turn.copy()
    after_tool.items.append(
        FunctionCallOutput(
            call_id="1", name="get_ticket_status", output="Ticket 3344", is_error=False
        )
    )
    answering_read_back = CallState(pending_write=cast(PendingWrite, object()))

    assert knowledge_first.searches_before_reply(caller_turn, CallState())
    assert not knowledge_first.searches_before_reply(after_tool, CallState())
    assert not knowledge_first.searches_before_reply(caller_turn, answering_read_back)


def test_later_turns_still_see_what_earlier_answers_were_based_on() -> None:
    office_hours = "[office-hours-and-holidays#core-hours] Core hours\nRiyadh: 08:00 to 18:00."
    call = CallState(
        knowledge_lookups=[
            ("What are your office hours?", office_hours),
            ("CSV export fails, please open a ticket.", NO_KNOWLEDGE_BASE_MATCH),
            ("Do you have office hours on Friday?", office_hours),
            ("yes", KNOWLEDGE_BASE_UNAVAILABLE),
        ]
    )

    notes = knowledge_first.earlier_knowledge_notes(call, "every_turn")

    assert notes == [knowledge_first.EARLIER_KNOWLEDGE.format(passages=office_hours)]
    assert knowledge_first.earlier_knowledge_notes(call, "tool") == []


def test_a_call_with_nothing_found_yet_adds_no_earlier_knowledge() -> None:
    call = CallState(knowledge_lookups=[("Hello", NO_KNOWLEDGE_BASE_MATCH)])

    assert knowledge_first.earlier_knowledge_notes(call, "every_turn") == []
