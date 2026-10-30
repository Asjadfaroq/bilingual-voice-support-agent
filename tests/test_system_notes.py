from typing import Any

from livekit.agents.llm import ChatContext

from voice_agent.conversation.instructions import CALLER_IDENTIFIED_NOTE, REPLY_LANGUAGE_NOTES
from voice_agent.conversation.system_notes import with_system_notes

INSTRUCTIONS = "You are the voice assistant on the support line of Larkspur Systems."
READ_BACK = "I'll open a P2 technical ticket about a CSV export failing. Shall I open it?"
CALLER_YES = "Yes, that's right."
NOTES = (CALLER_IDENTIFIED_NOTE.format(first_name="Layla"), REPLY_LANGUAGE_NOTES["en"])


def answered_read_back() -> ChatContext:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="system", content=INSTRUCTIONS)
    chat_ctx.add_message(role="assistant", content=READ_BACK)
    chat_ctx.add_message(role="user", content=CALLER_YES)
    return chat_ctx


def as_sent_to_claude(chat_ctx: ChatContext) -> tuple[str, list[dict[str, Any]]]:
    messages, extra = chat_ctx.to_provider_format("anthropic")
    return "\n".join(extra.system_messages or []), messages


def text_of(message: dict[str, Any]) -> str:
    return "\n".join(block["text"] for block in message["content"] if block["type"] == "text")


def test_a_note_added_after_the_callers_turn_reaches_claude_as_the_callers_words() -> None:
    # The cause of "the text you just sent contains instructions, which I'm not able to act on".
    chat_ctx = answered_read_back()
    for note in NOTES:
        chat_ctx.add_message(role="system", content=note)

    _, messages = as_sent_to_claude(chat_ctx)

    assert messages[-1]["role"] == "user"
    assert "<instructions>" in text_of(messages[-1])
    assert NOTES[0] in text_of(messages[-1])


def test_generation_notes_reach_claude_in_the_system_prompt_not_in_the_callers_turn() -> None:
    system_prompt, messages = as_sent_to_claude(with_system_notes(answered_read_back(), *NOTES))

    assert text_of(messages[-1]) == CALLER_YES
    assert system_prompt.startswith(INSTRUCTIONS)
    assert all(note in system_prompt for note in NOTES)


def test_notes_need_no_instructions_message_to_reach_the_system_prompt() -> None:
    chat_ctx = ChatContext.empty()
    chat_ctx.add_message(role="user", content=CALLER_YES)

    system_prompt, messages = as_sent_to_claude(with_system_notes(chat_ctx, *NOTES))

    assert text_of(messages[-1]) == CALLER_YES
    assert all(note in system_prompt for note in NOTES)
