from collections.abc import AsyncGenerator

from livekit.agents.llm import ChatContext, ChatMessage

from voice_agent.conversation.reply_filter import guarded_reply

RETRY_NOTE = "Reply in Arabic."


def scripted(*replies: str) -> tuple[list[ChatContext], object]:
    contexts: list[ChatContext] = []
    remaining = list(replies)

    async def start_reply(chat_ctx: ChatContext) -> AsyncGenerator[str, None]:
        contexts.append(chat_ctx)
        for word in remaining.pop(0).split(" "):
            yield word + " "

    return contexts, start_reply


async def spoken(start_reply: object, language: str) -> str:
    stream = guarded_reply(start_reply, ChatContext.empty(), language, RETRY_NOTE, 6)  # type: ignore[arg-type]
    return "".join([chunk async for chunk in stream]).strip()  # type: ignore[misc]


async def test_the_model_thinking_aloud_about_the_caller_is_never_spoken() -> None:
    _, start_reply = scripted(
        "The caller wants a human, so I'll offer a callback. I can arrange a callback for you."
    )

    assert await spoken(start_reply, "en") == "I can arrange a callback for you."


async def test_saying_which_language_the_reply_is_in_is_never_spoken() -> None:
    _, start_reply = scripted("Sure, I'll continue in English. Growth is forty-nine riyals.")

    assert await spoken(start_reply, "en") == "Growth is forty-nine riyals."


async def test_an_english_reply_to_an_arabic_caller_is_regenerated_in_arabic() -> None:
    contexts, start_reply = scripted(
        "To check that ticket, I first need to identify you.", "لحظة، أتحقق من تذكرتك."
    )

    assert await spoken(start_reply, "ar") == "لحظة، أتحقق من تذكرتك."
    notes = [i.text_content for i in contexts[-1].items if isinstance(i, ChatMessage)]
    assert RETRY_NOTE in notes


async def test_an_english_sentence_later_in_an_arabic_reply_is_dropped() -> None:
    _, start_reply = scripted("لحظة، أتحقق لك. I need to identify them first. وجدت حسابك.")

    assert await spoken(start_reply, "ar") == "لحظة، أتحقق لك. وجدت حسابك."


async def test_names_emails_and_numbers_do_not_make_an_arabic_reply_english() -> None:
    reply = "سأضيفك باسم Sara Ahmed والبريد sara.ahmed@example.com على Ledgerly."
    contexts, start_reply = scripted(reply)

    assert await spoken(start_reply, "ar") == reply
    assert len(contexts) == 1
