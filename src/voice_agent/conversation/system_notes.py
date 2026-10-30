from livekit.agents.llm import ChatContext, ChatMessage


def with_system_notes(chat_ctx: ChatContext, *notes: str) -> ChatContext:
    """A copy of the context with the notes added to the agent's instructions."""
    # LiveKit sends only a leading system message to Claude as the system prompt and turns any
    # later one into "<instructions>" text inside the caller's turn, where the model took notes
    # such as "Do not ask for their name" for instructions from the caller and refused them.
    chat_ctx = chat_ctx.copy()
    if not notes:
        return chat_ctx
    first = chat_ctx.items[0] if chat_ctx.items else None
    if isinstance(first, ChatMessage) and first.role == "system":
        chat_ctx.items[0] = first.model_copy(update={"content": [*first.content, *notes]})
    else:
        chat_ctx.items.insert(0, ChatMessage(role="system", content=list(notes)))
    return chat_ctx
