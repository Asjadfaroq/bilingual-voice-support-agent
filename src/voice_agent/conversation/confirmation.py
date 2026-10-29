import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from livekit.agents.llm import ChatContext, ChatMessage

ARABIC_DIACRITICS = re.compile("[\u064b-\u0652\u0640]")
# Alef forms, taa marbuta and alef maqsura are spelled interchangeably in transcripts.
ARABIC_LETTER_VARIANTS = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0629": "\u0647",
        "\u0649": "\u064a",
    }
)

AFFIRMATIVE_WORDS = frozenset(
    {"yes", "yeah", "yep", "yup", "sure", "correct", "right", "ok", "okay", "confirm", "confirmed"}
    | {"نعم", "ايوه", "ايوا", "اي", "ايه", "تمام", "اكيد", "صح", "طيب", "موافق", "اوكي"}
)
AFFIRMATIVE_PHRASES = ("go ahead", "please do", "do it", "that's right", "يا ريت")
# Any of these makes the answer a no or a hesitation, never an explicit yes.
NEGATING_WORDS = frozenset(
    {"no", "nope", "not", "don't", "dont", "wait", "cancel", "but", "actually", "wrong", "stop"}
    | {"لا", "لأ", "مو", "مش", "ليس", "لحظه", "انتظر", "بس", "الغ", "الغي", "غلط", "خطا", "مب"}
)

PerformWrite = Callable[[], Awaitable[str]]

NOTHING_TO_CONFIRM = (
    "There is no request waiting for confirmation. Do not tell the caller anything was saved."
)
WAITING_FOR_ANSWER = (
    "The caller has not answered the read-back yet. Read it back and wait for their answer."
)
NOT_CONFIRMED = (
    "The caller did not give a clear yes, so nothing was saved. Tell them nothing was "
    "created and ask what they would like to change."
)


@dataclass(frozen=True)
class PendingWrite:
    read_back: str
    perform: PerformWrite
    asked_after_message_id: str | None


def normalize_words(text: str) -> str:
    without_marks = ARABIC_DIACRITICS.sub("", text.lower()).translate(ARABIC_LETTER_VARIANTS)
    return " ".join(re.findall(r"[\w']+", without_marks))


def is_explicit_yes(answer: str) -> bool:
    words = normalize_words(answer)
    tokens = set(words.split())
    if tokens & NEGATING_WORDS:
        return False
    return bool(tokens & AFFIRMATIVE_WORDS) or any(
        phrase in words for phrase in AFFIRMATIVE_PHRASES
    )


def caller_messages(history: ChatContext) -> list[ChatMessage]:
    return [item for item in history.items if isinstance(item, ChatMessage) and item.role == "user"]


def propose_write(read_back: str, perform: PerformWrite, history: ChatContext) -> PendingWrite:
    messages = caller_messages(history)
    return PendingWrite(read_back, perform, messages[-1].id if messages else None)


def answer_to_read_back(history: ChatContext, pending: PendingWrite) -> ChatMessage | None:
    messages = caller_messages(history)
    asked_ids = [message.id for message in messages]
    if pending.asked_after_message_id is None:
        return messages[0] if messages else None
    if pending.asked_after_message_id not in asked_ids:
        return None
    position = asked_ids.index(pending.asked_after_message_id)
    return messages[position + 1] if position + 1 < len(messages) else None


@dataclass(frozen=True)
class ConfirmationResult:
    reply_for_llm: str
    pending_resolved: bool


async def confirm_before_write(
    pending: PendingWrite | None, history: ChatContext
) -> ConfirmationResult:
    if pending is None:
        return ConfirmationResult(NOTHING_TO_CONFIRM, pending_resolved=True)
    answer = answer_to_read_back(history, pending)
    if answer is None:
        return ConfirmationResult(WAITING_FOR_ANSWER, pending_resolved=False)
    # Only the caller's very next turn counts, so a later yes can never approve an old read-back.
    if not is_explicit_yes(answer.text_content or ""):
        return ConfirmationResult(NOT_CONFIRMED, pending_resolved=True)
    return ConfirmationResult(await pending.perform(), pending_resolved=True)


def spell_for_read_back(value: str) -> str:
    spoken_symbols = {"@": "at", ".": "dot", "-": "dash", "_": "underscore", "+": "plus"}
    return " ".join(spoken_symbols.get(character, character) for character in value)
