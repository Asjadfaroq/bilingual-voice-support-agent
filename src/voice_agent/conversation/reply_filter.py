import re
import time
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from typing import Literal

from livekit.agents import FlushSentinel
from livekit.agents.llm import ChatChunk, ChatContext

from voice_agent.conversation.system_notes import with_system_notes
from voice_agent.pipeline.language import CallerLanguage, language_from_script, prose_text
from voice_agent.pipeline.spoken_text import to_spoken_text

ReplyChunk = ChatChunk | str | FlushSentinel
StartReply = Callable[[ChatContext], AsyncGenerator[ReplyChunk, None]]
SentenceVerdict = Literal["keep", "meta", "wrong_language"]
ReportHold = Callable[[float], None]

# A reply is in the language most of its letters are written in.
MAJORITY_SCRIPT = 0.5
SENTENCE_END = re.compile(r"[.!?؟\n]+(?:\s+|$)")
# Sentences about the caller in the third person, or about the reply language, are the
# model thinking aloud, never something to say to the caller.
META_SENTENCE = re.compile(
    r"\b(?:the caller|the user)\b|\b(?:identify|ask|offer|tell|help|greet|thank)\s+them\b"
    r"|\b(?:continu|repl|respond|answer|switch|speak)\w*\s+in\s+"
    r"(?:English|Arabic)\b|المتصل|المستخدم|(?:سأكمل|سأرد|سأتحدث|سأجيب)\s+(?:باللغة\s+)?"
    r"(?:ال)?(?:عربية|إنجليزية)",
    re.IGNORECASE,
)


def ignore_hold(seconds: float) -> None:
    # Text-mode evals and unit tests have no caller waiting on audio.
    return None


def prose_language(text: str, min_letters: int) -> CallerLanguage | None:
    prose = prose_text(text)
    if sum(character.isalpha() for character in prose) < min_letters:
        return None
    return language_from_script(prose, MAJORITY_SCRIPT)


def judge_sentence(sentence: str, expected: CallerLanguage, min_letters: int) -> SentenceVerdict:
    if META_SENTENCE.search(sentence):
        return "meta"
    language = prose_language(sentence, min_letters)
    return "wrong_language" if language is not None and language != expected else "keep"


class SentenceSplitter:
    def __init__(self) -> None:
        self._pending = ""

    def push(self, text: str) -> list[str]:
        self._pending += text
        ends = [match.end() for match in SENTENCE_END.finditer(self._pending)]
        if not ends:
            return []
        complete, self._pending = self._pending[: ends[-1]], self._pending[ends[-1] :]
        starts = [0, *ends[:-1]]
        return [complete[start:end].strip() for start, end in zip(starts, ends, strict=True)]

    def flush(self) -> str:
        remainder, self._pending = self._pending.strip(), ""
        return remainder


def split_text(chunk: ReplyChunk) -> tuple[str, ReplyChunk | None]:
    if isinstance(chunk, str):
        return chunk, None
    if not isinstance(chunk, ChatChunk) or chunk.delta is None or not chunk.delta.content:
        return "", chunk
    if not chunk.delta.tool_calls and chunk.usage is None:
        return chunk.delta.content, None
    without_text = chunk.model_copy(
        update={"delta": chunk.delta.model_copy(update={"content": None})}
    )
    return chunk.delta.content, without_text


async def guarded_reply(
    start_reply: StartReply,
    chat_ctx: ChatContext,
    language: CallerLanguage,
    retry_note: str,
    min_letters: int,
    report_hold: ReportHold,
) -> AsyncGenerator[ReplyChunk, None]:
    stream, splitter = start_reply(chat_ctx), SentenceSplitter()
    retried = spoke = False
    # The hold is how long the caller waits on the filter: from the model's first text to the
    # first sentence let through, including a regenerated reply.
    first_text_at: float | None = None

    def report_first_sentence() -> None:
        if not spoke and first_text_at is not None:
            report_hold(time.perf_counter() - first_text_at)

    while True:
        restart = False
        async for chunk in stream:
            text, rest_of_chunk = split_text(chunk)
            if text and first_text_at is None:
                first_text_at = time.perf_counter()
            sentences = splitter.push(text)
            if rest_of_chunk is not None and text == "" and not sentences:
                sentences = [remainder] if (remainder := splitter.flush()) else []
            for sentence in sentences:
                verdict = judge_sentence(sentence, language, min_letters)
                if verdict == "wrong_language" and not spoke and not retried:
                    restart = True
                    break
                if verdict == "keep":
                    report_first_sentence()
                    spoke = True
                    yield sentence + " "
            if restart:
                break
            if rest_of_chunk is not None:
                spoke = True
                yield rest_of_chunk
        if not restart:
            remainder = splitter.flush()
            if remainder and judge_sentence(remainder, language, min_letters) == "keep":
                report_first_sentence()
                yield remainder
            return
        # A reply that opens in the wrong language is regenerated once with a firmer note.
        await stream.aclose()
        retried, splitter = True, SentenceSplitter()
        stream = start_reply(with_system_notes(chat_ctx, retry_note))


async def spoken_sentences(
    text: AsyncIterable[str], language_of: Callable[[str], CallerLanguage]
) -> AsyncGenerator[str, None]:
    splitter = SentenceSplitter()
    async for chunk in text:
        for sentence in splitter.push(chunk):
            yield to_spoken_text(sentence, language_of(sentence)) + " "
    if remainder := splitter.flush():
        yield to_spoken_text(remainder, language_of(remainder))
