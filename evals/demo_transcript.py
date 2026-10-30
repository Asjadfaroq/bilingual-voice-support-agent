import asyncio
import re
import shutil
import sys
import textwrap

from livekit import rtc
from livekit.agents.types import ATTRIBUTE_TRANSCRIPTION_FINAL, TOPIC_TRANSCRIPTION

LABEL_WIDTH = 9
WORD_WITH_SPACING = re.compile(r"\s+|\S+\s*")
CALLER_LABEL = "\033[1;36mCaller\033[0m"
AGENT_LABEL = "\033[1;32mAgent\033[0m"
# The ANSI codes take no columns, so labels are padded by their visible length.
VISIBLE_LABEL_LENGTH = {CALLER_LABEL: len("Caller"), AGENT_LABEL: len("Agent")}


class LiveTranscript:
    """Prints the call as the agent publishes it: its own words as it speaks them, and the
    caller's turns as its speech-to-text heard them."""

    def __init__(self, room: rtc.Room, caller_identity: str) -> None:
        self._caller_identity = caller_identity
        self._column = 0
        self._tasks: set[asyncio.Task[None]] = set()
        room.register_text_stream_handler(TOPIC_TRANSCRIPTION, self._on_stream)

    def heading(self, text: str) -> None:
        print(f"\n\033[1m{text}\033[0m\n", flush=True)

    def _on_stream(self, reader: rtc.TextStreamReader, participant_identity: str) -> None:
        if participant_identity == self._caller_identity:
            task = asyncio.ensure_future(self._show_caller_turn(reader))
        else:
            task = asyncio.ensure_future(self._show_agent_speech(reader))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _show_caller_turn(self, reader: rtc.TextStreamReader) -> None:
        text = (await reader.read_all()).strip()
        # Interim transcripts are replaced by the final one; only that is printed.
        if text and reader.info.attributes.get(ATTRIBUTE_TRANSCRIPTION_FINAL) == "true":
            self._end_line()
            width = shutil.get_terminal_size().columns - LABEL_WIDTH
            lines = textwrap.wrap(text, width) or [text]
            print(self._label(CALLER_LABEL) + lines[0])
            for line in lines[1:]:
                print(" " * LABEL_WIDTH + line)
            print(flush=True)

    async def _show_agent_speech(self, reader: rtc.TextStreamReader) -> None:
        started = False
        async for chunk in reader:
            if not chunk.strip() and not started:
                continue
            if not started:
                self._end_line()
                sys.stdout.write(self._label(AGENT_LABEL))
                self._column, started = LABEL_WIDTH, True
            self._write_wrapped(chunk)
        if started:
            self._end_line()
            print(flush=True)

    def _write_wrapped(self, chunk: str) -> None:
        width = shutil.get_terminal_size().columns
        # Each word keeps the spacing the agent streamed after it.
        for word in WORD_WITH_SPACING.findall(chunk):
            if word.isspace() and self._column == LABEL_WIDTH:
                continue
            if self._column + len(word.rstrip()) > width and self._column > LABEL_WIDTH:
                sys.stdout.write("\n" + " " * LABEL_WIDTH)
                self._column = LABEL_WIDTH
                word = word.lstrip()
            sys.stdout.write(word)
            self._column += len(word)
        sys.stdout.flush()

    def _end_line(self) -> None:
        if self._column:
            sys.stdout.write("\n")
            self._column = 0

    def _label(self, label: str) -> str:
        return label + " " * (LABEL_WIDTH - VISIBLE_LABEL_LENGTH[label])
