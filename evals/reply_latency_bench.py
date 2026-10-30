import argparse
import asyncio
import re
import time
from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any

from bench_report import percentiles, save_results
from bench_setup import read_manifest
from fake_hubspot import FakeHubSpot
from livekit.agents import Agent, AgentSession, MetricsCollectedEvent, ModelSettings
from livekit.agents.metrics import LLMMetrics
from psycopg_pool import AsyncConnectionPool

from voice_agent.config import Settings, get_settings
from voice_agent.conversation.call_state import CallState, call_resources_for
from voice_agent.conversation.knowledge_first import KnowledgeSearchMode
from voice_agent.conversation.support_agent import SupportAgent
from voice_agent.knowledge.ingest import TextEmbedder, build_sentence_embedder
from voice_agent.pipeline.components import build_llm
from voice_agent.pipeline.language import language_settings_for

SSML_BREAK = re.compile(r"\s*<break[^>]*/>\s*")
TEXT_STAGES = ("first_text", "answer_text", "llm_time_to_first_token", "reply_filter_hold")


class TimedSupportAgent(SupportAgent):
    """Records when each speech's first text leaves the agent, which is what a voice gets."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.first_text_times: list[float] = []

    async def transcription_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncGenerator[str, None]:
        first = True
        async for chunk in Agent.default.transcription_node(self, text, model_settings):
            if first and str(chunk).strip():
                self.first_text_times.append(time.perf_counter())
                first = False
            yield chunk


def utterance_texts(settings: Settings, language: str) -> list[str]:
    return [SSML_BREAK.sub(" ", entry["text"]) for entry in read_manifest(settings, language)]


async def bench_text_turns(
    settings: Settings,
    pool: AsyncConnectionPool,
    embed: TextEmbedder,
    mode: KnowledgeSearchMode,
    language: str,
) -> list[dict[str, float | None]]:
    call, holds, first_tokens = CallState(), [], []
    resources = call_resources_for(settings, pool, embed, None, FakeHubSpot().client(), call)
    agent = TimedSupportAgent(
        language_settings_for(settings), None, call,
        kb_search_mode=mode, report_reply_filter_hold=holds.append,
    )  # fmt: skip
    session = AgentSession(llm=build_llm(settings), userdata=resources)

    def record_first_token(event: MetricsCollectedEvent) -> None:
        if isinstance(event.metrics, LLMMetrics) and event.metrics.ttft >= 0:
            first_tokens.append(event.metrics.ttft)

    session.on("metrics_collected", record_first_token)
    await session.start(agent)
    # The greeting is not an answer to any turn, so it finishes before the clock starts.
    if session.current_speech is not None:
        await session.current_speech.wait_for_playout()
    texts, turns = utterance_texts(settings, language), []
    for index in range(settings.bench_turns_per_language):
        agent.first_text_times.clear()
        holds.clear()
        first_tokens.clear()
        started = time.perf_counter()
        await session.run(user_input=texts[index % len(texts)])
        speeches = agent.first_text_times
        turns.append({
            "first_text": speeches[0] - started if speeches else None,
            "answer_text": speeches[-1] - started if speeches else None,
            "llm_time_to_first_token": first_tokens[0] if first_tokens else None,
            "reply_filter_hold": holds[0] if holds else None,
        })  # fmt: skip
        print(f"{mode} {language} {index + 1}: {turns[-1]}")
    await session.aclose()
    return turns


def summarise_text_turns(turns: list[dict[str, float | None]]) -> dict[str, object]:
    summary: dict[str, object] = {"turns": len(turns)}
    for stage in TEXT_STAGES:
        values = [value for turn in turns if (value := turn[stage]) is not None]
        if values:
            summary[stage] = percentiles(values)
    return summary


async def run_text_bench(modes: list[KnowledgeSearchMode], languages: list[str]) -> None:
    settings = get_settings()
    embed = build_sentence_embedder(settings.embedding_model)
    async with AsyncConnectionPool(settings.database_url, min_size=1, max_size=2) as pool:
        for mode in modes:
            by_language = {
                language: await bench_text_turns(settings, pool, embed, mode, language)
                for language in languages
            }
            save_results("text_mode", mode, {
                "languages": {lang: summarise_text_turns(t) for lang, t in by_language.items()},
                "turns": by_language,
            })  # fmt: skip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time replies in text mode, without voice.")
    parser.add_argument("--modes", nargs="+", default=["tool", "every_turn"])
    parser.add_argument("--languages", nargs="+", default=["en", "ar"])
    arguments = parser.parse_args()
    asyncio.run(run_text_bench(arguments.modes, arguments.languages))
