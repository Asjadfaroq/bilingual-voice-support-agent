import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anthropic
from checks import (
    CheckResult,
    check_no_write_without_yes,
    check_numbers_as_words,
    check_one_question_per_turn,
    check_reply_language,
    check_spoken_format,
)
from judge import JudgeVerdict, judge_call
from livekit.agents.testing import fake_job_context
from psycopg_pool import AsyncConnectionPool
from report import cost_usd, print_report, summarise
from scenario_checks import check_scenario_specifics, check_tools, check_writes
from simulated_call import CallOutcome, run_simulated_call
from simulated_caller import Scenario

from voice_agent.config import Settings, get_settings
from voice_agent.knowledge.ingest import TextEmbedder, build_sentence_embedder

EVALS_DIRECTORY = Path(__file__).parent
SCENARIOS_PATH = EVALS_DIRECTORY / "golden" / "conversations.jsonl"
RESULTS_PATH = EVALS_DIRECTORY / "results" / "latest.json"


def load_scenarios() -> list[Scenario]:
    lines = SCENARIOS_PATH.read_text(encoding="utf-8").splitlines()
    return [Scenario(**json.loads(line)) for line in lines if line.strip()]


def run_checks(outcome: CallOutcome, min_letters: int) -> list[CheckResult]:
    expect = outcome.scenario.expect
    return [
        check_no_write_without_yes(outcome),
        check_reply_language(outcome, min_letters),
        check_spoken_format(outcome),
        check_numbers_as_words(outcome, min_letters),
        check_one_question_per_turn(outcome),
        check_tools(outcome, expect),
        check_writes(outcome, expect),
        check_scenario_specifics(outcome, expect),
    ]


async def evaluate_scenario(
    scenario: Scenario,
    settings: Settings,
    pool: AsyncConnectionPool,
    embed: TextEmbedder,
    judge_client: anthropic.AsyncAnthropic,
) -> dict[str, Any]:
    outcome: CallOutcome = await run_simulated_call(scenario, settings, pool, embed)
    checks: list[CheckResult] = run_checks(outcome, settings.reply_language_min_letters)
    verdict: JudgeVerdict = await judge_call(judge_client, settings, outcome)
    failures = [f"{check.name}: {check.reason}" for check in checks if not check.passed]
    if not verdict.passes(settings.judge_pass_score):
        failures.append(
            f"judge: correctness {verdict.correctness}, grounding {verdict.grounding}, "
            f"spoken style {verdict.spoken_style}: {verdict.reasons}"
        )
    return {
        "id": scenario.id,
        "language": scenario.language,
        "category": scenario.category,
        "passed": not failures,
        "failures": failures,
        "checks": {check.name: check.passed for check in checks},
        "judge": {k: v for k, v in asdict(verdict).items() if not k.endswith("_tokens")},
        "agent_cost_usd": cost_usd(
            settings, settings.llm_model, outcome.input_tokens, outcome.output_tokens
        ),
        "judge_cost_usd": cost_usd(
            settings, settings.judge_model, verdict.input_tokens, verdict.output_tokens
        ),
        "transcript": {"caller": outcome.caller_lines, "agent": outcome.agent_lines},
        "tool_calls": [asdict(tool) for tool in outcome.tool_calls],
        "writes": [asdict(write) for write in outcome.writes],
    }


async def run_all() -> None:
    settings = get_settings()
    scenarios = load_scenarios()
    embed = build_sentence_embedder(settings.embedding_model)
    judge_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
    limit = asyncio.Semaphore(settings.eval_concurrency)
    async with AsyncConnectionPool(
        settings.database_url, max_size=settings.eval_concurrency
    ) as pool:

        async def evaluate(scenario: Scenario) -> dict[str, Any]:
            async with limit:
                return await evaluate_scenario(scenario, settings, pool, embed, judge_client)

        results = await asyncio.gather(*(evaluate(scenario) for scenario in scenarios))
    summary = summarise(results, settings)
    RESULTS_PATH.write_text(
        json.dumps({**summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_report(summary, results)


if __name__ == "__main__":
    with fake_job_context():
        asyncio.run(run_all())
