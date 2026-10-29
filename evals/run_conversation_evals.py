import asyncio
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
from checks import CheckResult, run_checks
from judge import JudgeVerdict, judge_call
from livekit.agents.testing import fake_job_context
from psycopg_pool import AsyncConnectionPool
from simulated_call import CallOutcome, Scenario, run_simulated_call

from voice_agent.config import Settings, get_settings
from voice_agent.knowledge.ingest import TextEmbedder, build_sentence_embedder

EVALS_DIRECTORY = Path(__file__).parent
SCENARIOS_PATH = EVALS_DIRECTORY / "golden" / "conversations.jsonl"
RESULTS_PATH = EVALS_DIRECTORY / "results" / "latest.json"
TOKENS_PER_PRICE_UNIT = 1_000_000


def load_scenarios() -> list[Scenario]:
    lines = SCENARIOS_PATH.read_text(encoding="utf-8").splitlines()
    return [Scenario(**json.loads(line)) for line in lines if line.strip()]


def cost_usd(settings: Settings, model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = settings.model_prices_usd_per_million_tokens[model]
    return (input_tokens * input_price + output_tokens * output_price) / TOKENS_PER_PRICE_UNIT


async def evaluate_scenario(
    scenario: Scenario,
    settings: Settings,
    pool: AsyncConnectionPool,
    embed: TextEmbedder,
    judge_client: anthropic.AsyncAnthropic,
) -> dict[str, Any]:
    outcome: CallOutcome = await run_simulated_call(scenario, settings, pool, embed)
    checks: list[CheckResult] = run_checks(outcome)
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


def summarise(results: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    def pass_counts(key: str) -> dict[str, dict[str, int]]:
        groups: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
        for scenario in results:
            groups[scenario[key]]["total"] += 1
            groups[scenario[key]]["passed"] += scenario["passed"]
        return dict(sorted(groups.items()))

    check_names = results[0]["checks"].keys()
    agent_cost = sum(r["agent_cost_usd"] for r in results)
    judge_cost = sum(r["judge_cost_usd"] for r in results)
    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "judge_pass_score": settings.judge_pass_score,
        "scenarios": len(results),
        "passed": sum(r["passed"] for r in results),
        "by_category": pass_counts("category"),
        "by_language": pass_counts("language"),
        "check_pass_rates": {
            name: sum(r["checks"][name] for r in results) / len(results) for name in check_names
        },
        "judge_means": {
            score: sum(r["judge"][score] for r in results) / len(results)
            for score in ("correctness", "grounding", "spoken_style")
        },
        "cost_usd": {"agent": agent_cost, "judge": judge_cost, "total": agent_cost + judge_cost},
    }


def print_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    print(f"{'category':<26}{'en':>8}{'ar':>8}{'total':>9}")
    for category, totals in summary["by_category"].items():
        by_language = {
            language: [
                r["passed"]
                for r in results
                if r["category"] == category and r["language"] == language
            ]
            for language in ("en", "ar")
        }
        cells = [f"{sum(v)}/{len(v)}" if v else "-" for v in by_language.values()]
        print(f"{category:<26}{cells[0]:>8}{cells[1]:>8}{totals['passed']:>5}/{totals['total']}")
    total_cost = summary["cost_usd"]["total"]
    print(f"\npassed {summary['passed']}/{summary['scenarios']}; cost ${total_cost:.4f}")
    for scenario in results:
        for failure in scenario["failures"]:
            print(f"FAIL {scenario['id']}: {failure}")


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
