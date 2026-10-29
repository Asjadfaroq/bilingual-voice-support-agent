from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from voice_agent.config import Settings

TOKENS_PER_PRICE_UNIT = 1_000_000


def cost_usd(settings: Settings, model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = settings.model_prices_usd_per_million_tokens[model]
    return (input_tokens * input_price + output_tokens * output_price) / TOKENS_PER_PRICE_UNIT


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
