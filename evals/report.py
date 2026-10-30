from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from voice_agent.config import Settings
from voice_agent.metrics.tracing import model_cost


def cost_usd(settings: Settings, model: str, input_tokens: int, output_tokens: int) -> float:
    cost = model_cost(
        settings.model_prices_usd_per_million_tokens, model, input_tokens, output_tokens
    )
    if cost is None:
        raise ValueError(
            f"No price is configured for {model} in model_prices_usd_per_million_tokens"
        )
    return cost["total"]


def summarise(results: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    def pass_counts(key: str) -> dict[str, dict[str, int]]:
        groups: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
        for scenario in results:
            groups[scenario[key]]["total"] += 1
            groups[scenario[key]]["passed"] += scenario["passed"]
        return dict(sorted(groups.items()))

    check_names = results[0]["checks"].keys()
    agent_cost = sum(scenario["agent_cost_usd"] for scenario in results)
    judge_cost = sum(scenario["judge_cost_usd"] for scenario in results)
    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "agent_model": settings.llm_model,
        "judge_model": settings.judge_model,
        "judge_pass_score": settings.judge_pass_score,
        "scenarios": len(results),
        "passed": sum(scenario["passed"] for scenario in results),
        "by_category": pass_counts("category"),
        "by_language": pass_counts("language"),
        "check_pass_rates": {
            name: sum(scenario["checks"][name] for scenario in results) / len(results)
            for name in check_names
        },
        "judge_means": {
            score: sum(scenario["judge"][score] for scenario in results) / len(results)
            for score in ("correctness", "grounding", "spoken_style")
        },
        "cost_usd": {"agent": agent_cost, "judge": judge_cost, "total": agent_cost + judge_cost},
    }


def print_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    print(f"{'category':<26}{'en':>8}{'ar':>8}{'total':>9}")
    for category, totals in summary["by_category"].items():
        by_language = {
            language: [
                scenario["passed"]
                for scenario in results
                if scenario["category"] == category and scenario["language"] == language
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
