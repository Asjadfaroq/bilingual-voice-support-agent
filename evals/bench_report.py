import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from bench_caller import AgentObservations

RESULTS_PATH = Path("evals/results/latency.json")

STAGES = (
    "end_to_end",
    "end_of_turn_delay",
    "transcription_delay",
    "llm_time_to_first_token",
    "reply_filter_hold",
    "tts_time_to_first_byte",
    "caller_heard_first_audio",
    "caller_heard_answer",
)


@dataclass(frozen=True)
class BenchTurn:
    utterance_id: str
    language: str
    cut_off: bool
    answered: bool
    stages: dict[str, float | None]


def first_after(times: list[float], moment: float) -> float | None:
    return next((time for time in times if time > moment), None)


def describe_turn(
    utterance_id: str, language: str, observations: AgentObservations, started: float, ended: float
) -> BenchTurn:
    onsets = observations.audio_onsets
    first_audio = first_after(onsets, ended)
    thinking = [time for time, state in observations.states if state == "thinking" and time > ended]
    answer_audio = first_after(onsets, thinking[-1]) if thinking else first_audio
    in_turn = [measurement for time, measurement in observations.measurements if time > started]
    turn = next((m for m in in_turn if m["kind"] == "turn"), {})
    hold = next((m["seconds"] for m in in_turn if m["kind"] == "reply_filter_hold"), None)
    node_first_token = turn.get("llm_time_to_first_token")
    stages: dict[str, float | None] = {
        "end_to_end": turn.get("end_to_end"),
        "end_of_turn_delay": turn.get("end_of_turn_delay"),
        "transcription_delay": turn.get("transcription_delay"),
        # LiveKit times the first token out of llm_node, which is after the reply filter.
        "llm_time_to_first_token": (
            node_first_token - hold
            if node_first_token is not None and hold is not None
            else node_first_token
        ),
        "reply_filter_hold": hold,
        "tts_time_to_first_byte": turn.get("tts_time_to_first_byte"),
        "caller_heard_first_audio": first_audio - ended if first_audio else None,
        "caller_heard_answer": answer_audio - ended if answer_audio else None,
    }
    cut_off = any(started < onset < ended for onset in onsets)
    return BenchTurn(utterance_id, language, cut_off, first_audio is not None, stages)  # type: ignore[arg-type]


def percentiles(values: list[float]) -> dict[str, float | int]:
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
        "n": len(values),
    }


def summarise_language(turns: list[BenchTurn]) -> dict[str, object]:
    stage_summary = {}
    for stage in STAGES:
        values = [value for turn in turns if (value := turn.stages[stage]) is not None]
        if values:
            stage_summary[stage] = percentiles(values)
    return {
        "turns": len(turns),
        "answered": sum(turn.answered for turn in turns),
        "cut_offs": sum(turn.cut_off for turn in turns),
        "stages_seconds": stage_summary,
    }


def describe_machine() -> dict[str, str]:
    model, cpu = platform.platform(), platform.processor()
    if platform.system() == "Darwin":
        model, cpu = (
            subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
            for name in ("hw.model", "machdep.cpu.brand_string")
        )
    return {
        "model": model,
        "os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
        "machine": platform.machine(),
        "cpu": cpu,
        "python": platform.python_version(),
    }


def turns_as_json(turns: list[BenchTurn]) -> list[dict[str, object]]:
    return [asdict(turn) for turn in turns]


def save_results(section: str, label: str, run: dict[str, object]) -> None:
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    results["machine"] = describe_machine()
    results.setdefault(section, {})[label] = run
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
