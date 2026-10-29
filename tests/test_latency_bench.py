import pytest
from bench_caller import AgentObservations
from bench_report import describe_turn

from voice_agent.metrics.turn_latency import reply_filter_hold_reporter

CALLER_STARTED, CALLER_ENDED = 10.0, 12.0
TURN_MEASUREMENT = {"kind": "turn", "end_to_end": 2.0, "llm_time_to_first_token": 1.1}


def observed(*audio_onsets: float) -> AgentObservations:
    return AgentObservations(
        audio_onsets=list(audio_onsets),
        states=[(13.5, "thinking"), (14.1, "speaking")],
        measurements=[
            (14.0, {"kind": "reply_filter_hold", "seconds": 0.2}),
            (16.0, TURN_MEASUREMENT),
        ],
    )


def test_agent_audio_while_the_caller_is_still_speaking_is_a_cut_off() -> None:
    turn = describe_turn("en-pause", "en", observed(11.0, 14.2), CALLER_STARTED, CALLER_ENDED)

    assert turn.cut_off


def test_a_reply_after_the_caller_finished_is_timed_from_the_end_of_their_speech() -> None:
    turn = describe_turn("en-price", "en", observed(14.2), CALLER_STARTED, CALLER_ENDED)

    assert not turn.cut_off
    assert turn.stages["caller_heard_first_audio"] == pytest.approx(2.2)
    # LiveKit's first token is timed after the reply filter, so the filter's hold is taken out.
    assert turn.stages["llm_time_to_first_token"] == pytest.approx(0.9)
    assert turn.stages["reply_filter_hold"] == 0.2


def test_the_filter_hold_is_logged_without_publishing_outside_the_bench(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="voice_agent.metrics.turn_latency"):
        reply_filter_hold_reporter(None)(0.25)

    assert caplog.messages == ["reply filter held the first sentence for 0.25s"]
