import pytest

from voice_agent.pipeline.language import (
    ReplyVoice,
    choose_reply_voice,
    detect_caller_language,
    language_from_stt,
)

DOMINANCE = 0.8


def test_arabic_transcript_is_arabic_even_when_the_stt_tags_it_english() -> None:
    assert detect_caller_language("هل تدعمون الدفع عن طريق مدى؟", "en", DOMINANCE) == "ar"


def test_english_transcript_is_english_even_when_the_stt_tags_it_arabic() -> None:
    assert detect_caller_language("Do you support Apple Pay?", "ar", DOMINANCE) == "en"


@pytest.mark.parametrize(("stt_language", "expected"), [("ar", "ar"), ("en", "en")])
def test_mixed_script_turn_follows_the_stt_language(stt_language: str, expected: str) -> None:
    assert detect_caller_language("أبغى أعرف عن Growth plan", stt_language, DOMINANCE) == expected


def test_turn_without_letters_or_stt_language_has_no_language() -> None:
    assert detect_caller_language("149", None, DOMINANCE) is None


@pytest.mark.parametrize(
    ("stt_language", "expected"),
    [("ara", "ar"), ("ar-SA", "ar"), ("eng", "en"), ("en-US", "en"), ("fr", None), (None, None)],
)
def test_stt_language_codes_are_normalised_to_supported_languages(
    stt_language: str | None, expected: str | None
) -> None:
    assert language_from_stt(stt_language) == expected


def test_reply_voice_uses_the_language_override_or_falls_back_to_the_default() -> None:
    overrides = {"ar": "arabic-voice"}

    assert choose_reply_voice("ar", "default-voice", overrides) == ReplyVoice("ar", "arabic-voice")
    assert choose_reply_voice("en", "default-voice", overrides) == ReplyVoice("en", "default-voice")


def test_an_email_in_an_arabic_line_does_not_stop_it_being_arabic() -> None:
    line = "أنا خالد العتيبي، إيميلي khaled.otaibi@example.com، سجلني عندكم كعميل لو سمحت."

    assert detect_caller_language(line, None, DOMINANCE) == "ar"
