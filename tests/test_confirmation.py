import pytest

from voice_agent.conversation.confirmation import is_explicit_yes, spell_for_read_back


@pytest.mark.parametrize(
    "answer",
    [
        "Yes, please open it.",
        "Sure, go ahead",
        "That's right",
        "نعم، افتحها",
        "أيوه تمام",
        "إي نعم",
    ],
)
def test_clear_yes_in_english_or_arabic_counts_as_confirmation(answer: str) -> None:
    assert is_explicit_yes(answer)


@pytest.mark.parametrize(
    "answer",
    [
        "No, don't open that.",
        "Yes, but change the priority.",
        "Wait, not yet",
        "Maybe",
        "Hmm",
        "لا، لا تفتحها",
        "لحظة",
        "نعم بس غير الأولوية",
    ],
)
def test_no_hesitation_or_a_conditional_yes_is_never_confirmation(answer: str) -> None:
    assert not is_explicit_yes(answer)


def test_email_is_spelled_out_symbol_by_symbol_for_the_read_back() -> None:
    assert spell_for_read_back("sa.r-a@ex.com") == "s a dot r dash a at e x dot c o m"
