import pytest

from voice_agent.conversation.reply_filter import SentenceSplitter, spoken_sentences
from voice_agent.pipeline.spoken_text import to_spoken_text


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        (
            "Growth costs 149 SAR a month.",
            "Growth costs one hundred and forty-nine riyals a month.",
        ),
        ("Starter is SAR 49.", "Starter is forty-nine riyals."),
        ("Up to 5 users and a 14-day trial.", "Up to five users and a fourteen-day trial."),
        ("Open 08:00 to 18:00.", "Open eight AM to six PM."),
        ("Back at 10:30.", "Back at ten thirty AM."),
        ("Save 30%.", "Save thirty percent."),
        (
            "This is a priority P2 ticket, not a P1.",
            "This is a priority two ticket, not a priority one.",
        ),
        ("Your ticket ends in 8010.", "Your ticket ends in eight zero one zero."),
        (
            "Updated on 27 October 2026.",
            "Updated on the twenty-seventh of October twenty twenty-six.",
        ),
    ],
)
def test_english_numbers_prices_times_and_priorities_become_words(
    written: str, spoken: str
) -> None:
    assert to_spoken_text(written, "en") == spoken


@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("باقة النمو تكلف 149 ريالًا.", "باقة النمو تكلف مائة وتسعة وأربعون ريالًا."),
        ("المبتدئ 49 SAR.", "المبتدئ تسعة وأربعون ريالًا."),
        ("السعر 5 ريال.", "السعر خمسة ريالات."),
        ("السعر 1 ريال.", "السعر ريال واحد."),
        ("المكتب من 08:00 حتى 18:30.", "المكتب من الثامنة صباحًا حتى السادسة والنصف مساءً."),
        ("تذكرة P2 تنتهي بـ ٣٣٤٤.", "تذكرة الأولوية الثانية تنتهي بـ ثلاثة ثلاثة أربعة أربعة."),
        ("خصم 30%.", "خصم ثلاثون بالمئة."),
        ("صدّر بصيغة CSV أو PDF.", "صدّر بصيغة سي إس في أو بي دي إف."),
        ("اشترك في Ledgerly.", "اشترك في ليدجرلي."),
    ],
)
def test_arabic_numbers_prices_times_and_priorities_become_arabic_words(
    written: str, spoken: str
) -> None:
    assert to_spoken_text(written, "ar") == spoken


def test_a_spelled_email_in_an_arabic_reply_is_spoken_with_arabic_letter_names() -> None:
    spoken = to_spoken_text("البريد s a r a at e x dot c o m", "ar")

    assert spoken == "البريد إس إيه آر إيه آت إي إكس دوت سي أو إم"


def test_english_replies_keep_latin_terms_as_they_are() -> None:
    assert (
        to_spoken_text("Export as CSV or PDF from Ledgerly.", "en")
        == "Export as CSV or PDF from Ledgerly."
    )


def test_sentences_are_split_only_once_they_are_complete() -> None:
    splitter = SentenceSplitter()

    assert splitter.push("Growth costs 149") == []
    assert splitter.push(" SAR. Starter is ") == ["Growth costs 149 SAR."]
    assert splitter.flush() == "Starter is"


async def test_streamed_text_is_spoken_per_sentence_in_its_language() -> None:
    async def streamed() -> object:
        for piece in ["It costs 1", "49 SAR. ", "Ticket ", "8010."]:
            yield piece

    spoken = [sentence async for sentence in spoken_sentences(streamed(), lambda _: "en")]

    assert "".join(spoken).strip() == (
        "It costs one hundred and forty-nine riyals. Ticket eight zero one zero."
    )
