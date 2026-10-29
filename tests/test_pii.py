import logging

import pytest

from voice_agent.privacy.pii import PiiMaskingFilter, mask_pii


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("my email is sara.a+x@example.co.uk", "my email is [email]"),
        ("call me on 0551234567", "call me on [phone]"),
        ("it's +966 55 123 4567", "it's [phone]"),
        ("رقمي ٠٥٥١٢٣٤٥٦٧", "رقمي [phone]"),
        ("my iqama is 2123456789", "my iqama is [national id]"),
        ("national id 1098765432", "national id [national id]"),
        ("هويتي ١٠٩٨٧٦٥٤٣٢", "هويتي [national id]"),
    ],
)
def test_emails_phones_and_saudi_ids_are_masked(text: str, expected: str) -> None:
    assert mask_pii(text) == expected


def test_spelled_out_read_backs_are_masked() -> None:
    read_back = (
        "the email s a r a dot v o i c e dash e 2 e at e x a m p l e dot c o m, "
        "phone plus 9 6 6 5 5 1 2 3 4 5 6 7. Is that correct?"
    )

    assert (
        mask_pii(read_back) == "the email [spelled value], phone [spelled value]. Is that correct?"
    )


def test_short_numbers_that_callers_need_stay_readable() -> None:
    text = "invoice 4412, a P1 ticket ending 8 0 1 0, 149 riyals on 29 October, a to b"

    assert mask_pii(text) == text


def test_log_records_are_masked_including_livekit_transcript_fields() -> None:
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "caller said %s", ("0551234567",), None
    )
    setattr(record, "lk.pii.user_input", "email sara@example.com")

    PiiMaskingFilter().filter(record)

    assert record.getMessage() == "caller said [phone]"
    assert getattr(record, "lk.pii.user_input") == "email [email]"
