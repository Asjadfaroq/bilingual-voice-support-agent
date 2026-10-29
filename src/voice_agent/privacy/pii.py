import logging
import re

DIGIT = r"[0-9\u0660-\u0669]"
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Saudi national IDs start with 1 and iqama numbers with 2; both are ten digits.
NATIONAL_ID_PATTERN = re.compile(rf"(?<!{DIGIT})[12\u0661\u0662]{DIGIT}{{9}}(?!{DIGIT})")
# Dates such as 2026-10-29 have the shape of a phone number, so they are excluded.
PHONE_PATTERN = re.compile(
    rf"(?<![\w+])(?!\d{{4}}-\d{{2}}-\d{{2}})\+?{DIGIT}(?:[\s-]?{DIGIT}){{7,14}}(?!{DIGIT})"
)

# Read-backs spell values out ("s a r a at e x dot c o m"), which no email or phone pattern
# matches, so any run of six or more single characters and spoken symbols is masked.
SPELLED_TOKEN = r"(?:[^\W_]|dot|dash|underscore|plus|at)"
SPELLED_VALUE_PATTERN = re.compile(
    rf"(?<!\w){SPELLED_TOKEN}(?:[\s,]+{SPELLED_TOKEN}){{5,}}(?!\w)", re.IGNORECASE
)

SPELLED_VALUE_PLACEHOLDER = "[spelled value]"
EMAIL_PLACEHOLDER = "[email]"
NATIONAL_ID_PLACEHOLDER = "[national id]"
PHONE_PLACEHOLDER = "[phone]"


def mask_pii(text: str) -> str:
    masked = SPELLED_VALUE_PATTERN.sub(SPELLED_VALUE_PLACEHOLDER, text)
    masked = EMAIL_PATTERN.sub(EMAIL_PLACEHOLDER, masked)
    masked = NATIONAL_ID_PATTERN.sub(NATIONAL_ID_PLACEHOLDER, masked)
    return PHONE_PATTERN.sub(PHONE_PLACEHOLDER, masked)


class PiiMaskingFilter(logging.Filter):
    """Masks the rendered message and string extras, which LiveKit uses for transcripts."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_pii(record.getMessage())
        record.args = None
        for name, value in vars(record).items():
            if name.startswith("lk.") and isinstance(value, str):
                setattr(record, name, mask_pii(value))
        return True
