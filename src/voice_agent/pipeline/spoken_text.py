import re
from collections.abc import Callable

from num2words import num2words

from voice_agent.pipeline.language import ARABIC_INDIC_DIGITS, CallerLanguage

# Four or more digits without a unit are identifiers (ticket and invoice numbers): said digit
# by digit, which is how callers repeat them back.
IDENTIFIER_MIN_DIGITS = 4

DIGIT_WORDS = {
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "ar": ["صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"],
}
PRIORITY_WORDS = {
    "en": {"1": "one", "2": "two", "3": "three"},
    "ar": {"1": "الأولى", "2": "الثانية", "3": "الثالثة"},
}
ARABIC_HOURS = (
    "الثانية عشرة", "الواحدة", "الثانية", "الثالثة", "الرابعة", "الخامسة", "السادسة",
    "السابعة", "الثامنة", "التاسعة", "العاشرة", "الحادية عشرة",
)  # fmt: skip
ENGLISH_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
ARABIC_MONTHS = "يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر"
ARABIC_TERMS = {
    "CSV": "سي إس في", "PDF": "بي دي إف", "QR": "كيو آر", "ZATCA": "زاتكا", "SADAD": "سداد",
    "SAR": "ريال", "IT": "تقنية المعلومات", "Ledgerly": "ليدجرلي", "Larkspur": "لاركسبر",
    "Systems": "سيستمز", "Apple Pay": "آبل باي", "Visa": "فيزا", "Mastercard": "ماستركارد",
    "mada": "مدى", "at": "آت", "dot": "دوت", "dash": "داش", "underscore": "أندرسكور", "plus": "بلس",
}  # fmt: skip
ARABIC_LETTER_NAMES = {
    "A": "إيه", "B": "بي", "C": "سي", "D": "دي", "E": "إي", "F": "إف", "G": "جي",
    "H": "إتش", "I": "آي", "J": "جيه", "K": "كيه", "L": "إل", "M": "إم", "N": "إن",
    "O": "أو", "P": "بي", "Q": "كيو", "R": "آر", "S": "إس", "T": "تي", "U": "يو",
    "V": "في", "W": "دبليو", "X": "إكس", "Y": "واي", "Z": "زد",
}  # fmt: skip

NUMBER = r"\d[\d,]*(?:\.\d+)?"
PRICE = re.compile(
    rf"\bSAR\s*({NUMBER})|({NUMBER})\s*(?:SAR\b|riyals?\b|ريال[\w\u064b-\u0652]*)", re.IGNORECASE
)
PERCENT = re.compile(rf"({NUMBER})\s*(?:%|percent\b|بالمئة|بالمائة)", re.IGNORECASE)
TIME = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
PRIORITY = re.compile(r"(?:\bpriority\s+|ب?الأولوية\s+|أولوية\s+)?\bP([1-3])\b", re.IGNORECASE)
DAY_BEFORE_MONTH = re.compile(rf"\b(\d{{1,2}})\s+({ENGLISH_MONTHS}|{ARABIC_MONTHS})")
YEAR_AFTER_MONTH = re.compile(rf"({ENGLISH_MONTHS}|{ARABIC_MONTHS})\s+((?:19|20)\d\d)\b")
AROUND_THE_CLOCK = re.compile(r"\b24\s*/\s*7\b")
BARE_NUMBER = re.compile(r"\d[\d,]*")


def cardinal(value: str, language: CallerLanguage) -> str:
    words = num2words(
        float(value.replace(",", "")) if "." in value else int(value.replace(",", "")),
        lang=language,
    )
    return words.replace(" و ", " و") if language == "ar" else words


def riyals(amount: str, language: CallerLanguage) -> str:
    if language == "en":
        return f"{cardinal(amount, 'en')} {'riyal' if amount == '1' else 'riyals'}"
    # Arabic counted nouns: 1 and 2 have their own forms, 3 to 10 take the plural, 11 to 99 the
    # accusative singular, and round hundreds the bare singular.
    count = int(float(amount.replace(",", "")))
    if count in (1, 2):
        return {1: "ريال واحد", 2: "ريالان"}[count]
    remainder = count % 100
    noun = "ريالات" if 3 <= remainder <= 10 else "ريالًا" if remainder >= 11 else "ريال"
    return f"{cardinal(amount, 'ar')} {noun}"


def clock_time(hour: int, minute: int, language: CallerLanguage) -> str:
    if language == "en":
        spoken_minute = (
            ""
            if minute == 0
            else f" {num2words(minute)}"
            if minute >= 10
            else f" oh {num2words(minute)}"
        )
        return f"{num2words(hour % 12 or 12)}{spoken_minute} {'AM' if hour < 12 else 'PM'}"
    named_minutes = {0: "", 15: " والربع", 30: " والنصف"}
    spoken_minute = named_minutes.get(minute, f" و{cardinal(str(minute), 'ar')} دقيقة")
    period = "صباحًا" if hour < 12 else "مساءً"
    return f"{ARABIC_HOURS[hour % 12]}{spoken_minute} {period}"


def spoken_date_day(day: str, month: str, language: CallerLanguage) -> str:
    if language == "en":
        return f"the {num2words(int(day), to='ordinal')} of {month}"
    return f"{cardinal(day, 'ar')} {month}"


def spoken_year(month: str, year: str, language: CallerLanguage) -> str:
    words = num2words(int(year), to="year") if language == "en" else cardinal(year, "ar")
    return f"{month} {words}"


def spoken_number(digits: str, language: CallerLanguage) -> str:
    if len(digits) >= IDENTIFIER_MIN_DIGITS and "," not in digits:
        return " ".join(DIGIT_WORDS[language][int(digit)] for digit in digits)
    return cardinal(digits, language)


def arabic_latin_words(text: str) -> str:
    for term, spoken in sorted(ARABIC_TERMS.items(), key=lambda item: -len(item[0])):
        text = re.sub(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", spoken, text)
    # Spelled-out letters and acronyms take their Arabic letter names.
    return re.sub(
        r"(?<![A-Za-z])(?:[A-Za-z]|[A-Z]{2,5})(?![A-Za-z])",
        lambda match: " ".join(ARABIC_LETTER_NAMES[letter.upper()] for letter in match.group(0)),
        text,
    )


def to_spoken_text(text: str, language: CallerLanguage) -> str:
    rules: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
        (AROUND_THE_CLOCK, lambda _: "around the clock" if language == "en" else "على مدار الساعة"),
        (PRICE, lambda match: riyals(match.group(1) or match.group(2), language)),
        (
            PERCENT,
            lambda match: (
                f"{cardinal(match.group(1), language)} "
                + ("percent" if language == "en" else "بالمئة")
            ),
        ),
        (TIME, lambda match: clock_time(int(match.group(1)), int(match.group(2)), language)),
        (YEAR_AFTER_MONTH, lambda match: spoken_year(match.group(1), match.group(2), language)),
        (DAY_BEFORE_MONTH, lambda match: spoken_date_day(match.group(1), match.group(2), language)),
        (
            PRIORITY,
            lambda match: (
                ("priority " if language == "en" else "الأولوية ")
                + PRIORITY_WORDS[language][match.group(1)]
            ),
        ),
        (BARE_NUMBER, lambda match: spoken_number(match.group(0), language)),
    ]
    spoken = text.translate(ARABIC_INDIC_DIGITS)
    for pattern, replacement in rules:
        spoken = pattern.sub(replacement, spoken)
    return arabic_latin_words(spoken) if language == "ar" else spoken
