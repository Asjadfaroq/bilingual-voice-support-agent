import re
from dataclasses import dataclass
from typing import Literal

from livekit.agents.language import LanguageCode

from voice_agent.config import Settings
from voice_agent.privacy.pii import EMAIL_PATTERN, SPELLED_VALUE_PATTERN

CallerLanguage = Literal["en", "ar"]
SUPPORTED_LANGUAGES: dict[str, CallerLanguage] = {"en": "en", "ar": "ar"}
LANGUAGE_NAMES: dict[CallerLanguage, str] = {"en": "English", "ar": "Arabic"}
ARABIC_INDIC_DIGITS = str.maketrans({chr(0x0660 + digit): str(digit) for digit in range(10)})


@dataclass(frozen=True)
class LanguageSettings:
    default_voice_id: str
    voice_id_by_language: dict[str, str]
    script_dominance_threshold: float
    reply_language_min_letters: int


def language_settings_for(settings: Settings) -> LanguageSettings:
    return LanguageSettings(
        default_voice_id=settings.tts_voice_id,
        voice_id_by_language=settings.tts_voice_id_by_language,
        script_dominance_threshold=settings.script_dominance_threshold,
        reply_language_min_letters=settings.reply_language_min_letters,
    )


# Product names, emails, spelled values and numbers look the same in both languages, so they
# are removed before a line's language is judged from its script.
NAMES_IN_ANY_LANGUAGE = re.compile(
    r"Larkspur Systems|Larkspur|Ledgerly|Apple Pay|Mastercard|Visa|mada|ZATCA|SADAD|CSV|PDF|QR|SAR"
    r"|\bP[1-3]\b|\d+",
    re.IGNORECASE,
)
ARABIC_LETTER = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
LATIN_LETTER = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class ReplyVoice:
    language: CallerLanguage
    voice_id: str


def language_from_stt(stt_language: str | None) -> CallerLanguage | None:
    if not stt_language:
        return None
    return SUPPORTED_LANGUAGES.get(LanguageCode(stt_language).language)


def prose_text(text: str) -> str:
    without_emails = SPELLED_VALUE_PATTERN.sub(" ", EMAIL_PATTERN.sub(" ", text))
    return NAMES_IN_ANY_LANGUAGE.sub(" ", without_emails)


def language_from_script(transcript: str, dominance_threshold: float) -> CallerLanguage | None:
    arabic_letters = len(ARABIC_LETTER.findall(transcript))
    latin_letters = len(LATIN_LETTER.findall(transcript))
    total_letters = arabic_letters + latin_letters
    if total_letters == 0:
        return None
    if arabic_letters / total_letters >= dominance_threshold:
        return "ar"
    if latin_letters / total_letters >= dominance_threshold:
        return "en"
    return None


# A reply is in the language most of its letters are written in.
MAJORITY_SCRIPT = 0.5


def prose_language(text: str, min_letters: int) -> CallerLanguage | None:
    prose = prose_text(text)
    if sum(character.isalpha() for character in prose) < min_letters:
        return None
    return language_from_script(prose, MAJORITY_SCRIPT)


def phrase_in_language(phrase: str, language: CallerLanguage, min_letters: int) -> bool:
    # Too few letters, once names, numbers and codes are removed, to be in the wrong language.
    return prose_language(phrase, min_letters) in (None, language)


def detect_caller_language(
    transcript: str, stt_language: str | None, dominance_threshold: float
) -> CallerLanguage | None:
    # The STT plugin reports "en" whenever ElevenLabs omits the language, so a transcript
    # written almost wholly in one script outranks the tag; the tag decides mixed turns.
    prose = prose_text(transcript)
    return language_from_script(prose, dominance_threshold) or language_from_stt(stt_language)


def reply_voice_for(language: CallerLanguage, settings: LanguageSettings) -> ReplyVoice:
    voice_id = settings.voice_id_by_language.get(language, settings.default_voice_id)
    return ReplyVoice(language=language, voice_id=voice_id)
