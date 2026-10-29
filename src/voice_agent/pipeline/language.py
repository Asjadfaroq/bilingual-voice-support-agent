import re
from dataclasses import dataclass
from typing import Literal

from livekit.agents.language import LanguageCode

CallerLanguage = Literal["en", "ar"]
SUPPORTED_LANGUAGES: dict[str, CallerLanguage] = {"en": "en", "ar": "ar"}

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


def detect_caller_language(
    transcript: str, stt_language: str | None, dominance_threshold: float
) -> CallerLanguage | None:
    # The STT plugin reports "en" whenever ElevenLabs omits the language, so a transcript
    # written almost wholly in one script outranks the tag; the tag decides mixed turns.
    return language_from_script(transcript, dominance_threshold) or language_from_stt(stt_language)


def choose_reply_voice(
    language: CallerLanguage, default_voice_id: str, voice_id_by_language: dict[str, str]
) -> ReplyVoice:
    return ReplyVoice(
        language=language, voice_id=voice_id_by_language.get(language, default_voice_id)
    )
