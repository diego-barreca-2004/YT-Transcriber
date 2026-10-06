from dataclasses import dataclass

import pytest
from youtube_transcript_api import NoTranscriptFound

from yt_transcriber.errors import TranscriptError
from yt_transcriber.transcript import select_transcript


@dataclass
class FakeTranscript:
    language_code: str
    is_generated: bool


class FakeTranscriptList:
    """Mimics youtube_transcript_api.TranscriptList: manual transcripts first."""

    def __init__(self, *transcripts):
        self.transcripts = sorted(transcripts, key=lambda t: t.is_generated)

    def __iter__(self):
        return iter(self.transcripts)

    def _find(self, codes, generated):
        for code in codes:
            for t in self.transcripts:
                if t.language_code == code and t.is_generated == generated:
                    return t
        raise NoTranscriptFound("dQw4w9WgXcQ", codes, self)

    def find_manually_created_transcript(self, codes):
        return self._find(codes, generated=False)

    def find_generated_transcript(self, codes):
        return self._find(codes, generated=True)


MANUAL_DE = FakeTranscript("de", False)
MANUAL_EN = FakeTranscript("en", False)
AUTO_EN = FakeTranscript("en", True)
AUTO_KO = FakeTranscript("ko", True)
MANUAL_IT = FakeTranscript("it", False)


@pytest.mark.parametrize(
    "available, lang, expected",
    [
        ((AUTO_EN, MANUAL_EN, MANUAL_DE), None, MANUAL_EN),
        ((AUTO_EN, MANUAL_DE), None, AUTO_EN),
        ((AUTO_KO, MANUAL_DE), None, MANUAL_DE),
        ((AUTO_KO,), None, AUTO_KO),
        ((MANUAL_EN, MANUAL_IT), "it", MANUAL_IT),
        ((MANUAL_EN, AUTO_KO), "ko", AUTO_KO),
    ],
)
def test_selection_order(available, lang, expected):
    assert select_transcript(FakeTranscriptList(*available), lang) is expected


def test_missing_language_lists_available_ones():
    with pytest.raises(TranscriptError, match="Available languages for this video: en, ko"):
        select_transcript(FakeTranscriptList(MANUAL_EN, AUTO_KO), "it")


@pytest.mark.parametrize("lang", [None, "en"])
def test_no_transcripts(lang):
    with pytest.raises(TranscriptError, match="no transcript"):
        select_transcript(FakeTranscriptList(), lang)
