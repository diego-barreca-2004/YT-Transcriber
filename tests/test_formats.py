import json

import pytest

from yt_transcriber import formats
from yt_transcriber.formats import (
    _format_srt_timestamp,
    _group_into_paragraphs,
    render,
    segments_to_json,
    segments_to_markdown,
    segments_to_srt,
    segments_to_text,
)

SEGMENTS = [
    {"text": " Hello there. ", "start": 0.0, "duration": 1.5},
    {"text": "   ", "start": 1.5, "duration": 0.5},
    {"text": "Ciao — 你好 ♪", "start": 2.0, "duration": 2.25},
]
META = {
    "video_id": "dQw4w9WgXcQ",
    "language": "English",
    "language_code": "en",
    "is_generated": False,
    "source": "youtube",
}


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (61.001, "00:01:01,001"),
        (3600 + 2 * 60 + 3.4567, "01:02:03,457"),
        (59.9996, "00:01:00,000"),
    ],
)
def test_srt_timestamp(seconds, expected):
    assert _format_srt_timestamp(seconds) == expected


def test_text_skips_blank_segments_and_strips():
    assert segments_to_text(SEGMENTS) == "Hello there. Ciao — 你好 ♪"


def test_srt_numbering_skips_blank_segments():
    assert segments_to_srt(SEGMENTS) == (
        "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n"
        "2\n00:00:02,000 --> 00:00:04,250\nCiao — 你好 ♪\n"
    )


def test_json_keeps_raw_segments_and_unicode():
    output = segments_to_json(SEGMENTS, META)
    assert "你好" in output  # not escaped
    assert json.loads(output) == {"meta": META, "segments": SEGMENTS}


def test_markdown_document():
    output = segments_to_markdown(SEGMENTS, META, "dQw4w9WgXcQ")
    lines = output.splitlines()
    assert lines[0] == "# YouTube transcript — dQw4w9WgXcQ"
    assert "- **Language:** English (`en`)" in lines
    assert "- **Source:** youtube" in lines
    assert "- **Auto-generated:** no" in lines
    assert "- **Segments:** 3" in lines
    assert "- **Duration:** ~4.2 s" in lines
    assert lines[-3:] == ["## Transcript", "", "Hello there. Ciao — 你好 ♪"]


def test_markdown_omits_unknown_metadata():
    output = segments_to_markdown([{"text": "hi"}], {"language_code": "ko"}, "abc")
    assert "- **Language:** `ko`" in output
    assert "Source" not in output
    assert "Auto-generated" not in output
    assert "Duration" not in output


def test_paragraphs_break_at_sentence_end_after_target(monkeypatch):
    monkeypatch.setattr(formats, "_MD_PARAGRAPH_TARGET_CHARS", 10)
    monkeypatch.setattr(formats, "_MD_PARAGRAPH_MAX_CHARS", 1000)
    segments = [{"text": t} for t in ["one two", "three four.", "five", "six."]]
    assert _group_into_paragraphs(segments) == ["one two three four.", "five six."]


def test_paragraphs_respect_hard_limit_without_punctuation(monkeypatch):
    monkeypatch.setattr(formats, "_MD_PARAGRAPH_TARGET_CHARS", 5)
    monkeypatch.setattr(formats, "_MD_PARAGRAPH_MAX_CHARS", 12)
    segments = [{"text": "aaaa"}, {"text": "bbbb"}, {"text": "cccc"}, {"text": "dddddddddddddddd"}]
    assert _group_into_paragraphs(segments) == ["aaaa bbbb", "cccc", "dddddddddddddddd"]


@pytest.mark.parametrize("fmt", formats.FORMATS)
def test_render_all_formats(fmt):
    assert render(fmt, SEGMENTS, META, "dQw4w9WgXcQ")


def test_render_unknown_format():
    with pytest.raises(ValueError):
        render("docx", SEGMENTS, META, "dQw4w9WgXcQ")
