"""Output formatters: plain text, JSON, SubRip (SRT) and Markdown.

Every formatter takes the same raw data produced by
``transcript.fetch_transcript_segments`` or
``whisper_fallback.transcribe_with_whisper``:

- ``segments``: list of ``{"text": str, "start": float, "duration": float}``
- ``meta``: dict with ``video_id``, ``language``, ``language_code``,
  ``is_generated`` and ``source`` (``"youtube"`` or ``"whisper"``)
"""
from __future__ import annotations

import json

FORMATS = ("txt", "json", "srt", "md")

# Markdown paragraph sizing: close a paragraph at the end of a sentence once
# it reaches the target length, and always before it would exceed the maximum.
_MD_PARAGRAPH_TARGET_CHARS = 600
_MD_PARAGRAPH_MAX_CHARS = 1200


def segments_to_text(segments) -> str:
    """Join the non-empty segments into a single line of plain text."""
    return " ".join(s["text"].strip() for s in segments if s["text"].strip())


def segments_to_json(segments, meta) -> str:
    """Serialize metadata and raw segments as ``{"meta": ..., "segments": [...]}``.

    Segments are kept exactly as received (no stripping, no filtering), so the
    JSON output is a faithful copy of the source data.
    """
    return json.dumps({"meta": meta, "segments": segments}, ensure_ascii=False, indent=2)


def _format_srt_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp ``HH:MM:SS,mmm`` (comma, not dot)."""
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    """Format segments as SubRip subtitles, skipping segments with no text."""
    blocks = []
    index = 1
    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        start = s["start"]
        end = start + s["duration"]
        blocks.append(
            f"{index}\n"
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n"
            f"{text}\n"
        )
        index += 1
    return "\n".join(blocks)


def _group_into_paragraphs(segments) -> list[str]:
    """Group segments into readable paragraphs for the Markdown output.

    A paragraph ends after a segment that closes a sentence (``. ! ? …``) once
    it is at least ``_MD_PARAGRAPH_TARGET_CHARS`` long, and is always closed
    before adding a segment that would push it past
    ``_MD_PARAGRAPH_MAX_CHARS``. The hard limit matters for auto-generated
    captions, which usually have no punctuation at all. A single segment is
    never split.
    """
    paragraphs: list[str] = []
    current = ""
    for s in segments:
        text = s["text"].strip()
        if not text:
            continue
        if current and len(current) + 1 + len(text) > _MD_PARAGRAPH_MAX_CHARS:
            paragraphs.append(current)
            current = ""
        current = f"{current} {text}" if current else text
        if len(current) >= _MD_PARAGRAPH_TARGET_CHARS and text[-1] in ".!?…":
            paragraphs.append(current)
            current = ""
    if current:
        paragraphs.append(current)
    return paragraphs


def segments_to_markdown(segments, meta, video_id) -> str:
    """Format a transcript as a Markdown document.

    The document has an H1 title, a metadata list (link, language, source,
    auto-generated flag, segment count, total duration) and a ``Transcript``
    section with the text grouped into paragraphs.

    The transcript text is inserted verbatim: Markdown special characters are
    not escaped, because escaping would alter the original text.
    """
    lines = [f"# YouTube transcript — {video_id}", ""]
    lines.append(f"- **Video:** [`{video_id}`](https://www.youtube.com/watch?v={video_id})")

    language = meta.get("language")
    language_code = meta.get("language_code")
    if language and language_code and language != language_code:
        lines.append(f"- **Language:** {language} (`{language_code}`)")
    elif language or language_code:
        lines.append(f"- **Language:** `{language or language_code}`")

    if meta.get("source"):
        lines.append(f"- **Source:** {meta['source']}")
    if meta.get("is_generated") is not None:
        lines.append(f"- **Auto-generated:** {'yes' if meta['is_generated'] else 'no'}")

    lines.append(f"- **Segments:** {len(segments)}")
    ends = [
        s["start"] + s["duration"]
        for s in segments
        if isinstance(s.get("start"), (int, float)) and isinstance(s.get("duration"), (int, float))
    ]
    if ends:
        total_seconds = max(ends)
        duration_label = f"{total_seconds:.1f} s"
        minutes, secs = divmod(int(round(total_seconds)), 60)
        if minutes:
            duration_label += f" ({minutes} min {secs:02d} s)"
        lines.append(f"- **Duration:** ~{duration_label}")

    lines.extend(["", "## Transcript", ""])
    for paragraph in _group_into_paragraphs(segments):
        lines.extend([paragraph, ""])
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def render(fmt: str, segments, meta, video_id: str) -> str:
    """Render a transcript in one of ``FORMATS``."""
    if fmt == "json":
        return segments_to_json(segments, meta)
    if fmt == "srt":
        return segments_to_srt(segments)
    if fmt == "md":
        return segments_to_markdown(segments, meta, video_id)
    if fmt == "txt":
        return segments_to_text(segments)
    raise ValueError(f"unknown format: {fmt!r}")
