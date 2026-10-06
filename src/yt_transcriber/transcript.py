"""Fetch the transcript of a single YouTube video (``yt-transcript`` command).

Captions are fetched with youtube-transcript-api. When a video has captions
disabled, the audio is transcribed locally with Whisper instead (optional,
see ``whisper_fallback``). Only YouTube rate limiting is retried; every other
error fails immediately with a clear message on stderr and a non-zero exit
code.
"""
from __future__ import annotations

import argparse
import re
import sys
import threading
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi

from . import __version__
from ._console import use_utf8_output
from .errors import TranscriptError, call_with_retries, describe_error
from .formats import FORMATS, render, segments_to_text

_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
_WATCH_HOSTS = ("youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com")
_SHORT_HOST = "youtu.be"
_KNOWN_HOSTS = _WATCH_HOSTS + (_SHORT_HOST,)
_PATH_PREFIXES = ("/embed/", "/shorts/", "/live/", "/v/")

WHISPER_MODELS = ("tiny", "base", "small", "medium", "large-v3", "turbo")
WHISPER_DEVICES = ("auto", "cpu", "cuda")


def _validated_video_id(candidate: str) -> str:
    if not _VIDEO_ID_RE.fullmatch(candidate):
        raise ValueError(
            f"{candidate!r} is not a valid YouTube video ID or URL "
            "(a video ID is 11 characters: letters, digits, '-' or '_')"
        )
    return candidate


def extract_video_id(value: str) -> str:
    """Return the video ID from a YouTube URL or a bare 11-character ID.

    Supported URLs: ``watch?v=``, ``youtu.be/``, ``/embed/``, ``/shorts/``,
    ``/live/`` and ``/v/`` on youtube.com, m.youtube.com, music.youtube.com and
    youtube-nocookie.com, with or without ``https://`` and ``www.``.
    Raises ValueError for anything else.
    """
    value = value.strip()
    if not value:
        raise ValueError("the video ID or URL cannot be empty")

    # Without a scheme, urlparse() does not fill in the host, so add one to
    # strings that look like YouTube URLs ("youtu.be/...", "www.youtube.com/...").
    if "//" not in value and any(host in value.lower() for host in _KNOWN_HOSTS):
        value = f"https://{value}"

    parsed = urlparse(value)
    host = (parsed.hostname or "").removeprefix("www.")

    if host in _WATCH_HOSTS:
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
            if candidate:
                return _validated_video_id(candidate)
        for prefix in _PATH_PREFIXES:
            if parsed.path.startswith(prefix):
                candidate = parsed.path[len(prefix):].split("/")[0]
                if candidate:
                    return _validated_video_id(candidate)
        raise ValueError(f"could not find a video ID in the URL {value!r}")
    if host == _SHORT_HOST:
        candidate = parsed.path.lstrip("/").split("/")[0]
        if candidate:
            return _validated_video_id(candidate)
        raise ValueError(f"could not find a video ID in the URL {value!r}")
    if host:
        raise ValueError(f"not a YouTube URL: {value!r}")

    return _validated_video_id(value)


def select_transcript(transcript_list, lang: str | None):
    """Pick which of the available transcripts to use.

    With ``lang``: the manual transcript in that language, then the
    auto-generated one; otherwise an error listing the available languages.

    Without ``lang``: manual English, auto-generated English, then the first
    manual transcript in any language, then the first auto-generated one. This
    always returns something when at least one transcript exists.
    """
    if lang:
        for finder in (
            transcript_list.find_manually_created_transcript,
            transcript_list.find_generated_transcript,
        ):
            try:
                return finder([lang])
            except NoTranscriptFound:
                pass
        available = sorted({t.language_code for t in transcript_list})
        if available:
            raise TranscriptError(
                f"no transcript in language '{lang}'. "
                f"Available languages for this video: {', '.join(available)}."
            )
        raise TranscriptError("no transcript is available for this video.")

    for finder in (
        transcript_list.find_manually_created_transcript,
        transcript_list.find_generated_transcript,
    ):
        try:
            return finder(["en"])
        except NoTranscriptFound:
            pass

    transcripts = list(transcript_list)  # manual transcripts come first
    for t in transcripts:
        if not t.is_generated:
            return t
    for t in transcripts:
        if t.is_generated:
            return t

    raise TranscriptError("no transcript is available for this video.")


def fetch_transcript_segments(video_id: str, lang: str | None = None):
    """Fetch the captions of a video from YouTube.

    Returns ``(segments, meta)``: ``segments`` is a list of
    ``{"text", "start", "duration"}`` dicts, ``meta`` describes the chosen
    transcript, with ``source="youtube"``.
    """
    transcript_list = YouTubeTranscriptApi().list(video_id)
    transcript = select_transcript(transcript_list, lang)
    fetched = transcript.fetch()

    segments = [
        {"text": snippet.text, "start": snippet.start, "duration": snippet.duration}
        for snippet in fetched
    ]
    meta = {
        "video_id": video_id,
        "language": transcript.language,
        "language_code": transcript.language_code,
        "is_generated": transcript.is_generated,
        "source": "youtube",
    }
    return segments, meta


def load_whisper_fallback():
    """Return ``transcribe_with_whisper``, or raise TranscriptError if the
    optional Whisper dependencies are not installed."""
    try:
        from .whisper_fallback import transcribe_with_whisper
    except ImportError as exc:
        raise TranscriptError(
            f"the Whisper fallback is not installed ({exc}). Install it with: "
            "pip install \"yt-transcriber[whisper]\" (see the README), or pass "
            "--no-whisper-fallback."
        ) from exc
    return transcribe_with_whisper


def get_transcript(
    video_id: str,
    *,
    lang: str | None = None,
    whisper_fallback: bool = True,
    whisper_model: str = "base",
    whisper_device: str = "auto",
    retries: int = 2,
    retry_delay: float = 5.0,
    gpu_lock: threading.Lock | None = None,
    log: Callable[[str], None] | None = None,
):
    """Get the transcript of one video: YouTube captions first, then Whisper.

    Returns ``(segments, meta)``. Raises on failure; ``describe_error`` turns
    any raised exception into a user-facing message.
    """
    if log is None:

        def log(msg: str) -> None:
            print(msg, file=sys.stderr)

    try:
        segments, meta = call_with_retries(
            fetch_transcript_segments,
            video_id,
            lang,
            retries=retries,
            retry_delay=retry_delay,
            video_id=video_id,
            log=log,
        )
    except TranscriptsDisabled as exc:
        if not whisper_fallback:
            raise TranscriptError(
                f"captions are disabled for video '{video_id}' and the Whisper "
                "fallback is turned off (--no-whisper-fallback)."
            ) from exc
        transcribe_with_whisper = load_whisper_fallback()
        log(
            f"Captions are disabled for '{video_id}', transcribing it locally "
            f"with Whisper (model: {whisper_model})..."
        )
        try:
            segments, meta = transcribe_with_whisper(
                video_id, lang, whisper_model, device=whisper_device, gpu_lock=gpu_lock
            )
        except Exception as exc2:
            raise TranscriptError(
                f"captions are disabled for video '{video_id}' and the local Whisper "
                f"transcription failed ({type(exc2).__name__}: {exc2})"
            ) from exc2

    if not segments_to_text(segments):
        raise TranscriptError(f"the transcript for video '{video_id}' is empty.")
    return segments, meta


def write_output(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` as UTF-8 with LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8", newline="\n")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Options shared by ``yt-transcript`` and ``yt-playlist``."""
    parser.add_argument(
        "--lang",
        metavar="CODE",
        help=(
            "preferred transcript language (e.g. en, it, ko). Default: English "
            "(manual, then auto-generated), then the first available language"
        ),
    )
    parser.add_argument(
        "--format",
        default="txt",
        choices=FORMATS,
        help=(
            "output format: plain text (default), JSON with metadata and timed "
            "segments, SubRip subtitles, or Markdown"
        ),
    )
    parser.add_argument(
        "--no-whisper-fallback",
        action="store_true",
        help="do not transcribe videos with disabled captions locally with Whisper",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        choices=WHISPER_MODELS,
        help="Whisper model size for the local fallback (default: base)",
    )
    parser.add_argument(
        "--whisper-device",
        default="auto",
        choices=WHISPER_DEVICES,
        help=(
            "device for the Whisper fallback: 'auto' uses a CUDA GPU when one "
            "is available and falls back to the CPU (default: auto)"
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        metavar="N",
        help=(
            "extra attempts when YouTube rate-limits this IP (default: 2). "
            "Other errors are never retried. 0 disables retrying"
        ),
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help=(
            "wait before the first retry; doubles at each attempt, with ±20%% "
            "jitter (default: 5)"
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")


def validate_common_arguments(args: argparse.Namespace) -> str | None:
    """Return an error message for invalid shared options, or None."""
    if args.retries < 0:
        return "--retries must be an integer >= 0."
    if args.retry_delay < 0:
        return "--retry-delay must be >= 0."
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-transcript",
        description="Print or save the transcript of a single YouTube video.",
        epilog=(
            "Video IDs that start with '-' must be passed as a URL or after "
            "'--' (e.g. yt-transcript -- -abc123def45)."
        ),
    )
    parser.add_argument("video", help="YouTube video URL or 11-character video ID")
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="write the transcript to this file instead of stdout",
    )
    add_common_arguments(parser)
    return parser


def main(argv=None) -> int:
    use_utf8_output()
    args = build_arg_parser().parse_args(argv)

    try:
        video_id = extract_video_id(args.video)
    except ValueError as exc:
        print(f"Error: {exc}.", file=sys.stderr)
        return 2

    problem = validate_common_arguments(args)
    if problem:
        print(f"Error: {problem}", file=sys.stderr)
        return 2

    try:
        segments, meta = get_transcript(
            video_id,
            lang=args.lang,
            whisper_fallback=not args.no_whisper_fallback,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            retries=args.retries,
            retry_delay=args.retry_delay,
        )
    except Exception as exc:
        message, _is_retryable = describe_error(exc, video_id)
        print(f"Error: {message}", file=sys.stderr)
        return 1

    content = render(args.format, segments, meta, video_id)
    if args.output:
        out_path = Path(args.output)
        write_output(out_path, content)
        print(f"Wrote {out_path}", file=sys.stderr)
    else:
        print(content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
