"""Fetch transcripts for many videos in parallel (``yt-playlist`` command).

Videos can come from any combination of:

- explicit URLs or IDs given as arguments,
- ``--file PATH``: a text file with one URL or ID per line (blank lines and
  lines starting with ``#`` are ignored),
- ``--playlist URL``: a YouTube playlist, expanded with yt-dlp without
  downloading anything.

The combined list is de-duplicated, keeping the first occurrence (arguments,
then file, then playlist). Each video goes through the same pipeline as
``yt-transcript``; a failing video never stops the others. Progress and a
final summary are printed to stderr.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yt_dlp

from ._console import SilentYtdlpLogger, use_utf8_output
from .errors import TranscriptError, describe_error
from .formats import render
from .transcript import (
    add_common_arguments,
    extract_video_id,
    get_transcript,
    validate_common_arguments,
    write_output,
)

_YT_HOST_HINTS = ("youtube.com", "youtu.be")


def extract_playlist_id(value: str) -> str:
    """Return the playlist ID from a ``playlist?list=`` or ``watch?v=...&list=``
    URL, or the value itself if it is not a URL (treated as a bare playlist ID)."""
    value = value.strip()
    if not value:
        raise ValueError("the playlist URL or ID cannot be empty")

    candidate = value
    if "//" not in candidate and any(host in candidate.lower() for host in _YT_HOST_HINTS):
        candidate = f"https://{candidate}"

    parsed = urlparse(candidate)
    list_id = parse_qs(parsed.query).get("list", [""])[0]
    if list_id:
        return list_id
    if parsed.netloc:
        raise ValueError(f"no 'list' parameter in the playlist URL {value!r}")
    return value


def expand_playlist(playlist_value: str) -> list[str]:
    """Return the video IDs of a YouTube playlist, in playlist order.

    Raises TranscriptError if the URL is malformed or the playlist does not
    exist, is private, or has no accessible videos.
    """
    try:
        playlist_id = extract_playlist_id(playlist_value)
    except ValueError as exc:
        raise TranscriptError(str(exc)) from exc
    # Always use the canonical playlist URL: a watch?v=...&list=... URL can be
    # resolved by yt-dlp as a single video instead of the playlist entries.
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "logger": SilentYtdlpLogger(),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        reason = str(exc).removeprefix("ERROR: ")
        raise TranscriptError(
            f"could not read playlist '{playlist_id}' (it does not exist, is "
            f"private, or is unreachable): {reason}"
        ) from exc
    except Exception as exc:
        raise TranscriptError(
            f"unexpected error reading playlist '{playlist_id}' "
            f"({type(exc).__name__}): {exc}"
        ) from exc

    entries = (info or {}).get("entries") or []
    ids = [entry["id"] for entry in entries if entry and entry.get("id")]
    if not ids:
        raise TranscriptError(f"playlist '{playlist_id}' is empty or has no accessible videos.")
    return ids


def read_video_list_file(path: str) -> list[str]:
    """Read one URL or ID per line, skipping blank lines and ``#`` comments."""
    try:
        # utf-8-sig also accepts files saved with a BOM (e.g. by Windows Notepad).
        raw_lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError as exc:
        raise TranscriptError(f"file not found: {path!r}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise TranscriptError(f"could not read {path!r}: {exc}") from exc

    return [
        line.strip()
        for line in raw_lines
        if line.strip() and not line.strip().startswith("#")
    ]


def resolve_video_ids(
    explicit_values: list[str],
    file_path: str | None,
    playlist_value: str | None,
) -> tuple[list[str], list[str]]:
    """Combine all sources into one de-duplicated list of video IDs.

    Returns ``(ids, parse_errors)``. Invalid values from the arguments or the
    file are reported in ``parse_errors`` and do not stop the others. A
    playlist that cannot be expanded raises TranscriptError.
    """
    raw_values = list(explicit_values)
    if file_path:
        raw_values.extend(read_video_list_file(file_path))

    ids: list[str] = []
    seen: set[str] = set()
    parse_errors: list[str] = []

    def add(video_id: str) -> None:
        if video_id not in seen:
            seen.add(video_id)
            ids.append(video_id)

    for raw in raw_values:
        try:
            add(extract_video_id(raw))
        except ValueError as exc:
            parse_errors.append(str(exc))

    if playlist_value:
        for video_id in expand_playlist(playlist_value):
            add(video_id)

    return ids, parse_errors


@dataclass
class VideoResult:
    """Outcome for one video: ``segments``/``meta`` on success, ``error`` on failure."""

    video_id: str
    ok: bool
    segments: list | None = None
    meta: dict | None = None
    error: str | None = None

    @property
    def source(self) -> str | None:
        return self.meta.get("source") if self.meta else None


def _process_one_video(video_id: str, args: argparse.Namespace, gpu_lock: threading.Lock) -> VideoResult:
    """Run the single-video pipeline in a worker thread. Never raises."""
    try:
        segments, meta = get_transcript(
            video_id,
            lang=args.lang,
            whisper_fallback=not args.no_whisper_fallback,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            retries=args.retries,
            retry_delay=args.retry_delay,
            gpu_lock=gpu_lock,
        )
    except Exception as exc:
        message, _is_retryable = describe_error(exc, video_id)
        return VideoResult(video_id, False, error=message)
    return VideoResult(video_id, True, segments=segments, meta=meta)


def run_batch(video_ids: list[str], args: argparse.Namespace) -> int:
    """Process the videos in parallel, write or print the results and print a
    summary. Returns 0 if at least one video succeeded, 1 otherwise."""
    total = len(video_ids)
    # One lock for the whole batch: only one Whisper model runs at a time.
    gpu_lock = threading.Lock()
    results: dict[str, VideoResult] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(_process_one_video, video_id, args, gpu_lock): video_id
            for video_id in video_ids
        }
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results[result.video_id] = result
            status = "ok" if result.ok else "failed"
            print(f"[{done}/{total}] {result.video_id}: {status}", file=sys.stderr)

    ordered = [results[video_id] for video_id in video_ids]
    succeeded = [r for r in ordered if r.ok]
    failed = [r for r in ordered if not r.ok]

    if args.stdout:
        _print_results(ordered, succeeded, args.format)
    else:
        out_dir = Path(args.output_dir)
        for r in succeeded:
            out_path = out_dir / f"{r.video_id}.{args.format}"
            write_output(out_path, render(args.format, r.segments, r.meta, r.video_id))
            print(f"Wrote {out_path}", file=sys.stderr)

    n_youtube = sum(1 for r in succeeded if r.source == "youtube")
    n_whisper = sum(1 for r in succeeded if r.source == "whisper")
    print(
        f"\nSummary: {total} videos, {len(succeeded)} succeeded "
        f"({n_youtube} from YouTube captions, {n_whisper} with Whisper), "
        f"{len(failed)} failed.",
        file=sys.stderr,
    )
    for r in failed:
        print(f"  {r.video_id}: {r.error}", file=sys.stderr)

    return 0 if succeeded else 1


def _print_results(ordered: list[VideoResult], succeeded: list[VideoResult], fmt: str) -> None:
    if fmt == "json":
        # A single JSON object for the whole batch, so stdout stays parseable.
        payload = {
            r.video_id: (
                {"ok": True, "meta": r.meta, "segments": r.segments}
                if r.ok
                else {"ok": False, "error": r.error}
            )
            for r in ordered
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    # Other formats: one block per video, separated by a header line. The
    # combined output is meant for reading, not as a single valid .srt/.md file.
    for r in succeeded:
        print(f"=== {r.video_id} ===")
        print(render(fmt, r.segments, r.meta, r.video_id))
        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yt-playlist",
        description=(
            "Fetch the transcripts of many YouTube videos in parallel, from URLs/IDs, "
            "a text file and/or a playlist."
        ),
    )
    parser.add_argument(
        "videos",
        nargs="*",
        metavar="VIDEO",
        help="YouTube video URLs or IDs (can be combined with --file and --playlist)",
    )
    parser.add_argument(
        "--playlist",
        metavar="URL",
        help="YouTube playlist URL (playlist?list=... or watch?v=...&list=...) or ID",
    )
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="text file with one video URL or ID per line ('#' starts a comment)",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        default="transcripts",
        metavar="DIR",
        help="directory for the <video_id>.<format> files (default: ./transcripts)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help=(
            "print the transcripts to stdout instead of writing files. With "
            "--format json, prints one JSON object for the whole batch"
        ),
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="only print the resolved video IDs, one per line, without fetching anything",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="with --list-only, print the IDs as a JSON array",
    )
    parser.add_argument(
        "-j",
        "--concurrency",
        type=int,
        default=4,
        metavar="N",
        help="number of videos processed in parallel (default: 4)",
    )
    add_common_arguments(parser)
    return parser


def _print_parse_errors(parse_errors: list[str]) -> None:
    print("Warning: these values were not recognized and were skipped:", file=sys.stderr)
    for msg in parse_errors:
        print(f"  - {msg}", file=sys.stderr)


def main(argv=None) -> int:
    use_utf8_output()
    args = build_arg_parser().parse_args(argv)

    if not args.videos and not args.file and not args.playlist:
        print(
            "Error: no videos given. Pass video URLs/IDs, --file PATH or --playlist URL.",
            file=sys.stderr,
        )
        return 2

    problem = validate_common_arguments(args)
    if problem is None and args.concurrency < 1:
        problem = "--concurrency must be an integer >= 1."
    if problem:
        print(f"Error: {problem}", file=sys.stderr)
        return 2

    try:
        ids, parse_errors = resolve_video_ids(args.videos, args.file, args.playlist)
    except TranscriptError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.list_only:
        if args.json:
            print(json.dumps(ids))
        else:
            for video_id in ids:
                print(video_id)
        if parse_errors:
            _print_parse_errors(parse_errors)
            return 2
        return 0

    if parse_errors:
        _print_parse_errors(parse_errors)
    if not ids:
        print("Error: no valid videos to process.", file=sys.stderr)
        return 1

    return run_batch(ids, args)


if __name__ == "__main__":
    sys.exit(main())
