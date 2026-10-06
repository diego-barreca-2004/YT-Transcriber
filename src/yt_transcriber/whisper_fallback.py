"""Local transcription with Whisper (faster-whisper), used when a video has
captions disabled.

1. yt-dlp downloads only the audio track into a temporary directory.
2. faster-whisper transcribes it on a CUDA GPU when available, else on the CPU.
3. The temporary audio file is always deleted, even if a step fails.

This module needs the optional dependencies: ``pip install "yt-transcriber[whisper]"``.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import ctranslate2
import yt_dlp
from faster_whisper import WhisperModel

from ._console import SilentYtdlpLogger

# Whisper only reports an ISO 639-1 code; show a readable name for common ones,
# like youtube-transcript-api does (e.g. "English").
_LANGUAGE_NAMES = {
    "en": "English",
    "it": "Italian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian",
    "nl": "Dutch",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
}

# Whisper resamples everything to 16 kHz mono, so a low-bitrate stream is just
# as accurate and much faster to download. "bestaudio[...]" (rather than
# "worstaudio") keeps yt-dlp's language preference, so videos with dubbed
# audio tracks are transcribed in their original language.
_AUDIO_FORMAT = "bestaudio[abr<=70]/bestaudio/worst"


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _normalize_language_code(lang: str | None) -> str | None:
    """Turn a YouTube-style code (``es-419``, ``pt-BR``) into Whisper's
    two-letter code. None means "let Whisper detect the language"."""
    if not lang:
        return None
    return lang.strip().split("-")[0].lower() or None


def _download_audio(video_id: str, dest_dir: str, attempts: int = 3) -> str:
    """Download the audio track of a video into ``dest_dir`` and return its path.

    YouTube occasionally answers a stream request with HTTP 403. A new
    extraction gets fresh stream URLs, so the whole download is retried a few
    times before giving up.
    """
    ydl_opts = {
        "format": _AUDIO_FORMAT,
        "outtmpl": str(Path(dest_dir) / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "logger": SilentYtdlpLogger(),
    }
    for attempt in range(1, attempts + 1):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
                audio_path = ydl.prepare_filename(info)
            break
        except yt_dlp.utils.DownloadError as exc:
            if attempt == attempts or "HTTP Error 403" not in str(exc):
                raise
            time.sleep(attempt)

    if not Path(audio_path).exists():
        raise RuntimeError(f"audio download for '{video_id}' produced no file ({audio_path!r})")
    return audio_path


def _transcribe_on(device: str, audio_path: str, model_size: str, language: str | None):
    if device == "cuda":
        supported = ctranslate2.get_supported_compute_types("cuda")
        compute_type = "float16" if "float16" in supported else "auto"
    else:
        compute_type = "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    raw_segments, info = model.transcribe(audio_path, language=language)
    # transcribe() returns a lazy generator: the actual inference runs while
    # it is consumed here.
    segments = [
        {"text": seg.text, "start": seg.start, "duration": seg.end - seg.start}
        for seg in raw_segments
    ]
    return segments, info


def _transcribe(audio_path: str, model_size: str, device: str, language: str | None):
    """Run Whisper on the requested device.

    In ``auto`` mode a CUDA GPU is used when one is detected. If the GPU run
    fails (not enough VRAM, missing CUDA libraries, ...) the error is reported
    and the transcription is retried on the CPU, so the user still gets a
    result. With an explicit ``cuda`` device, errors are raised instead.
    """
    if device == "cpu":
        return _transcribe_on("cpu", audio_path, model_size, language)
    if device == "auto" and ctranslate2.get_cuda_device_count() == 0:
        _log("No CUDA GPU detected, running Whisper on the CPU (this is slower).")
        return _transcribe_on("cpu", audio_path, model_size, language)

    try:
        _log("Running Whisper on the CUDA GPU.")
        return _transcribe_on("cuda", audio_path, model_size, language)
    except Exception as exc:
        if device == "cuda":
            raise
        _log(
            f"Whisper failed on the GPU ({type(exc).__name__}: {exc}). Retrying on "
            "the CPU (slower). If this keeps happening, check the available VRAM, "
            "the NVIDIA driver and the CUDA/cuDNN libraries, or pass "
            "--whisper-device cpu."
        )
        return _transcribe_on("cpu", audio_path, model_size, language)


def transcribe_with_whisper(
    video_id: str,
    lang: str | None = None,
    model_size: str = "base",
    *,
    device: str = "auto",
    gpu_lock: threading.Lock | None = None,
):
    """Download the audio of a video and transcribe it locally.

    Returns ``(segments, meta)`` in the same shape as
    ``transcript.fetch_transcript_segments``, with ``source="whisper"`` and
    ``is_generated=True``.

    gpu_lock: when several videos are processed in parallel, a shared lock
    makes sure only one Whisper model runs at a time (several models at once
    can exhaust GPU memory). Downloads still run in parallel.
    """
    tmp_dir = tempfile.mkdtemp(prefix="yt_transcriber_")
    try:
        audio_path = _download_audio(video_id, tmp_dir)
        language = _normalize_language_code(lang)
        if gpu_lock is not None:
            with gpu_lock:
                segments, info = _transcribe(audio_path, model_size, device, language)
        else:
            segments, info = _transcribe(audio_path, model_size, device, language)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    meta = {
        "video_id": video_id,
        "language": _LANGUAGE_NAMES.get(info.language, info.language),
        "language_code": info.language,
        "is_generated": True,
        "source": "whisper",
    }
    return segments, meta
