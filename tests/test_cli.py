"""Command-line tests. Network access is replaced by fakes."""
import json
import sys

import pytest
from youtube_transcript_api import TranscriptsDisabled, VideoUnavailable

from yt_transcriber import playlist, transcript

SEGMENTS = [
    {"text": "Hello", "start": 0.0, "duration": 1.0},
    {"text": "world ♪", "start": 1.0, "duration": 1.0},
]


def fake_fetch(video_id, lang=None):
    # "unavailable" and "noCaptions1" are 11 characters, so they pass ID validation.
    if video_id == "unavailable":
        raise VideoUnavailable(video_id)
    if video_id == "noCaptions1":
        raise TranscriptsDisabled(video_id)
    meta = {
        "video_id": video_id,
        "language": "English",
        "language_code": "en",
        "is_generated": False,
        "source": "youtube",
    }
    return SEGMENTS, meta


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(transcript, "fetch_transcript_segments", fake_fetch)


def test_prints_text_to_stdout(capsys):
    assert transcript.main(["dQw4w9WgXcQ"]) == 0
    assert capsys.readouterr().out == "Hello world ♪\n"


def test_writes_file_with_lf_line_endings(tmp_path):
    out = tmp_path / "nested" / "out.srt"
    assert transcript.main(["https://youtu.be/dQw4w9WgXcQ", "--format", "srt", "-o", str(out)]) == 0
    data = out.read_bytes()
    assert b"\r\n" not in data
    assert data.decode("utf-8").startswith("1\n00:00:00,000 --> 00:00:01,000\nHello\n")


def test_invalid_video_is_a_usage_error(capsys):
    assert transcript.main(["not a video"]) == 2
    assert "not a valid YouTube video ID" in capsys.readouterr().err


def test_invalid_retries(capsys):
    assert transcript.main(["dQw4w9WgXcQ", "--retries", "-1"]) == 2


def test_disabled_captions_without_fallback(capsys):
    assert transcript.main(["noCaptions1", "--no-whisper-fallback"]) == 1
    assert "--no-whisper-fallback" in capsys.readouterr().err


def test_missing_whisper_extra_is_explained(monkeypatch, capsys):
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "yt_transcriber.whisper_fallback", None)
    assert transcript.main(["noCaptions1"]) == 1
    assert 'pip install "yt-transcriber[whisper]"' in capsys.readouterr().err


def test_whisper_fallback_is_used(monkeypatch, capsys):
    calls = []

    def fake_whisper(video_id, lang, model_size, *, device, gpu_lock):
        calls.append((video_id, lang, model_size, device))
        return [{"text": "from whisper", "start": 0.0, "duration": 1.0}], {"source": "whisper"}

    monkeypatch.setattr(transcript, "load_whisper_fallback", lambda: fake_whisper)
    args = ["noCaptions1", "--lang", "it", "--whisper-model", "tiny", "--whisper-device", "cpu"]
    assert transcript.main(args) == 0
    assert capsys.readouterr().out == "from whisper\n"
    assert calls == [("noCaptions1", "it", "tiny", "cpu")]


def test_playlist_writes_one_file_per_video(tmp_path, capsys):
    list_file = tmp_path / "videos.txt"
    list_file.write_text("﻿# comment\n\ndQw4w9WgXcQ\nbad value!\n", encoding="utf-8")
    args = [
        "https://youtu.be/dQw4w9WgXcQ",
        "9bZkp7q19f0",
        "unavailable",
        "--file",
        str(list_file),
        "--format",
        "md",
        "--no-whisper-fallback",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    assert playlist.main(args) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["9bZkp7q19f0.md", "dQw4w9WgXcQ.md"]
    err = capsys.readouterr().err
    assert "bad value!" in err
    assert "Summary: 3 videos, 2 succeeded (2 from YouTube captions, 0 with Whisper), 1 failed." in err


def test_playlist_stdout_json_is_one_document(capsys):
    assert playlist.main(["dQw4w9WgXcQ", "unavailable", "--stdout", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dQw4w9WgXcQ"]["ok"] is True
    assert payload["unavailable"]["ok"] is False


def test_playlist_fails_when_every_video_fails(tmp_path):
    assert playlist.main(["unavailable", "--output-dir", str(tmp_path)]) == 1


def test_playlist_list_only_merges_and_dedups(monkeypatch, capsys):
    monkeypatch.setattr(playlist, "expand_playlist", lambda value: ["9bZkp7q19f0", "dQw4w9WgXcQ"])
    assert playlist.main(["dQw4w9WgXcQ", "--playlist", "PL123", "--list-only", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == ["dQw4w9WgXcQ", "9bZkp7q19f0"]


def test_playlist_requires_a_source(capsys):
    assert playlist.main([]) == 2
