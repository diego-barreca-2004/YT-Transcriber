"""Whisper fallback logic, without downloading audio or running a model."""
import pytest

pytest.importorskip("faster_whisper")

import yt_dlp  # noqa: E402

from yt_transcriber import whisper_fallback as wf  # noqa: E402


@pytest.mark.parametrize(
    "lang, expected",
    [(None, None), ("", None), ("en", "en"), ("pt-BR", "pt"), ("es-419", "es"), (" IT ", "it")],
)
def test_normalize_language_code(lang, expected):
    assert wf._normalize_language_code(lang) == expected


@pytest.fixture
def runs(monkeypatch):
    """Record which devices _transcribe_on is asked to use. The GPU run fails
    when the audio path is "cuda fails"."""
    calls = []

    def fake_transcribe_on(device, audio_path, model_size, language):
        calls.append(device)
        if device == "cuda" and audio_path == "cuda fails":
            raise RuntimeError("CUDA out of memory")
        return [], None

    monkeypatch.setattr(wf, "_transcribe_on", fake_transcribe_on)
    return calls


def test_auto_without_gpu_uses_cpu(monkeypatch, runs):
    monkeypatch.setattr(wf.ctranslate2, "get_cuda_device_count", lambda: 0)
    wf._transcribe("audio", "tiny", "auto", None)
    assert runs == ["cpu"]


def test_auto_with_gpu_uses_cuda(monkeypatch, runs):
    monkeypatch.setattr(wf.ctranslate2, "get_cuda_device_count", lambda: 1)
    wf._transcribe("audio", "tiny", "auto", None)
    assert runs == ["cuda"]


def test_auto_falls_back_to_cpu_when_gpu_fails(monkeypatch, runs, capsys):
    monkeypatch.setattr(wf.ctranslate2, "get_cuda_device_count", lambda: 1)
    wf._transcribe("cuda fails", "tiny", "auto", None)
    assert runs == ["cuda", "cpu"]
    assert "Retrying on the CPU" in capsys.readouterr().err


def test_explicit_cuda_does_not_fall_back(monkeypatch, runs):
    monkeypatch.setattr(wf.ctranslate2, "get_cuda_device_count", lambda: 1)
    with pytest.raises(RuntimeError):
        wf._transcribe("cuda fails", "tiny", "cuda", None)
    assert runs == ["cuda"]


def test_explicit_cpu_never_touches_cuda(monkeypatch, runs):
    monkeypatch.setattr(wf.ctranslate2, "get_cuda_device_count", lambda: 1)
    wf._transcribe("audio", "tiny", "cpu", None)
    assert runs == ["cpu"]


class FakeYoutubeDL:
    """Fails with the queued errors, then 'downloads' an empty file."""

    errors: list = []
    calls = 0

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        FakeYoutubeDL.calls += 1
        if FakeYoutubeDL.errors:
            raise FakeYoutubeDL.errors.pop(0)
        return {"id": "dQw4w9WgXcQ", "ext": "webm"}

    def prepare_filename(self, info):
        path = self.opts["outtmpl"].replace("%(id)s", info["id"]).replace("%(ext)s", info["ext"])
        open(path, "wb").close()
        return path


@pytest.fixture
def fake_ydl(monkeypatch):
    FakeYoutubeDL.calls = 0
    FakeYoutubeDL.errors = []
    monkeypatch.setattr(wf.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(wf.time, "sleep", lambda _: None)
    return FakeYoutubeDL


def test_download_retries_http_403(tmp_path, fake_ydl):
    fake_ydl.errors = [yt_dlp.utils.DownloadError("ERROR: HTTP Error 403: Forbidden")]
    path = wf._download_audio("dQw4w9WgXcQ", str(tmp_path))
    assert path.endswith("dQw4w9WgXcQ.webm")
    assert fake_ydl.calls == 2


def test_download_does_not_retry_other_errors(tmp_path, fake_ydl):
    fake_ydl.errors = [yt_dlp.utils.DownloadError("ERROR: Video unavailable")]
    with pytest.raises(yt_dlp.utils.DownloadError):
        wf._download_audio("dQw4w9WgXcQ", str(tmp_path))
    assert fake_ydl.calls == 1


def test_download_gives_up_after_three_403s(tmp_path, fake_ydl):
    fake_ydl.errors = [yt_dlp.utils.DownloadError("HTTP Error 403: Forbidden")] * 3
    with pytest.raises(yt_dlp.utils.DownloadError):
        wf._download_audio("dQw4w9WgXcQ", str(tmp_path))
    assert fake_ydl.calls == 3
