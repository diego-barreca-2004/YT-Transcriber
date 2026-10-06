import pytest
from youtube_transcript_api import (
    AgeRestricted,
    IpBlocked,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
)

from yt_transcriber import errors
from yt_transcriber.errors import TranscriptError, call_with_retries, describe_error

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "exc, fragment, retryable",
    [
        (RequestBlocked(VIDEO_ID), "temporarily blocking", True),
        (IpBlocked(VIDEO_ID), "temporarily blocking", True),
        (TranscriptsDisabled(VIDEO_ID), "captions are disabled", False),
        (VideoUnplayable(VIDEO_ID, "reason", []), "not playable", False),
        (VideoUnavailable(VIDEO_ID), "unavailable", False),
        (AgeRestricted(VIDEO_ID), "could not retrieve the transcript", False),
        (TranscriptError("custom message"), "custom message", False),
        (ConnectionError("network down"), "unexpected error (ConnectionError)", False),
    ],
)
def test_describe_error(exc, fragment, retryable):
    message, is_retryable = describe_error(exc, VIDEO_ID)
    assert fragment in message
    assert is_retryable is retryable


class Flaky:
    """Raise the given exceptions in order, then return "ok"."""

    def __init__(self, *exceptions):
        self.exceptions = list(exceptions)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.exceptions:
            raise self.exceptions.pop(0)
        return "ok"


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(errors.time, "sleep", sleeps.append)
    return sleeps


def test_retries_rate_limit_then_succeeds(no_sleep):
    func = Flaky(RequestBlocked(VIDEO_ID), IpBlocked(VIDEO_ID))
    logs = []
    result = call_with_retries(func, retries=2, retry_delay=5, video_id=VIDEO_ID, log=logs.append)
    assert result == "ok"
    assert func.calls == 3
    assert len(logs) == 2
    # Exponential backoff with ±20% jitter: ~5s then ~10s.
    assert 4 <= no_sleep[0] <= 6
    assert 8 <= no_sleep[1] <= 12


def test_gives_up_after_last_attempt():
    func = Flaky(*[RequestBlocked(VIDEO_ID)] * 5)
    with pytest.raises(RequestBlocked):
        call_with_retries(func, retries=2, retry_delay=0, video_id=VIDEO_ID, log=lambda _: None)
    assert func.calls == 3


def test_permanent_errors_are_not_retried(no_sleep):
    func = Flaky(VideoUnavailable(VIDEO_ID))
    with pytest.raises(VideoUnavailable):
        call_with_retries(func, retries=5, retry_delay=1, video_id=VIDEO_ID)
    assert func.calls == 1
    assert no_sleep == []


def test_zero_retries():
    func = Flaky(RequestBlocked(VIDEO_ID))
    with pytest.raises(RequestBlocked):
        call_with_retries(func, retries=0, retry_delay=1, video_id=VIDEO_ID)
    assert func.calls == 1
