"""Error classification and retry logic shared by both command-line tools.

``describe_error`` turns an exception raised while fetching a transcript into
a short, user-facing message and decides whether retrying makes sense.
``call_with_retries`` retries a call with exponential backoff, but only for
errors that ``describe_error`` marks as retryable.

Only YouTube rate limiting (``RequestBlocked`` / ``IpBlocked``) is retryable:
it is a temporary problem with the caller's IP, not with the video. Every
other error (video unavailable, captions disabled, language not found, ...)
is permanent for that video, so retrying would only waste time.
"""
from __future__ import annotations

import random
import sys
import time
from typing import Callable

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeTranscriptApiException,
)

# Random jitter (±20%) applied to each retry delay, so that parallel workers
# hitting the same rate limit do not all retry at the same instant.
_JITTER_FRACTION = 0.2


class TranscriptError(RuntimeError):
    """An error whose message is already written for the end user."""


def describe_error(exc: Exception, video_id: str) -> tuple[str, bool]:
    """Return ``(message, is_retryable)`` for an exception raised for ``video_id``."""
    if isinstance(exc, RequestBlocked):
        # Also covers IpBlocked (a subclass). The library's own message is a
        # long paragraph about paid proxies, which is not useful to end users.
        return (
            "YouTube is temporarily blocking requests from this IP address "
            "(too much traffic). Try again later or from another network.",
            True,
        )
    if isinstance(exc, TranscriptsDisabled):
        return (f"captions are disabled for video '{video_id}'.", False)
    if isinstance(exc, VideoUnplayable):
        return (
            f"video '{video_id}' is not playable "
            "(e.g. restricted content or a live stream that is still processing).",
            False,
        )
    if isinstance(exc, VideoUnavailable):
        # Without authentication YouTube does not distinguish private, removed
        # and non-existent videos, so neither do we.
        return (f"video '{video_id}' is unavailable (removed, private or never existed).", False)
    if isinstance(exc, TranscriptError):
        return (str(exc), False)
    if isinstance(exc, CouldNotRetrieveTranscript):
        return (f"could not retrieve the transcript for '{video_id}': {exc}", False)
    if isinstance(exc, YouTubeTranscriptApiException):
        return (str(exc), False)

    # Network errors, YouTube page changes, unexpected bugs.
    return (f"unexpected error ({type(exc).__name__}): {exc}", False)


def call_with_retries(
    func: Callable,
    *args,
    retries: int,
    retry_delay: float,
    video_id: str,
    log: Callable[[str], None] | None = None,
    **kwargs,
):
    """Call ``func(*args, **kwargs)``, retrying only on retryable errors.

    retries: extra attempts after the first one (0 disables retrying).
    retry_delay: seconds to wait before the first retry; the delay doubles at
        each further attempt (5s, 10s, 20s, ... by default) with ±20% jitter.
    video_id: used for the log message and to classify the exception.
    log: receives one line per retry (default: print to stderr).

    Non-retryable errors are re-raised immediately. If the last attempt also
    fails, its exception is re-raised.
    """
    if log is None:

        def log(msg: str) -> None:
            print(msg, file=sys.stderr)

    total_attempts = retries + 1
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            attempt += 1
            _message, is_retryable = describe_error(exc, video_id)
            if not is_retryable or attempt >= total_attempts:
                raise
            delay = retry_delay * (2 ** (attempt - 1))
            delay += delay * random.uniform(-_JITTER_FRACTION, _JITTER_FRACTION)
            delay = max(0.0, delay)
            log(
                f"Attempt {attempt + 1}/{total_attempts} in {delay:.1f}s "
                f"(YouTube rate limit, video '{video_id}')..."
            )
            time.sleep(delay)
