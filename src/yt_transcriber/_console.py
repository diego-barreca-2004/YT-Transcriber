"""Console helpers shared by the command-line entry points."""
from __future__ import annotations

import sys


def use_utf8_output() -> None:
    """Make stdout/stderr write UTF-8 regardless of the platform locale.

    Transcripts routinely contain non-ASCII text (accents, CJK, music notes).
    On Windows, redirecting output to a file or pipe otherwise falls back to
    the legacy code page (e.g. cp1252) and crashes with UnicodeEncodeError.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


class SilentYtdlpLogger:
    """Swallow yt-dlp's own console output; its errors still surface as exceptions."""

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass
