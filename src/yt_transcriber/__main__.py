"""Allow ``python -m yt_transcriber``, equivalent to the ``yt-transcript`` command."""
import sys

from .transcript import main

sys.exit(main())
