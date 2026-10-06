# Manual testing against real videos

The automated test suite (`pytest`) runs offline. Before a release, it is worth
checking the tools against real YouTube videos, because YouTube changes its
pages and endpoints often. These videos cover every code path and have been
stable for years.

| Video ID | Video | What it exercises |
| --- | --- | --- |
| `dQw4w9WgXcQ` | Rick Astley – Never Gonna Give You Up | Manual captions in several languages (en, de-DE, ja, pt-BR, es-419) plus auto-generated English |
| `9bZkp7q19f0` | PSY – Gangnam Style | Only auto-generated Korean captions: tests the language fallback and `--lang` errors |
| `aqz-KE-bpKQ` | Big Buck Bunny | Captions disabled: triggers the Whisper fallback |
| `00000000000` | – | Well-formed ID of a video that does not exist: "unavailable" error |
| `jfKfPfyJRdk` | Lofi Girl live stream | Unplayable video: a per-video error that must not stop a batch |

Playlist: `https://www.youtube.com/playlist?list=PL0BAFA2DA294B0900`
("Mango Open Movie" by the Blender Foundation, 10 videos, published in 2009).

## Checklist

```bash
yt-transcript dQw4w9WgXcQ                       # English manual captions
yt-transcript dQw4w9WgXcQ --lang ja --format srt
yt-transcript 9bZkp7q19f0                       # falls back to Korean auto-generated
yt-transcript 9bZkp7q19f0 --lang it             # error listing the available languages
yt-transcript aqz-KE-bpKQ --whisper-model tiny  # Whisper fallback
yt-transcript 00000000000                       # "unavailable" error, exit code 1
yt-transcript jfKfPfyJRdk                       # "not playable" error, exit code 1

yt-playlist --list-only --playlist "https://www.youtube.com/playlist?list=PL0BAFA2DA294B0900"
yt-playlist dQw4w9WgXcQ 9bZkp7q19f0 aqz-KE-bpKQ 00000000000 jfKfPfyJRdk \
    --format md --whisper-model tiny            # 3 succeeded, 2 failed, exit code 0
```

## Notes

- Big Buck Bunny has almost no speech, so Whisper output for it is mostly
  hallucinated filler. Use it to check that the fallback runs and produces a
  file, not to judge accuracy. To judge accuracy, run Whisper on
  `dQw4w9WgXcQ` directly and compare it with the English captions:

  ```python
  from yt_transcriber.whisper_fallback import transcribe_with_whisper
  segments, meta = transcribe_with_whisper("dQw4w9WgXcQ", None, "base")
  ```

- Without signing in, YouTube reports private, removed and non-existent
  videos the same way, so the tool cannot tell them apart either.
- If many requests are made in a short time, YouTube blocks the caption
  endpoint for your IP address for a while (`RequestBlocked`). Playlist
  expansion uses a different endpoint and usually keeps working. Switching
  network (for example to a phone hotspot) gets a new IP address.
