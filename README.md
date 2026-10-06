# YT-Transcriber

[![CI](https://github.com/diego-barreca-2004/YT-Transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/diego-barreca-2004/YT-Transcriber/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Command-line tools to get the transcript of YouTube videos: one video, a whole
playlist, or a list of URLs, as plain text, JSON, SRT subtitles or Markdown.

When a video has no captions at all, YT-Transcriber downloads the audio and
transcribes it locally with [Whisper](https://github.com/SYSTRAN/faster-whisper),
on your GPU if you have one, otherwise on the CPU.

```console
$ yt-transcript https://youtu.be/dQw4w9WgXcQ
[♪♪♪] ♪ We're no strangers to love ♪ ♪ You know the rules
and so do I ♪ ♪ A full commitment's
...

$ yt-playlist --playlist "https://www.youtube.com/playlist?list=PL0BAFA2DA294B0900" --format srt
[1/10] dhzPnX07jAs: ok
...
Summary: 10 videos, 10 succeeded (10 from YouTube captions, 0 with Whisper), 0 failed.
```

## Features

- **Any YouTube link:** `watch?v=`, `youtu.be`, Shorts, live, embed, music.youtube.com, or a bare video ID.
- **Smart language choice:** English when available, otherwise the video's own language; `--lang` to pick one.
- **Four output formats:** `txt`, `json` (timed segments + metadata), `srt`, `md`.
- **Batches in parallel:** playlists, text files with one URL per line, or many URLs at once, with duplicates removed. One broken video never stops the others.
- **Whisper fallback** for videos with captions disabled, using the GPU (CUDA) when available. Optional, so the basic install stays small.
- **Rate-limit handling:** when YouTube temporarily blocks your IP, requests are retried with exponential backoff. Permanent errors are never retried.
- Clear error messages and meaningful exit codes, so it works well in scripts.

## Installation

You need **Python 3.11 or newer**. The commands below install YT-Transcriber
in its own isolated environment with [pipx](https://pipx.pypa.io/) (recommended)
or [uv](https://docs.astral.sh/uv/). Both work on Linux, macOS and Windows.

With the Whisper fallback (about 300 MB of dependencies, plus the Whisper model
downloaded on first use):

```bash
pipx install "yt-transcriber[whisper] @ git+https://github.com/diego-barreca-2004/YT-Transcriber.git"
# or
uv tool install "yt-transcriber[whisper] @ git+https://github.com/diego-barreca-2004/YT-Transcriber.git"
```

Captions only (small install; videos without captions fail with a clear message):

```bash
pipx install "git+https://github.com/diego-barreca-2004/YT-Transcriber.git"
```

You can also use plain pip inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "yt-transcriber[whisper] @ git+https://github.com/diego-barreca-2004/YT-Transcriber.git"
```

To update later (YouTube changes often, and yt-dlp follows those changes quickly):

```bash
pipx upgrade yt-transcriber        # or: uv tool upgrade yt-transcriber
```

If the `yt-transcript` command is not found after installing, run
`pipx ensurepath` and open a new terminal, or use `python -m yt_transcriber`
instead.

## Usage

### One video: `yt-transcript`

```bash
yt-transcript dQw4w9WgXcQ                                 # print plain text
yt-transcript "https://youtu.be/dQw4w9WgXcQ" --lang de    # pick a language
yt-transcript dQw4w9WgXcQ --format srt -o subtitles.srt   # save SubRip subtitles
yt-transcript dQw4w9WgXcQ --format md -o transcript.md    # readable Markdown document
yt-transcript dQw4w9WgXcQ --format json > transcript.json # timed segments + metadata
```

### Many videos: `yt-playlist`

Sources can be combined freely; the result is de-duplicated.

```bash
# A playlist: one file per video in ./transcripts
yt-playlist --playlist "https://www.youtube.com/playlist?list=PL0BAFA2DA294B0900"

# A text file with one URL or ID per line ('#' starts a comment), saved as Markdown
yt-playlist --file videos.txt --format md --output-dir notes/

# Several URLs, printed to stdout instead of saved
yt-playlist dQw4w9WgXcQ https://youtu.be/9bZkp7q19f0 --stdout

# Only list the video IDs of a playlist, without fetching anything
yt-playlist --list-only --playlist "https://www.youtube.com/playlist?list=PL0BAFA2DA294B0900"
```

With `--stdout --format json`, the whole batch is printed as a single JSON
object keyed by video ID, including the failed videos and their errors.

### Options

| Option | Commands | Description |
| --- | --- | --- |
| `--lang CODE` | both | Preferred language (`en`, `it`, `pt-BR`, ...). Default: English, then the first available language. |
| `--format {txt,json,srt,md}` | both | Output format. Default: `txt`. |
| `-o, --output PATH` | `yt-transcript` | Write to a file instead of stdout. |
| `-d, --output-dir DIR` | `yt-playlist` | Directory for `<video_id>.<format>` files. Default: `./transcripts`. |
| `--stdout` | `yt-playlist` | Print everything to stdout instead of writing files. |
| `--file PATH` | `yt-playlist` | Read video URLs or IDs from a text file. |
| `--playlist URL` | `yt-playlist` | Add every video of a playlist. |
| `--list-only` (`--json`) | `yt-playlist` | Only print the resolved video IDs (as a JSON array). |
| `-j, --concurrency N` | `yt-playlist` | Videos processed in parallel. Default: 4. |
| `--no-whisper-fallback` | both | Fail instead of transcribing videos without captions. |
| `--whisper-model NAME` | both | `tiny`, `base` (default), `small`, `medium`, `large-v3` or `turbo`. |
| `--whisper-device {auto,cpu,cuda}` | both | Where Whisper runs. Default: `auto` (GPU if available, else CPU). |
| `--retries N` | both | Extra attempts when YouTube rate-limits your IP. Default: 2. |
| `--retry-delay SECONDS` | both | First retry delay, doubled at each attempt. Default: 5. |

Run `yt-transcript --help` or `yt-playlist --help` for the full reference.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success (for `yt-playlist`: at least one video succeeded). |
| `1` | The transcript could not be retrieved (for `yt-playlist`: every video failed). |
| `2` | Invalid arguments, for example a malformed URL. |

## How it works

1. **Captions first.** [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
   lists the captions of the video and picks one: the language you asked for
   (manual captions before auto-generated ones), or, by default, English,
   then the first manual track, then the first auto-generated track.
2. **Whisper fallback.** If the video has captions disabled, [yt-dlp](https://github.com/yt-dlp/yt-dlp)
   downloads only a low-bitrate audio track to a temporary folder and
   [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcribes it.
   The audio file is always deleted afterwards. In a batch, downloads run in
   parallel but only one Whisper model runs at a time, to avoid running out of
   GPU memory.
3. **Retries.** Only YouTube's temporary IP blocks are retried (5 s, 10 s, 20 s, ...
   with random jitter). Errors such as "video unavailable" fail immediately.

## Whisper and GPUs

- The first time you use a model, it is downloaded from Hugging Face and cached
  (roughly: `tiny` 75 MB, `base` 145 MB, `small` 480 MB, `medium` 1.5 GB,
  `turbo` 1.6 GB, `large-v3` 3 GB). Bigger models are more accurate and slower.
- **NVIDIA GPU (Linux/Windows):** Whisper uses it automatically when the CUDA 12
  cuBLAS and cuDNN 9 libraries are installed. See the
  [faster-whisper GPU instructions](https://github.com/SYSTRAN/faster-whisper#gpu).
  If the GPU fails (for example, not enough memory), the tool says so and
  retries on the CPU.
- **CPU (any computer, including Apple Silicon Macs):** works out of the box,
  just slower. `tiny` or `base` keep it fast.

## Troubleshooting

- **"YouTube is temporarily blocking requests from this IP address":** you
  made many requests in a short time (or you are on a cloud/VPN IP that YouTube
  blocks). Wait a while, lower `--concurrency`, or switch network.
- **Audio download fails (HTTP 403 or "format not available"):** update
  first (`pipx upgrade yt-transcriber`). yt-dlp also works best with a
  JavaScript runtime installed, ideally [Deno](https://deno.com/); see the
  [yt-dlp wiki](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
- **The program crashes with "Could not load library libcudnn..." or another
  CUDA error:** the GPU libraries are missing or incompatible. Use
  `--whisper-device cpu`, or install them as described above.
- **A video ID that starts with `-`** is read as an option. Pass the full URL
  instead, or put `--` before it: `yt-transcript -- -abc123def45`.

## Development

```bash
git clone https://github.com/diego-barreca-2004/YT-Transcriber.git
cd YT-Transcriber
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[whisper,dev]"
pytest
```

The test suite runs offline. [docs/manual-testing.md](docs/manual-testing.md)
lists real videos that cover every code path, for checks against the live
YouTube site. Issues and pull requests are welcome.

## Disclaimer

YT-Transcriber is an independent project. It is not affiliated with, endorsed
by or sponsored by YouTube or Google. It relies on undocumented YouTube
endpoints (through youtube-transcript-api and yt-dlp), so it can stop working
when YouTube changes them.

You are responsible for how you use it. Respect
[YouTube's Terms of Service](https://www.youtube.com/t/terms) and the copyright of
the content creators. Use it for personal, educational, research or
accessibility purposes, and do not redistribute transcripts of content you do
not have the rights to.

## License

[MIT](LICENSE) © 2026 Diego Barreca.

YT-Transcriber does not bundle any third-party code. Its dependencies are installed
separately and keep their own licenses: youtube-transcript-api (MIT),
yt-dlp (Unlicense), faster-whisper (MIT), CTranslate2 (MIT) and PyAV (BSD-3-Clause).
Whisper models are released by OpenAI under the MIT license.
