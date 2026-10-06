import pytest

from yt_transcriber.playlist import extract_playlist_id
from yt_transcriber.transcript import extract_video_id

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "value",
    [
        VIDEO_ID,
        f"  {VIDEO_ID}  ",
        f"https://www.youtube.com/watch?v={VIDEO_ID}",
        f"https://www.youtube.com/watch?v={VIDEO_ID}&t=42s&list=PL123",
        f"http://youtube.com/watch?feature=share&v={VIDEO_ID}",
        f"www.youtube.com/watch?v={VIDEO_ID}",
        f"youtube.com/watch?v={VIDEO_ID}",
        f"https://m.youtube.com/watch?v={VIDEO_ID}",
        f"https://music.youtube.com/watch?v={VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}",
        f"https://youtu.be/{VIDEO_ID}?t=10",
        f"youtu.be/{VIDEO_ID}",
        f"https://www.youtube.com/embed/{VIDEO_ID}",
        f"https://www.youtube-nocookie.com/embed/{VIDEO_ID}",
        f"youtube-nocookie.com/embed/{VIDEO_ID}",
        f"https://www.youtube.com/shorts/{VIDEO_ID}",
        f"https://www.youtube.com/live/{VIDEO_ID}?si=abc",
        f"https://www.youtube.com/v/{VIDEO_ID}",
        f"HTTPS://WWW.YOUTUBE.COM/watch?v={VIDEO_ID}",
    ],
)
def test_extract_video_id_accepts_supported_forms(value):
    assert extract_video_id(value) == VIDEO_ID


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "too-short",
        "0000000000",  # 10 characters
        "not a url://x",
        "dQw4w9WgXcQ!",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/channel/UC123",
        "https://youtu.be/",
    ],
)
def test_extract_video_id_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        extract_video_id(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("https://www.youtube.com/playlist?list=PL0BAFA2DA294B0900", "PL0BAFA2DA294B0900"),
        ("youtube.com/playlist?list=PL0BAFA2DA294B0900", "PL0BAFA2DA294B0900"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc", "PLabc"),
        ("PL0BAFA2DA294B0900", "PL0BAFA2DA294B0900"),
    ],
)
def test_extract_playlist_id(value, expected):
    assert extract_playlist_id(value) == expected


@pytest.mark.parametrize("value", ["", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
def test_extract_playlist_id_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        extract_playlist_id(value)
