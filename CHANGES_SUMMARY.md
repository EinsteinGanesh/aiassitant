# Changes Summary: YouTube Music Integration Fix

## Problem Analysis

The original code had several issues preventing YouTube Music playback:

1. **Used JioSaavn APIs instead of YouTube Music** - The `download_youtube_song` function only used JioSaavn APIs which are unreliable and don't provide YouTube Music content
2. **yt-dlp installed but unused** - Despite having yt-dlp in requirements, the code used `requests` + JioSaavn APIs
3. **No cookie authentication** - Large `cookies.txt` (850KB) existed but wasn't being used for YouTube Music authentication
4. **No autoplay implementation** - The `auto_play_next` flag existed but only used Gemini to suggest next song, not YouTube's native autoplay
5. **Missing ffmpeg** - Required for audio conversion but not installed in environment
6. **No error handling/fallback** - If one method failed, the entire music playback failed

## Solution Implemented

### 1. New YouTube Music Integration (`download_youtube_music_song`)

**File: `ai_assistant.py` (lines 159-320)**

- **Primary**: Uses `yt-dlp` with cookies to search and download from YouTube Music
- **Search**: `ytsearch` extractor to find songs on YouTube Music
- **Download**: Best audio quality (m4a, 256kbps+)
- **Conversion**: ffmpeg to 16kHz mono WAV for ESP32
- **Fallback**: Graceful fallback to JioSaavn APIs if YouTube Music fails
- **Metadata**: Returns song title, artist, video_id, duration, source

### 2. Autoplay System

**New Components:**
- `music_queue` (asyncio.Queue) - Pre-fetched song queue
- `prefetch_autoplay_songs()` - Background task to fetch similar songs
- `play_next_autoplay_song()` - Plays next song from queue
- `get_autoplay_suggestions()` - Searches YouTube for similar songs
- `autoplay_enabled` flag - Toggle autoplay on/off

**Flow:**
1. Song finishes playing → `auto_play_next = True`
2. `prefetch_autoplay_songs()` runs in background during playback
3. When `auto_play_next` triggers, `play_next_autoplay_song()` gets next from queue
4. If queue empty, fetches new suggestions on-demand
5. Sets `current_song_query` to trigger download/playback

### 3. Cookie Authentication

- Reads `cookies.txt` (exported from browser)
- Passes to yt-dlp via `cookiefile` option
- Enables access to:
  - YouTube Music Premium content
  - Private/age-restricted content
  - Region-locked content
  - Higher quality streams

### 4. Robust Error Handling

- **Try/except** at every network operation
- **Timeout handling** for downloads (30s) and ffmpeg (30s)
- **Graceful degradation**: YouTube Music -> JioSaavn -> Error
- **Detailed logging**: Shows which API/method is being tried
- **File cleanup**: Removes temp files on success/failure

### 5. Infrastructure Improvements

| File | Changes |
|------|---------|
| `requirements.txt` | Added `pydub` for audio processing |
| `Dockerfile` | Updated to Python 3.12, added ffmpeg, libsndfile1, non-root user |
| `start.sh` | New startup script with validation checks |
| `README.md` | Complete rewrite with setup instructions, troubleshooting |
| `.gitignore` | Added cookies.txt, temp audio files, IDE/OS files |
| `test_youtube_music.py` | New comprehensive test suite |

## Key Code Changes

### `download_youtube_music_song(query)` - Primary Function

```python
def download_youtube_music_song(query):
    """
    Download a song from YouTube Music using yt-dlp with cookies.
    Returns (wav_file_path, song_title, metadata_dict)
    """
    # 1. Search YouTube Music via yt-dlp
    search_results = search_youtube_music(query, max_results=3)
    
    # 2. Download best audio format
    ydl_opts = get_ydl_opts("youtube_audio.%(ext)s")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://music.youtube.com/watch?v={video_id}"])
    
    # 3. Convert to ESP32 format (16kHz mono WAV)
    subprocess.run([
        "ffmpeg", "-y", "-i", downloaded_file,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "current_song.wav"
    ])
    
    # 4. Return metadata for autoplay
    return "current_song.wav", title, {
        'title': title,
        'artist': uploader,
        'video_id': video_id,
        'duration': duration,
        'source': 'youtube_music'
    }
```

### Autoplay Integration in Main Loop

```python
if auto_play_next:
    auto_play_next = False
    played = await play_next_autoplay_song()
    if not played:
        update_esp32_screen("IDL")
```

### Background Prefetching

```python
async def prefetch_autoplay_songs():
    """Pre-fetch autoplay song suggestions in background"""
    title = last_song_metadata.get('title', '')
    artist = last_song_metadata.get('artist', '')
    suggestions = get_autoplay_suggestions(title, artist, count=3)
    for s in suggestions:
        await music_queue.put(s)
```

## How to Use

### 1. Prerequisites
```bash
# System dependencies
sudo apt-get install ffmpeg

# Python dependencies
pip install -r requirements.txt
pip install -U --pre "yt-dlp[default]"
```

### 2. Get YouTube Cookies (Critical!)
1. Install cookie exporter extension
2. Go to music.youtube.com, sign in
3. Export cookies for youtube.com and music.youtube.com
4. Save as `cookies.txt` in project root

### 3. Set Environment
```bash
export GEMINI_API_KEY="your-api-key"
```

### 4. Test Before Running
```bash
python test_youtube_music.py
```

### 5. Run Assistant
```bash
./start.sh
# OR
python ai_assistant.py
```

## Testing Checklist

Run `python test_youtube_music.py` and verify:
- [ ] ffmpeg found
- [ ] cookies.txt exists with YouTube domains
- [ ] yt-dlp search works
- [ ] YouTube Music search returns results
- [ ] Song download produces valid WAV
- [ ] Autoplay suggestions work

## Voice Commands (Tamil)

| Say | Action |
|-----|--------|
| "Hey Jarvis" | Wake word |
| "Play Arijit Singh song" | Play Arijit Singh song |
| "Next song" | Skip to next (autoplay) |
| "Pause" | Pause music |
| "Resume" | Resume music |
| "Volume up" | Volume up |
| "Volume down" | Volume down |

## Architecture Diagram

```
User Voice -> Wake Word -> Gemini Live API
                    |
            Tool Call: play_music("song name")
                    |
         download_youtube_music_song()
                    |
         yt-dlp + cookies -> YouTube Music
                    |
              ffmpeg -> 16kHz mono WAV
                    |
         stream_music_task() -> WebSocket -> ESP32 Speaker
                    |
            Song ends -> auto_play_next = True
                    |
         prefetch_autoplay_songs() (background)
                    |
         play_next_autoplay_song() -> Repeat
```

## Migration Notes

### Breaking Changes
- Function renamed: `download_youtube_song` -> `download_youtube_music_song`
- Return value: Now returns 3-tuple `(wav_path, title, metadata)` instead of 2-tuple
- Global `last_song_metadata` dict added for autoplay context

### Backward Compatibility
- JioSaavn fallback preserved as `download_jiosaavn_fallback()`
- Original `play_music` tool declaration unchanged
- All existing voice commands work identically

## Performance Considerations

1. **Search + Download**: ~5-15 seconds depending on network
2. **Conversion**: ~2-5 seconds via ffmpeg
3. **Autoplay Prefetch**: Runs in background during playback (no user delay)
4. **Memory**: Only one song cached at a time (~5-15MB)
5. **Cookies**: Re-export weekly for continued access

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Sign in required" | Re-export cookies.txt from logged-in browser |
| "Video unavailable" | yt-dlp auto-retries; update: `pip install -U --pre yt-dlp` |
| "ffmpeg not found" | Install ffmpeg system package |
| "No audio on ESP32" | Check WebSocket port 8000, ESP32 IP, audio format |
| "Autoplay not working" | Check `autoplay_enabled=True`, music_queue not empty |

## Future Enhancements

- [ ] YouTube Music playlist support
- [ ] Lyrics display on ESP32 screen
- [ ] Voice-controlled queue management
- [ ] Multi-room audio sync
- [ ] Offline caching of favorite songs
- [ ] Integration with YouTube Music API (when available)

---

**Author**: Senior Developer Implementation
**Date**: 2026-11-23
**Version**: 2.0.0 - YouTube Music Edition