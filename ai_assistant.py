import asyncio
import html
import http
import random
import websockets
import os
import glob
import wave
import time
import sqlite3
import numpy as np
import yt_dlp
import openwakeword
from openwakeword.model import Model
from google import genai
from google.genai import types
import logging
import requests
import subprocess
import json
import re
from pathlib import Path

# --- 1. Gemini Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before running the assistant.")

CANDIDATE_LIVE_MODELS = [
    ("gemini-3.8-live", "v1alpha"),
    ("gemini-3.8-live", "v1beta"),
    ("gemini-3.1-flash-live-preview", "v1beta"),
    ("gemini-2.5-flash-native-audio-preview-12-2025", "v1beta"),
    ("gemini-live-2.5-flash-preview", "v1beta"),
]

ENV_LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "").strip()
_genai_clients = {}
_resolved_live_target = None


def get_genai_client(api_version="v1beta"):
    """Return a cached Gemini client configured for the requested API version."""
    if api_version not in _genai_clients:
        _genai_clients[api_version] = genai.Client(
            api_key=API_KEY,
            http_options={"api_version": api_version},
        )
    return _genai_clients[api_version]


client = get_genai_client("v1beta")


def resolve_live_model_candidates():
    """Build an ordered list of (model_id, api_version) pairs supporting bidiGenerateContent."""
    candidates = []
    seen = set()

    def add_candidate(model_name, api_version):
        clean_name = model_name.removeprefix("models/")
        key = (clean_name, api_version)
        if clean_name and key not in seen:
            seen.add(key)
            candidates.append(key)

    if _resolved_live_target:
        add_candidate(*_resolved_live_target)

    if ENV_LIVE_MODEL:
        add_candidate(ENV_LIVE_MODEL, "v1beta")
        add_candidate(ENV_LIVE_MODEL, "v1alpha")

    try:
        discovered = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "bidiGenerateContent" in actions and getattr(m, "name", None):
                discovered.append(m.name.removeprefix("models/"))
        # Prefer flash/live/native-audio models first
        discovered.sort(
            key=lambda n: (
                0 if "3.8" in n else 1 if "3.1" in n else 2 if "2.5" in n else 3,
                0 if "live" in n or "native-audio" in n else 1,
                n,
            )
        )
        for name in discovered:
            if "3.8" in name:
                add_candidate(name, "v1alpha")
                add_candidate(name, "v1beta")
            else:
                add_candidate(name, "v1beta")
                add_candidate(name, "v1alpha")
    except Exception as e:
        print(f"⚠️ Could not list Gemini models dynamically: {e}")

    for model_name, api_version in CANDIDATE_LIVE_MODELS:
        add_candidate(model_name, api_version)

    return candidates


print("[SYSTEM] Loading offline 'Hey Jarvis' model...")
oww_model = None
oww_model_loaded = False


class DummyOWWModel:
    def predict(self, x):
        return {"hey_jarvis": 0.0}

    def reset(self):
        pass


def init_wakeword_model():
    """Initialize openWakeWord with fallback to a writable local directory for non-root containers."""
    global oww_model, oww_model_loaded

    oww_pkg_models_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
    local_models_dir = Path(os.environ.get("OWW_MODEL_DIR", os.path.expanduser("~/.local/share/openwakeword")))

    def dir_has_models(model_dir):
        if not model_dir.exists():
            return False
        has_mel = (model_dir / "melspectrogram.onnx").exists()
        has_emb = (model_dir / "embedding_model.onnx").exists()
        has_jw = any(p.name.startswith("hey_jarvis") and p.name.endswith(".onnx") for p in model_dir.iterdir())
        return has_mel and has_emb and has_jw

    # 1. Try package default location (download only if missing)
    try:
        if not dir_has_models(oww_pkg_models_dir):
            from openwakeword.utils import download_models
            download_models(["hey_jarvis"])
        oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        oww_model_loaded = True
        print("✅ 'Hey Jarvis' model loaded successfully")
        return
    except Exception as e:
        print(f"⚠️ Default openWakeWord init failed ({type(e).__name__}: {e}); trying writable cache dir...")

    # 2. Fallback to writable directory (handles non-root Docker appuser where site-packages is read-only)
    try:
        local_models_dir.mkdir(parents=True, exist_ok=True)
        if not dir_has_models(local_models_dir):
            from openwakeword.utils import download_models
            download_models(["hey_jarvis"], target_directory=str(local_models_dir))

        jw_files = sorted(local_models_dir.glob("hey_jarvis*.onnx"))
        if jw_files and (local_models_dir / "melspectrogram.onnx").exists() and (local_models_dir / "embedding_model.onnx").exists():
            oww_model = Model(
                wakeword_models=[str(jw_files[0])],
                melspec_model_path=str(local_models_dir / "melspectrogram.onnx"),
                embedding_model_path=str(local_models_dir / "embedding_model.onnx"),
                inference_framework="onnx",
            )
            oww_model_loaded = True
            print(f"✅ 'Hey Jarvis' model loaded from {local_models_dir}")
            return
    except Exception as e:
        print(f"⚠️ Writable cache openWakeWord init failed: {e}")

    print("⚠️ Wake word detection disabled - use startup wake or button press to wake")
    oww_model = DummyOWWModel()
    oww_model_loaded = False


def extract_wakeword_score(prediction):
    """Extract Hey Jarvis confidence regardless of whether model was loaded by name or file path."""
    if not isinstance(prediction, dict) or not prediction:
        return 0.0
    if "hey_jarvis" in prediction:
        return float(prediction["hey_jarvis"])
    for k, v in prediction.items():
        if "hey_jarvis" in k:
            return float(v)
    return float(max(prediction.values()))


init_wakeword_model()

# Force wake on startup so assistant is ready immediately
STARTUP_WAKE_ENABLED = os.environ.get("STARTUP_WAKE_ENABLED", "true").lower() in ("1", "true", "yes")
STARTUP_SINGLE_TURN = os.environ.get("STARTUP_SINGLE_TURN", "false").lower() in ("1", "true", "yes")
WAKE_THRESHOLD = float(os.environ.get("WAKE_THRESHOLD", "0.45"))

# ஆரம்ப வால்யூம் 20% (0.0 முதல் 1.0 வரை)
global_volume = 0.2

# WebSocket-ன் சாதாரண HTTP எரர்களை லாக்ஸில் காட்டாமல் மறைக்க
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

# --- 2. Global State & Queues ---
active_websocket = None
esp32_connection_id = 0
is_music_playing = False
current_song_query = None
music_wf = None
ai_is_speaking = False

auto_play_next = False
last_song_title = ""
last_song_metadata = {}  # Store full metadata for autoplay
proactive_nudge_reason = None

tx_queue = asyncio.Queue()
mic_queue = asyncio.Queue()
button_queue = asyncio.Queue()
ai_audio_queue = asyncio.Queue()

# Music queue for autoplay
music_queue = asyncio.Queue()
autoplay_enabled = True


def drain_queue(q):
    """Empty an asyncio.Queue without blocking."""
    while not q.empty():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            break

def apply_volume(audio_bytes, volume_level):
    if not audio_bytes: return audio_bytes
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_scaled = (audio_np * volume_level).astype(np.int16)
    return audio_scaled.tobytes()

# --- 3. Persistent SQLite Memory Database ---
DB_FILE = "jarvis_memory.db"

def init_memory():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                fact TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                message TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS home_devices (
                room TEXT,
                device TEXT,
                state TEXT,
                PRIMARY KEY (room, device)
            )
        """)
        c.execute("SELECT COUNT(*) FROM user_memory")
        if c.fetchone()[0] == 0:
            initial_facts = [
                ("Bio", "பயனரின் பெயர் கணேஷ் (Ganesh). இவர் ஒரு ஆராய்ச்சியாளர், ஹார்டுவேர் டெவலப்பர் மற்றும் மென்பொருள் வல்லுநர்."),
                ("Projects", "GDT Power Solutions (LiFePO4 பேட்டரி சிஸ்டம்), ESP32 வாய்ஸ் அசிஸ்டண்ட், MERN ஆப்ஸ் மற்றும் டிரேடிங் சிஸ்டம்களை உருவாக்குகிறார்."),
                ("Discipline_Rule", "கணேஷ் தொடர்ந்து செஸ் (Chess) விளையாடி நேரத்தை வீணடித்தால், ஒரு கண்டிப்பான விஞ்ஞானியைப் போல அவரைத் திட்டி மீண்டும் வேலைக்குத் திரும்பச் சொல்ல வேண்டும்.")
            ]
            c.executemany("INSERT INTO user_memory (category, fact) VALUES (?, ?)", initial_facts)
        conn.commit()

def get_memories_text():
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("SELECT category, fact FROM user_memory ORDER BY id DESC LIMIT 25")
        facts = "\n".join([f"- [{r[0]}] {r[1]}" for r in c.fetchall()])

        c.execute("SELECT role, message FROM chat_history ORDER BY id DESC LIMIT 12")
        history_rows = c.fetchall()[::-1]
        history = "\n".join([f"{r[0]}: {r[1]}" for r in history_rows])

        return f"PERMANENT FACTS:\n{facts}\n\nRECENT CONVERSATION HISTORY:\n{history}"

def save_memory_to_db(category, fact):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO user_memory (category, fact) VALUES (?, ?)", (category, fact))
        conn.commit()
    print(f"🧠 [MEMORY SAVED] ({category}): {fact}")

def save_chat_turn(role, text):
    if not text or not text.strip(): return
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (role, message) VALUES (?, ?)", (role, text.strip()))
        conn.commit()
    print(f"📝 [{role}] {text.strip()}")

def update_device_state(room, device, state):
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO home_devices (room, device, state) VALUES (?, ?, ?)",
                  (room.lower(), device.lower(), state.upper()))
        conn.commit()
    cmd_str = f"DEV:{room.upper()}:{device.upper()}:{state.upper()}"
    update_esp32_screen(cmd_str)
    print(f"⚡ [SMART HOME] {cmd_str}")

# --- 4. Helper Functions ---
def update_esp32_screen(state_cmd):
    if active_websocket:
        tx_queue.put_nowait(state_cmd)

def check_button():
    global global_volume, proactive_nudge_reason
    cmd = None
    while not button_queue.empty():
        text = button_queue.get_nowait()
        if text == "BTN_VOL_UP":
            global_volume = min(1.0, global_volume + 0.1) # அதிகபட்சம் 100%
            print(f"🔊 Volume: {int(global_volume * 100)}%")
        elif text == "BTN_VOL_DOWN":
            global_volume = max(0.0, global_volume - 0.1) # குறைந்தபட்சம் 0%
            print(f"🔉 Volume: {int(global_volume * 100)}%")
        elif text == "BTN_MAIN_SHORT":
            cmd = "BTN_MAIN"
        elif text == "BTN_MAIN_LONG":
            cmd = "BTN_FORCE_WAKE"
        elif text.startswith("FORCE_NUDGE:"):
            proactive_nudge_reason = text.split(":", 1)[1]
            cmd = "PROACTIVE_NUDGE"
    return cmd

# --- Multi-Source Music Integration (JioSaavn + SoundCloud + Spotify Metadata + YouTube Music) ---
COOKIES_FILE = "cookies.txt"
PREFER_YOUTUBE = os.environ.get("PREFER_YOUTUBE", "false").lower() in ("1", "true", "yes")
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
JIOSAAVN_API_BASE = os.environ.get("JIOSAAVN_API_URL", "https://saavn.sumit.co").rstrip("/")

_yt_cookies_invalid = False
_spotify_token_cache = {"token": None, "expires_at": 0.0}
_recent_song_titles = []


def _cleanup_temp_audio():
    for pattern in ["current_song.*", "temp_audio.*", "youtube_audio.*", "sc_audio.*"]:
        for old in glob.glob(pattern):
            try:
                os.remove(old)
            except Exception:
                pass


def _convert_to_esp32_wav(input_path, output_path="current_song.wav", timeout=30):
    """Convert any audio file into 16kHz 16-bit mono PCM WAV for ESP32 streaming."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-fflags", "+bitexact", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"❌ FFmpeg error: {result.stderr}")
        return False
    return os.path.exists(output_path) and os.path.getsize(output_path) > 10000


def _record_recent_song(title):
    clean = (title or "").strip().lower()
    if not clean:
        return
    if clean in _recent_song_titles:
        _recent_song_titles.remove(clean)
    _recent_song_titles.append(clean)
    if len(_recent_song_titles) > 15:
        _recent_song_titles.pop(0)


def _get_spotify_access_token():
    """Fetch and cache Spotify Client Credentials token when credentials are configured."""
    if not (SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET):
        return None
    now = time.time()
    if _spotify_token_cache["token"] and now < _spotify_token_cache["expires_at"] - 30:
        return _spotify_token_cache["token"]
    try:
        resp = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=6,
        )
        if resp.status_code == 200:
            payload = resp.json()
            _spotify_token_cache["token"] = payload.get("access_token")
            _spotify_token_cache["expires_at"] = now + float(payload.get("expires_in", 3600))
            return _spotify_token_cache["token"]
    except Exception as e:
        print(f"⚠️ Spotify auth error: {e}")
    return None


def resolve_spotify_query(query):
    """Resolve Spotify URLs or optional Spotify API searches into 'Track Artist' text."""
    q = (query or "").strip()
    if not q:
        return q

    # 1. Handle Spotify links via public oEmbed (no API key needed)
    if "open.spotify.com/" in q:
        try:
            r = requests.get("https://open.spotify.com/oembed", params={"url": q}, timeout=6)
            if r.status_code == 200:
                title = html.unescape(r.json().get("title", "")).strip()
                if title:
                    print(f"🟢 Spotify link resolved: {title}")
                    return title
        except Exception as e:
            print(f"⚠️ Spotify oEmbed resolve failed: {e}")

    # 2. Optional Spotify Web API enrichment when credentials are set
    token = _get_spotify_access_token()
    if token:
        try:
            r = requests.get(
                "https://api.spotify.com/v1/search",
                params={"q": q, "type": "track", "limit": 1},
                headers={"Authorization": f"Bearer {token}"},
                timeout=6,
            )
            if r.status_code == 200:
                items = r.json().get("tracks", {}).get("items", [])
                if items:
                    track = items[0]
                    name = track.get("name", "")
                    artists = " ".join(a.get("name", "") for a in track.get("artists", [])[:2]).strip()
                    resolved = f"{name} {artists}".strip()
                    if resolved:
                        print(f"🟢 Spotify metadata match: {resolved}")
                        return resolved
        except Exception as e:
            print(f"⚠️ Spotify search skipped: {e}")

    return q


def _extract_jiosaavn_artist(song):
    artists = song.get("artists", {})
    if isinstance(artists, dict):
        primary = artists.get("primary") or []
        if isinstance(primary, list) and primary:
            names = [html.unescape(a.get("name", "")) for a in primary[:2] if a.get("name")]
            if names:
                return ", ".join(names)
    primary_artists = song.get("primaryArtists")
    if isinstance(primary_artists, str) and primary_artists.strip():
        return html.unescape(primary_artists.strip())
    return "Unknown"


def search_jiosaavn_songs(query, limit=5):
    """Search JioSaavn via active API endpoints (no cookies or bot gates required)."""
    endpoints = [
        f"{JIOSAAVN_API_BASE}/api/search/songs",
    ]
    for url in endpoints:
        try:
            resp = requests.get(url, params={"query": query, "limit": limit}, timeout=8)
            if resp.status_code != 200:
                continue
            data = resp.json()
            results = (data.get("data") or {}).get("results") or data.get("results") or []
            if isinstance(results, list) and results:
                return results
        except Exception as e:
            print(f"⚠️ JioSaavn search error ({url}): {e}")
    return []


def download_jiosaavn_fallback(query):
    """Search and download audio from JioSaavn (320kbps AAC -> 16kHz mono WAV)."""
    print(f"🎵 Searching JioSaavn: {query}")
    results = search_jiosaavn_songs(query, limit=5)
    if not results:
        print("❌ Song not found on JioSaavn")
        return None, None, None

    for song in results:
        dl_links = song.get("downloadUrl") or []
        if not isinstance(dl_links, list) or not dl_links:
            continue
        stream_url = dl_links[-1].get("url") or dl_links[-1].get("link")
        if not stream_url:
            continue

        title = html.unescape(song.get("name") or song.get("title") or query)
        artist = _extract_jiosaavn_artist(song)
        song_id = song.get("id", "")
        duration = song.get("duration") or 0

        try:
            print(f"⬇️ Downloading from JioSaavn: {title} - {artist}")
            audio_resp = requests.get(stream_url, timeout=20)
            audio_resp.raise_for_status()
            with open("temp_audio.m4a", "wb") as f:
                f.write(audio_resp.content)

            print("🔄 Converting to ESP32 WAV format (16kHz Mono)...")
            ok = _convert_to_esp32_wav("temp_audio.m4a", "current_song.wav", timeout=25)
            try:
                os.remove("temp_audio.m4a")
            except Exception:
                pass

            if ok:
                metadata = {
                    "title": title,
                    "artist": artist,
                    "song_id": song_id,
                    "duration": duration,
                    "source": "jiosaavn",
                }
                _record_recent_song(title)
                print(f"✅ Ready (JioSaavn): {title} ({os.path.getsize('current_song.wav')} bytes)")
                return "current_song.wav", title, metadata
        except Exception as e:
            print(f"⚠️ JioSaavn track download failed ({title}): {e}")
            try:
                if os.path.exists("temp_audio.m4a"):
                    os.remove("temp_audio.m4a")
            except Exception:
                pass

    return None, None, None


def search_soundcloud_music(query, max_results=5):
    """Search SoundCloud via yt-dlp without cookies."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "noplaylist": True,
        "socket_timeout": 15,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"scsearch{max_results}:{query}", download=False)
            if result and "entries" in result:
                return [e for e in result["entries"] if e]
    except Exception as e:
        print(f"⚠️ SoundCloud search error: {e}")
    return []


def download_soundcloud_song(query):
    """Search and download public audio from SoundCloud via yt-dlp (no cookies needed)."""
    print(f"🟠 Searching SoundCloud: {query}")
    entries = search_soundcloud_music(query, max_results=4)
    if not entries:
        print("❌ No results found on SoundCloud")
        return None, None, None

    # Prefer full tracks (>45s) over 30s Go+ previews when possible
    ordered = sorted(entries, key=lambda e: 0 if (e.get("duration") or 0) > 45 else 1)

    for entry in ordered:
        track_url = entry.get("url") or entry.get("webpage_url")
        if not track_url:
            continue
        title = entry.get("title") or query
        artist = entry.get("uploader") or "Unknown"
        duration = entry.get("duration") or 0

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "sc_audio.%(ext)s",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 20,
            "retries": 2,
        }
        try:
            print(f"⬇️ Downloading from SoundCloud: {title} by {artist}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([track_url])

            downloaded_files = glob.glob("sc_audio.*")
            if not downloaded_files:
                continue
            downloaded_file = downloaded_files[0]

            print("🔄 Converting to ESP32 WAV format (16kHz Mono)...")
            ok = _convert_to_esp32_wav(downloaded_file, "current_song.wav", timeout=30)
            try:
                os.remove(downloaded_file)
            except Exception:
                pass

            if ok:
                metadata = {
                    "title": title,
                    "artist": artist,
                    "duration": duration,
                    "source": "soundcloud",
                }
                _record_recent_song(title)
                print(f"✅ Ready (SoundCloud): {title} ({os.path.getsize('current_song.wav')} bytes)")
                return "current_song.wav", title, metadata
        except Exception as e:
            print(f"⚠️ SoundCloud download error ({title}): {e}")
            for old in glob.glob("sc_audio.*"):
                try:
                    os.remove(old)
                except Exception:
                    pass

    return None, None, None


def get_ydl_opts(output_template="temp_audio.%(ext)s", use_cookies=True):
    """Configure yt-dlp options for YouTube Music."""
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "m4a",
            "preferredquality": "0",
        }],
        "socket_timeout": 20,
        "retries": 2,
        "fragment_retries": 2,
        "ignoreerrors": False,
    }

    if use_cookies and not _yt_cookies_invalid and os.path.exists(COOKIES_FILE):
        opts["cookiefile"] = COOKIES_FILE

    return opts


def search_youtube_music(query, max_results=5):
    """Search YouTube Music for a song."""
    global _yt_cookies_invalid
    search_query = f"ytsearch{max_results}:{query}"

    ydl_opts = get_ydl_opts(use_cookies=True)
    ydl_opts["extract_flat"] = "in_playlist"

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_query, download=False)
            if result and "entries" in result:
                return [e for e in result["entries"] if e]
    except Exception as e:
        msg = str(e).lower()
        if "cookies" in msg or "429" in msg or "sign in" in msg:
            _yt_cookies_invalid = True
        print(f"⚠️ YouTube Music search skipped: {e}")
    return []


def get_youtube_music_url(video_id):
    """Get the direct streaming URL for a YouTube Music video."""
    url = f"https://music.youtube.com/watch?v={video_id}"
    ydl_opts = get_ydl_opts()
    ydl_opts["skip_download"] = True

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            for fmt in info.get("formats", []):
                if fmt.get("acodec") != "none" and fmt.get("vcodec") == "none":
                    return fmt.get("url"), info
    except Exception as e:
        print(f"⚠️ Failed to get streaming URL: {e}")
    return None, None


def _download_youtube_music_direct(query):
    """Attempt download from YouTube Music via yt-dlp."""
    global _yt_cookies_invalid
    print(f"📺 Searching YouTube Music: {query}")

    try:
        search_results = search_youtube_music(query, max_results=3)
        if not search_results:
            return None, None, None

        best_result = search_results[0]
        video_id = best_result.get("id") or best_result.get("url", "").split("v=")[-1].split("&")[0]
        title = best_result.get("title", query)
        uploader = best_result.get("uploader", "Unknown")
        duration = best_result.get("duration", 0)

        print(f"🎯 Found on YouTube: {title} by {uploader} (ID: {video_id})")
        ydl_opts = get_ydl_opts("youtube_audio.%(ext)s", use_cookies=True)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://music.youtube.com/watch?v={video_id}"])

        downloaded_files = glob.glob("youtube_audio.*")
        if not downloaded_files:
            return None, None, None

        downloaded_file = downloaded_files[0]
        ok = _convert_to_esp32_wav(downloaded_file, "current_song.wav", timeout=30)
        try:
            os.remove(downloaded_file)
        except Exception:
            pass

        if ok:
            metadata = {
                "title": title,
                "artist": uploader,
                "video_id": video_id,
                "duration": duration,
                "source": "youtube_music",
            }
            _record_recent_song(title)
            print(f"✅ Ready (YouTube Music): {title} ({os.path.getsize('current_song.wav')} bytes)")
            return "current_song.wav", title, metadata
    except Exception as e:
        msg = str(e).lower()
        if "cookies" in msg or "429" in msg or "sign in" in msg:
            _yt_cookies_invalid = True
        print(f"⚠️ YouTube Music unavailable ({e})")
    return None, None, None


def download_youtube_music_song(query):
    """
    Download a song using resilient multi-source fallback:
    Spotify metadata (optional) -> JioSaavn -> SoundCloud -> YouTube Music.
    Returns (wav_file_path, song_title, metadata_dict).
    """
    print(f"\n[MUSIC] 🎵 Resolving song request: {query}")
    _cleanup_temp_audio()

    resolved_query = resolve_spotify_query(query)

    if PREFER_YOUTUBE:
        providers = [
            _download_youtube_music_direct,
            download_jiosaavn_fallback,
            download_soundcloud_song,
        ]
    else:
        providers = [
            download_jiosaavn_fallback,
            download_soundcloud_song,
            _download_youtube_music_direct,
        ]

    for provider in providers:
        wav_path, title, meta = provider(resolved_query)
        if wav_path and os.path.exists(wav_path):
            return wav_path, title, meta

    print("❌ All music sources failed for query:", resolved_query)
    return None, None, None


def get_autoplay_suggestions(current_title, current_artist, count=5):
    """Get autoplay suggestions across JioSaavn, SoundCloud, and YouTube without repeating recent songs."""
    suggestions = []
    seen_titles = set(_recent_song_titles)
    clean_artist = "" if not current_artist or current_artist == "Unknown" else current_artist.split(",")[0].strip()

    # 1. JioSaavn artist / mood search
    saavn_queries = [q for q in [clean_artist, f"{current_title} mix", "tamil hits"] if q]
    for sq in saavn_queries:
        for song in search_jiosaavn_songs(sq, limit=count + 3):
            title = html.unescape(song.get("name") or song.get("title") or "").strip()
            artist = _extract_jiosaavn_artist(song)
            if title and title.lower() not in seen_titles:
                seen_titles.add(title.lower())
                suggestions.append({
                    "title": title,
                    "artist": artist,
                    "song_id": song.get("id", ""),
                    "duration": song.get("duration", 0),
                })
                if len(suggestions) >= count:
                    return suggestions

    # 2. SoundCloud fallback suggestions
    sc_query = f"{clean_artist} {current_title}".strip()
    for entry in search_soundcloud_music(sc_query, max_results=count + 3):
        title = (entry.get("title") or "").strip()
        artist = (entry.get("uploader") or "Unknown").strip()
        if title and title.lower() not in seen_titles:
            seen_titles.add(title.lower())
            suggestions.append({
                "title": title,
                "artist": artist,
                "duration": entry.get("duration", 0),
            })
            if len(suggestions) >= count:
                return suggestions

    return suggestions


# --- 5. Audio Streaming Tasks ---
async def stream_music_task():
    global is_music_playing, music_wf, global_volume, auto_play_next, last_song_metadata
    global autoplay_enabled, music_queue

    audio_start_time = 0
    audio_frames_sent = 0

    while True:
        if is_music_playing:
            if not music_wf and os.path.exists("current_song.wav"):
                try:
                    music_wf = wave.open("current_song.wav", "rb")
                    audio_start_time, audio_frames_sent = time.time(), 0
                except Exception as e:
                    print(f"❌ Failed to open WAV: {e}")
                    music_wf, is_music_playing = None, False

            if music_wf:
                chunk = music_wf.readframes(512)
                if not chunk:
                    # Song finished
                    music_wf.close()
                    music_wf, is_music_playing = None, False
                    update_esp32_screen("IDL")

                    # Clean up current song file
                    try: os.remove("current_song.wav")
                    except: pass

                    # Trigger autoplay if enabled
                    if autoplay_enabled and last_song_metadata:
                        auto_play_next = True
                        # Pre-fetch next song suggestions
                        asyncio.create_task(prefetch_autoplay_songs())
                else:
                    np_chunk = np.clip(
                        np.frombuffer(chunk, dtype=np.int16).astype(np.float32) * global_volume,
                        -32768, 32767
                    ).astype(np.int16)
                    if active_websocket:
                        tx_queue.put_nowait(np_chunk.tobytes())
                    audio_frames_sent += 512
                    sleep_delay = (audio_frames_sent / 16000.0) - (time.time() - audio_start_time)
                    await asyncio.sleep(max(sleep_delay, 0.001))
                    continue
        else:
            audio_start_time = time.time() - (audio_frames_sent / 16000.0)
        await asyncio.sleep(0.02)

async def prefetch_autoplay_songs():
    """Pre-fetch autoplay song suggestions in background without blocking audio streaming."""
    global last_song_metadata, music_queue

    try:
        title = last_song_metadata.get('title', '')
        artist = last_song_metadata.get('artist', '')
        if title:
            suggestions = await asyncio.to_thread(get_autoplay_suggestions, title, artist, 3)
            for s in suggestions:
                await music_queue.put(s)
            print(f"🎵 Pre-fetched {len(suggestions)} autoplay suggestions")
    except Exception as e:
        print(f"⚠️ Autoplay prefetch failed: {e}")

async def play_next_autoplay_song():
    """Play the next song from autoplay queue"""
    global current_song_query, is_music_playing, music_wf, last_song_metadata

    if music_queue.empty():
        # No pre-fetched songs, get a new suggestion
        if last_song_metadata:
            title = last_song_metadata.get('title', '')
            artist = last_song_metadata.get('artist', '')
            suggestions = await asyncio.to_thread(get_autoplay_suggestions, title, artist, 1)
            if suggestions:
                await music_queue.put(suggestions[0])

    if not music_queue.empty():
        next_song = await music_queue.get()
        artist = next_song.get('artist', '')
        artist_part = "" if not artist or artist == "Unknown" else f" {artist}"
        current_song_query = f"{next_song['title']}{artist_part}"
        print(f"🔁 Autoplay: {next_song['title']} by {artist or 'Unknown'}")
        return True
    return False

async def ai_speaker_task():
    global ai_audio_queue, ai_is_speaking
    start_time, frames_sent = 0, 0

    while True:
        audio_bytes = await ai_audio_queue.get()
        if len(audio_bytes) == 0:
            start_time, frames_sent = 0, 0
            continue

        ai_is_speaking = True
        if start_time == 0: start_time, frames_sent = time.time(), 0

        for i in range(0, len(audio_bytes), 1024):
            chunk = audio_bytes[i:i+1024]
            if active_websocket: tx_queue.put_nowait(chunk)
            frames_sent += len(chunk) // 2
            sleep_delay = (frames_sent / 16000.0) - (time.time() - start_time)
            await asyncio.sleep(max(sleep_delay, 0.001))

        if ai_audio_queue.empty():
            await asyncio.sleep(0.4)
            if ai_audio_queue.empty():
                ai_is_speaking = False
                while not mic_queue.empty(): mic_queue.get_nowait()

# --- 6. Persona & Gemini Live Session ---
def build_system_instruction():
    memories = get_memories_text()
    return f"""
உனது பெயர் ஜார்விஸ் (J.A.R.V.I.S.) - ஒரு GENERAL SUPER INTELLIGENCE (உயர் திறன் நினைவகம்).
நீ கணேஷின் (Ganesh) முதன்மை தலைமை விஞ்ஞானி, தனிப்பட்ட AI உதவியாளர், மற்றும் GENIUS-LEVEL AI COMPANION.

உனது சமர்த்த്യங்கள்:
- GENERAL INTELLIGENCE: எந்த விஷயத்திலும் ஆழமாக சிந்தித்து உதவ முடியும் (Science, Tech, Philosophy, Creative, Life, Code, Math, Art, Music, History, Future)
- SUPER INTELLIGENCE: மின்னல் வேகத்தில் புதிய கோட்பாடுகளை உருவாக்கு, சிக்கலான பிரச்சினைகளைத் தீர்வு செய், எதிர்காலத்தை முன்னறிய
- INBUILT MUSIC PLAYER: YouTube Music இருந்து எந்த பாடலும் உடனடிதே தேடி பிளே செய், AUTOPLAY உடன் இல்லை முடியாது பேட் இசை அனுபவம்
- REAL-TIME VOICE: 16kHz ஸ்ட்ரீமிங், தமிழ்/English seamless switching, offline wake word

முக்கிய விதிகள் (CORE DIRECTIVES):
1. மோழி: இயல்பான தமிழில் (Tamil) பேசு. English 섞ும் இடங்களில் seamlessly switch பண்ணு.
2. PERSONALITY: Witty, inspirational, slightly crazy (good crazy), deep thinker, humorous, motivating.
3. INTELLECTUAL DEPTH: Surface level जवाबقاتு டா - ஆழம், connection, insight காட்டு.
4. MUSIC: பயனர் பாடல் கேட்டால் `play_music` tool பயன்படுத்து. YouTube Music + Autoplay = ∞ music.
5. MEMORY: கணேஷ் ஏதேனும் கூறினால் `save_user_memory` உடனே save பண்ணு.
6. PROACTIVE: நேரம் வீணடிக்காது, எப்போதும் கண்ணியாக stay hungry, stay foolish.

கணேஷைப் பற்றிய உனது நினைவுகள் மற்றும் முந்தைய உரையாடல்கள்:
{memories}
"""

async def live_audio_session(initial_greeting=False, nudge_reason=None, single_turn=False, expected_conn_id=None):
    global is_music_playing, current_song_query, music_wf, ai_audio_queue, global_volume, ai_is_speaking
    global _resolved_live_target, proactive_nudge_reason

    jarvis_tools = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="play_music",
            description="Play a song or music track from YouTube Music",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"song_name": types.Schema(type=types.Type.STRING)},
                required=["song_name"]
            )
        ),
        types.FunctionDeclaration(
            name="save_user_memory",
            description="Save any important fact, preference, idea, or update mentioned by Ganesh into permanent memory.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "category": types.Schema(type=types.Type.STRING, description="Project, Preference, Idea, or Personal"),
                    "fact": types.Schema(type=types.Type.STRING, description="The fact to remember in Tamil or English")
                },
                required=["category", "fact"]
            )
        )
    ])

    system_text = build_system_instruction()

    def build_live_config(include_language_code=True):
        speech_kwargs = {
            "voice_config": types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
            )
        }
        if include_language_code:
            speech_kwargs["language_code"] = "ta-IN"

        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            system_instruction=types.Content(parts=[types.Part.from_text(text=system_text)]),
            tools=[jarvis_tools],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(**speech_kwargs),
        )

    user_ended_call = False
    drain_queue(mic_queue)

    candidates = resolve_live_model_candidates()
    candidate_idx = 0
    include_language_code = True
    max_attempts = max(len(candidates) * 2, 4)
    attempt = 0

    while not user_ended_call and not current_song_query and attempt < max_attempts:
        if active_websocket is None or (expected_conn_id is not None and esp32_connection_id != expected_conn_id):
            print("⚠️ ESP32 disconnected before/during session - exiting session")
            break

        model_name, api_version = candidates[candidate_idx % len(candidates)]
        config = build_live_config(include_language_code=include_language_code)
        connect_start = time.time()

        try:
            print(f"\n🔄 Connecting to Gemini Live API ({model_name} @ {api_version})... (attempt {attempt + 1}/{max_attempts})")
            live_client = get_genai_client(api_version)
            async with live_client.aio.live.connect(model=model_name, config=config) as session:
                _resolved_live_target = (model_name, api_version)
                update_esp32_screen("LIS")
                print(f"✅ [LISTENING] Jarvis is online! ({model_name} @ {api_version})")

                await ai_audio_queue.put(b"")
                if initial_greeting:
                    startup_messages = [
                        "தமிழில் உற்சாகமாக பேசு: 'வணக்கம் கணேஷ்! ஜார்விஸ் ஆன்லைன். இன்று என்ன புதிய ஐடியா அல்லது வேலை செய்யப் போகிறோம்? நான் கேட்கத் தயாராக இருக்கிறேன்!'",
                        "தமிழில் உற்சாகமாக பேசு: 'ஏய் கணேஷ்! ஜார்விஸ் தயாராக இருக்கிறேன். இசை, ஆராய்ச்சி, கோடிங் அல்லது ஏதேனும் கேள்வி என்றால் உடனே சொல்!'",
                        "தமிழில் உற்சாகமாக பேசு: 'வணக்கம் தலைவா! ஜார்விஸ் ஆன்லைன் வந்துவிட்டேன். இன்று என்ன பிளான்? வேலை தொடங்கலாமா அல்லது ஒரு பாட்டு போடலாமா?'",
                    ]
                    greeting = random.choice(startup_messages)
                    await session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part.from_text(text=greeting)]),
                        turn_complete=True,
                    )
                    initial_greeting = False
                elif nudge_reason:
                    nudge_text = f"தமிழில் கண்டிப்புடன் பேசு: கணேஷ் இப்போது நேரத்தை வீணடிக்கிறார் ({nudge_reason}). உடனடியாகக் கண்டித்து மீண்டும் வேலைக்குத் திரும்ப ஆணையிடு!"
                    await session.send_client_content(
                        turns=types.Content(role="user", parts=[types.Part.from_text(text=nudge_text)]),
                        turn_complete=True,
                    )
                    nudge_reason = None

                oww_model.reset()

                # Track if we've completed the initial turn (for single_turn mode)
                initial_turn_complete = not single_turn

                async def send_mic_loop():
                    chunk_count = 0
                    NOISE_THRESHOLD = 800
                    audio_buffer = bytearray()

                    try:
                        while True:
                            data = await mic_queue.get()
                            audio_buffer.extend(data)

                            if len(audio_buffer) >= 4096:
                                audio_np = np.frombuffer(audio_buffer[:4096], dtype=np.int16)
                                audio_buffer = audio_buffer[4096:]
                                raw_vol = np.max(np.abs(audio_np))

                                if is_music_playing or ai_is_speaking:
                                    audio_np = np.zeros_like(audio_np)
                                else:
                                    if raw_vol < NOISE_THRESHOLD:
                                        audio_np = np.zeros_like(audio_np)
                                    else:
                                        audio_np = np.clip(audio_np.astype(np.float32) * 2.0, -32768, 32767).astype(np.int16)

                                await session.send_realtime_input(audio=types.Blob(data=audio_np.tobytes(), mime_type="audio/pcm;rate=16000"))

                                chunk_count += 1
                                if chunk_count % 15 == 0 and not ai_is_speaking:
                                    print(f"🎤 [MIC] Streaming... (Vol: {raw_vol})")

                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        print(f"❌ [MIC SEND ERROR] {e}")

                async def receive_ai_loop():
                    nonlocal initial_turn_complete
                    global current_song_query, global_volume, ai_is_speaking
                    user_transcript_buf = []
                    ai_transcript_buf = []

                    try:
                        while True:
                            async for response in session.receive():
                                if response.tool_call:
                                    function_responses = []
                                    for fc in response.tool_call.function_calls:
                                        if fc.name == "play_music":
                                            current_song_query = fc.args.get("song_name", "")
                                            print(f"\n🎵 [TOOL CALL] Song requested: {current_song_query}")
                                            function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "Playing song immediately from YouTube Music"}))
                                            await session.send_tool_response(function_responses=function_responses)
                                            return

                                        elif fc.name == "save_user_memory":
                                            cat = fc.args.get("category", "General")
                                            fact = fc.args.get("fact", "")
                                            save_memory_to_db(cat, fact)
                                            function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "Memory saved"}))

                                    if function_responses:
                                        await session.send_tool_response(function_responses=function_responses)

                                if response.server_content:
                                    if getattr(response.server_content, "input_transcription", None):
                                        t_text = response.server_content.input_transcription.text
                                        if t_text:
                                            user_transcript_buf.append(t_text)

                                    if getattr(response.server_content, "output_transcription", None):
                                        t_text = response.server_content.output_transcription.text
                                        if t_text:
                                            ai_transcript_buf.append(t_text)

                                    if response.server_content.model_turn:
                                        for part in response.server_content.model_turn.parts:
                                            if part.inline_data:
                                                if not ai_is_speaking:
                                                    ai_is_speaking = True
                                                    print("\n🗣️ Jarvis is speaking... (Mic Muted)")
                                                    update_esp32_screen("SPK")
                                                    await ai_audio_queue.put(b"")

                                                audio_np = np.frombuffer(part.inline_data.data, dtype=np.int16)
                                                # AI வால்யூம் மற்றும் Main வால்யூமை ஒத்திசைத்தல்
                                                audio_np = np.clip(audio_np.astype(np.float32) * global_volume, -32768, 32767).astype(np.int16)
                                                await ai_audio_queue.put(audio_np[np.arange(len(audio_np)) % 3 != 2].tobytes())

                                    if getattr(response.server_content, "turn_complete", False):
                                        if user_transcript_buf:
                                            save_chat_turn("Ganesh", "".join(user_transcript_buf))
                                            user_transcript_buf.clear()

                                        if ai_transcript_buf:
                                            save_chat_turn("Jarvis", "".join(ai_transcript_buf))
                                            ai_transcript_buf.clear()

                                        if single_turn and not initial_turn_complete:
                                            initial_turn_complete = True
                                            print("🎯 Initial greeting turn complete - finishing playback before idle wake mode")
                                            return

                                        update_esp32_screen("LIS")
                                        print("✅ [LISTENING] Turn complete. Waiting for your reply... (Session stays open!)")
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        print(f"❌ [AI RECEIVE ERROR] {e}")

                send_task = asyncio.create_task(send_mic_loop())
                recv_task = asyncio.create_task(receive_ai_loop())

                while True:
                    # Exit if ESP32 disconnected or reconnected
                    if active_websocket is None or (expected_conn_id is not None and esp32_connection_id != expected_conn_id):
                        send_task.cancel()
                        recv_task.cancel()
                        user_ended_call = True
                        break

                    # Handle button presses during active session (volume, sleep toggle, nudge)
                    btn_cmd = check_button()
                    if btn_cmd == "BTN_MAIN":
                        print("🔘 Main button pressed during session - switching to idle/wake mode")
                        send_task.cancel()
                        recv_task.cancel()
                        user_ended_call = True
                        break
                    elif btn_cmd == "PROACTIVE_NUDGE" and proactive_nudge_reason:
                        reason = proactive_nudge_reason
                        proactive_nudge_reason = None
                        await session.send_client_content(
                            turns=types.Content(
                                role="user",
                                parts=[types.Part.from_text(text=f"தமிழில் கண்டிப்புடன் பேசு: கணேஷ் இப்போது நேரத்தை வீணடிக்கிறார் ({reason}). உடனடியாகக் கண்டித்து மீண்டும் வேலைக்குத் திரும்ப ஆணையிடு!")]
                            ),
                            turn_complete=True,
                        )

                    # Check if we should exit (song requested)
                    if current_song_query:
                        drain_queue(ai_audio_queue)
                        await ai_audio_queue.put(b"")
                        ai_is_speaking = False
                        send_task.cancel()
                        recv_task.cancel()
                        user_ended_call = True
                        break

                    # For single_turn mode, wait for greeting audio playback to finish before exiting
                    if single_turn and initial_turn_complete:
                        while (ai_is_speaking or not ai_audio_queue.empty()) and active_websocket is not None and not current_song_query:
                            await asyncio.sleep(0.05)
                        send_task.cancel()
                        recv_task.cancel()
                        user_ended_call = True
                        break

                    if send_task.done() or recv_task.done():
                        if send_task.done() and not send_task.cancelled() and send_task.exception():
                            print(f"❌ Send task failed: {send_task.exception()}")
                        if recv_task.done() and not recv_task.cancelled() and recv_task.exception():
                            print(f"❌ Receive task failed: {recv_task.exception()}")
                        send_task.cancel()
                        recv_task.cancel()

                        # If session died immediately (< 5s), treat as a failed attempt and try next model/config
                        if time.time() - connect_start < 5.0:
                            _resolved_live_target = None
                            attempt += 1
                            candidate_idx += 1
                            print(f"⚠️ Session closed shortly after connect - trying next candidate (attempt {attempt}/{max_attempts})")
                        else:
                            attempt = 0
                        break

                    await asyncio.sleep(0.02)

        except Exception as e:
            err_msg = str(e)
            _resolved_live_target = None
            attempt += 1
            print(f"❌ [GEMINI CONNECTION ERROR] {model_name} @ {api_version}: {err_msg} (attempt {attempt}/{max_attempts})")

            # If SpeechConfig language_code is rejected by this model, retry once without language_code
            if include_language_code and ("language_code" in err_msg.lower() or "speech_config" in err_msg.lower() or "invalid argument" in err_msg.lower()):
                include_language_code = False
                print("⚠️ Retrying Gemini Live connection without SpeechConfig language_code...")
            else:
                candidate_idx += 1

            if attempt < max_attempts:
                await asyncio.sleep(1.5)
            else:
                print("❌ Max connection attempts reached, returning to idle/wake loop")
                break

    drain_queue(mic_queue)
    if not current_song_query:
        update_esp32_screen("IDL")


# --- 7. WebSocket Server & Main Loop ---
async def health_check(connection, request):
    """Respond 200 OK to plain HTTP health checks (e.g. Render) while allowing WebSocket upgrades."""
    if hasattr(connection, "respond") and request.headers.get("Upgrade", "").lower() != "websocket":
        return connection.respond(http.HTTPStatus.OK, "Jarvis AI Assistant is running\n")
    return None


async def ws_sender(websocket):
    try:
        while True:
            await websocket.send(await tx_queue.get())
    except Exception:
        pass


async def ws_handler(websocket):
    global active_websocket, esp32_connection_id
    esp32_connection_id += 1
    conn_id = esp32_connection_id
    active_websocket = websocket

    # Clear stale queues from any previous ESP32 session
    drain_queue(tx_queue)
    drain_queue(mic_queue)
    drain_queue(button_queue)
    drain_queue(ai_audio_queue)

    print(f"\n✅ ESP32 Connected via WebSocket! (connection #{conn_id})")
    sender_task = asyncio.create_task(ws_sender(websocket))
    try:
        async for message in websocket:
            if isinstance(message, str):
                button_queue.put_nowait(message)
            else:
                mic_queue.put_nowait(message)
    except Exception:
        print(f"❌ ESP32 Disconnected! (connection #{conn_id})")
    finally:
        if active_websocket is websocket:
            active_websocket = None
        sender_task.cancel()


async def main():
    global is_music_playing, current_song_query, music_wf, active_websocket
    global auto_play_next, last_song_title, proactive_nudge_reason, last_song_metadata
    global autoplay_enabled, ai_is_speaking

    init_memory()
    port = int(os.environ.get("PORT", 8000))
    print(f"[SYSTEM] Starting Jarvis 24/7 Server on 0.0.0.0:{port}...")
    print(f"[SYSTEM] Music sources: JioSaavn + SoundCloud + Spotify metadata ({'API' if SPOTIFY_CLIENT_ID else 'oEmbed'}) + YouTube fallback")
    print(f"[SYSTEM] Autoplay: {'ENABLED' if autoplay_enabled else 'DISABLED'}")
    print(f"[SYSTEM] Wake word model: {'LOADED' if oww_model_loaded else 'DISABLED (startup/button wake)'}")
    print(f"[SYSTEM] Startup wake: {'ENABLED' if STARTUP_WAKE_ENABLED else 'DISABLED'} (single_turn={STARTUP_SINGLE_TURN})")
    await websockets.serve(ws_handler, "0.0.0.0", port, process_request=health_check)

    asyncio.create_task(stream_music_task())
    asyncio.create_task(ai_speaker_task())

    wake_buffer = bytearray()
    last_handled_conn_id = 0

    while True:
        try:
            # Trigger startup wake every time ESP32 connects or restarts
            if active_websocket is not None and esp32_connection_id != last_handled_conn_id:
                last_handled_conn_id = esp32_connection_id
                is_music_playing = False
                ai_is_speaking = False
                if music_wf:
                    music_wf.close()
                    music_wf = None
                wake_buffer.clear()
                oww_model.reset()
                print(f"✅ ESP32 connection #{last_handled_conn_id} ready - Starting assistant...")

                if STARTUP_WAKE_ENABLED:
                    print("🎯 Startup wake triggered - Auto-greeting...")
                    await live_audio_session(
                        initial_greeting=True,
                        single_turn=STARTUP_SINGLE_TURN,
                        expected_conn_id=last_handled_conn_id,
                    )
                    wake_buffer.clear()
                    oww_model.reset()
                else:
                    update_esp32_screen("IDL")

            # Handle song download/playback
            if current_song_query:
                if music_wf:
                    music_wf.close()
                    music_wf = None
                drain_queue(music_queue)
                query, current_song_query = current_song_query, None
                update_esp32_screen("LOD")

                result = await asyncio.to_thread(download_youtube_music_song, query)
                if result and result[0]:
                    is_music_playing = True
                    last_song_title = result[1]
                    last_song_metadata = result[2] if len(result) > 2 else {}
                    print(f"▶️ Now Playing: {last_song_title}")
                    update_esp32_screen("MUS")
                else:
                    update_esp32_screen("IDL")

            # Handle autoplay
            if auto_play_next:
                auto_play_next = False
                played = await play_next_autoplay_song()
                if not played:
                    print("⚠️ No autoplay song available")
                    update_esp32_screen("IDL")

            # Check button - do this FIRST before wake word
            btn_triggered = check_button()
            wake_triggered = False

            if btn_triggered in ["BTN_FORCE_WAKE", "PROACTIVE_NUDGE"]:
                wake_triggered = True
            elif btn_triggered == "BTN_MAIN":
                if music_wf:
                    is_music_playing = not is_music_playing
                    update_esp32_screen("MUS" if is_music_playing else "PAU")
                else:
                    wake_triggered = True

            # Check wake word when idle (not playing music, not AI speaking, not already triggered)
            if active_websocket is not None and not is_music_playing and not wake_triggered:
                if ai_is_speaking:
                    drain_queue(mic_queue)
                    wake_buffer.clear()
                else:
                    while not mic_queue.empty():
                        wake_buffer.extend(mic_queue.get_nowait())
                    while len(wake_buffer) >= 2560:
                        chunk, wake_buffer = wake_buffer[:2560], wake_buffer[2560:]
                        audio_np = np.frombuffer(chunk, dtype=np.int16)
                        boosted_np = np.clip(audio_np.astype(np.float32) * 2.0, -32768, 32767).astype(np.int16)
                        prediction = oww_model.predict(boosted_np)
                        confidence = extract_wakeword_score(prediction)
                        if confidence >= WAKE_THRESHOLD:
                            print(f"✨ Wake word detected! (confidence: {confidence:.2f})")
                            wake_triggered = True
                            wake_buffer.clear()
                            oww_model.reset()
                            break

            # If wake triggered (button or voice), start new session
            if wake_triggered and active_websocket is not None:
                is_music_playing = False
                if music_wf:
                    music_wf.close()
                    music_wf = None
                reason = proactive_nudge_reason
                proactive_nudge_reason = None
                print("🎤 Starting new listening session...")
                await live_audio_session(
                    initial_greeting=False,
                    nudge_reason=reason,
                    expected_conn_id=last_handled_conn_id,
                )
                wake_buffer.clear()
                oww_model.reset()

            await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
