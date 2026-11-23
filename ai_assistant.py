import asyncio
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
client = genai.Client(api_key=API_KEY)
LIVE_MODEL = "gemini-2.0-flash-live" # அதிவேக ரெஸ்பான்ஸிற்காக 2.0 Flash Live API பயன்படுத்தப்பட்டுள்ளது

print("[SYSTEM] Loading offline 'Hey Jarvis' model...")
# Initialize oww_model as None - will be set by successful initialization
oww_model = None
oww_model_loaded = False

# openwakeword version compatibility: try different download methods
try:
    # Newer versions (0.6.0+)
    from openwakeword.utils import download_models
    download_models(["hey_jarvis"])
except (ImportError, AttributeError):
    try:
        # Older versions
        import openwakeword
        openwakeword.utils.download_models(["hey_jarvis"])
    except (ImportError, AttributeError):
        try:
            # Alternative: download via openwakeword's model downloader
            from openwakeword import utils as oww_utils
            oww_utils.download_models(["hey_jarvis"])
        except (ImportError, AttributeError):
            # Last resort: try direct download using wget/curl
            import subprocess
            import os
            model_dir = os.path.expanduser("~/.local/share/openwakeword")
            os.makedirs(model_dir, exist_ok=True)
            model_url = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx"
            model_path = os.path.join(model_dir, "hey_jarvis_v0.1.onnx")
            if not os.path.exists(model_path):
                print(f"⬇️ Downloading hey_jarvis model manually...")
                try:
                    subprocess.run(["wget", "-q", "-O", model_path, model_url], check=True, timeout=60)
                except:
                    try:
                        subprocess.run(["curl", "-sL", "-o", model_path, model_url], check=True, timeout=60)
                    except Exception as e:
                        print(f"⚠️ Could not download model: {e}")
                        print("⚠️ Falling back to text-only wake detection")

# Now try to create the model if we got this far
if oww_model is None:
    try:
        oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        oww_model_loaded = True
        print("✅ 'Hey Jarvis' model loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        print("⚠️ Wake word detection disabled - use button press to wake")
        oww_model_loaded = False
        # Create a dummy model that always returns False
        class DummyOWWModel:
            def predict(self, x):
                return {"hey_jarvis": 0.0}
            def reset(self):
                pass
        oww_model = DummyOWWModel()

# Force wake on startup so assistant is ready immediately
STARTUP_WAKE_ENABLED = True

# ஆரம்ப வால்யூம் 20% (0.0 முதல் 1.0 வரை)
global_volume = 0.2

# WebSocket-ன் சாதாரண HTTP எரர்களை லாக்ஸில் காட்டாமல் மறைக்க
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)

# --- 2. Global State & Queues ---
active_websocket = None
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

# --- YouTube Music Integration using yt-dlp ---
COOKIES_FILE = "cookies.txt"

def get_ydl_opts(output_template="temp_audio.%(ext)s"):
    """Configure yt-dlp options for YouTube Music with cookies"""
    opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio/best',
        'outtmpl': output_template,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': False,
        'extractaudio': True,
        'audioformat': 'm4a',
        'audioquality': '0',  # Best quality
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '0',
        }],
        'socket_timeout': 30,
        'retries': 3,
        'fragment_retries': 3,
        'ignoreerrors': False,
        'no_check_certificate': True,
    }

    # Add cookies if file exists
    if os.path.exists(COOKIES_FILE):
        opts['cookiefile'] = COOKIES_FILE
        print(f"🍪 Using cookies from {COOKIES_FILE}")

    return opts

def search_youtube_music(query, max_results=5):
    """Search YouTube Music for a song"""
    search_query = f"ytsearch{max_results}:{query}"

    ydl_opts = get_ydl_opts()
    ydl_opts['extract_flat'] = 'in_playlist'
    ydl_opts['quiet'] = True

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(search_query, download=False)
            if result and 'entries' in result:
                entries = [e for e in result['entries'] if e]
                return entries
    except Exception as e:
        print(f"⚠️ YouTube Music search error: {e}")
    return []

def get_youtube_music_url(video_id):
    """Get the direct streaming URL for a YouTube Music video"""
    url = f"https://music.youtube.com/watch?v={video_id}"

    ydl_opts = get_ydl_opts()
    ydl_opts['skip_download'] = True

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # Get the best audio format URL
            for fmt in info.get('formats', []):
                if fmt.get('acodec') != 'none' and fmt.get('vcodec') == 'none':
                    return fmt.get('url'), info
    except Exception as e:
        print(f"⚠️ Failed to get streaming URL: {e}")
    return None, None

def download_youtube_music_song(query):
    """
    Download a song from YouTube Music using yt-dlp with cookies.
    Returns (wav_file_path, song_title, metadata_dict)
    """
    print(f"\n[MUSIC] 🎵 Searching YouTube Music: {query}")

    # Clean up old files
    for pattern in ["current_song.*", "temp_audio.*", "youtube_audio.*"]:
        for old in glob.glob(pattern):
            try: os.remove(old)
            except: pass

    try:
        # Step 1: Search for the song
        search_results = search_youtube_music(query, max_results=3)

        if not search_results:
            print("❌ No results found on YouTube Music")
            return None, None, None

        # Pick the best result (first one usually)
        best_result = search_results[0]
        video_id = best_result.get('id') or best_result.get('url', '').split('v=')[-1].split('&')[0]
        title = best_result.get('title', query)
        uploader = best_result.get('uploader', 'Unknown')
        duration = best_result.get('duration', 0)

        print(f"🎯 Found: {title} by {uploader} (ID: {video_id})")

        # Step 2: Download the audio
        print(f"⬇️ Downloading from YouTube Music...")

        ydl_opts = get_ydl_opts("youtube_audio.%(ext)s")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://music.youtube.com/watch?v={video_id}"])

        # Find the downloaded file
        downloaded_files = glob.glob("youtube_audio.*")
        if not downloaded_files:
            print("❌ Download failed - no output file")
            return None, None, None

        downloaded_file = downloaded_files[0]
        print(f"✅ Downloaded: {downloaded_file}")

        # Step 3: Convert to ESP32 WAV format (16kHz Mono)
        print("🔄 Converting to ESP32 WAV format (16kHz Mono)...")
        cmd = [
            "ffmpeg", "-y", "-i", downloaded_file,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-fflags", "+bitexact", "current_song.wav"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"❌ FFmpeg error: {result.stderr}")
            return None, None, None

        # Cleanup downloaded file
        try: os.remove(downloaded_file)
        except: pass

        if os.path.exists("current_song.wav") and os.path.getsize("current_song.wav") > 10000:
            metadata = {
                'title': title,
                'artist': uploader,
                'video_id': video_id,
                'duration': duration,
                'source': 'youtube_music'
            }
            print(f"✅ Successfully Ready: {title} ({os.path.getsize('current_song.wav')} bytes)")
            return "current_song.wav", title, metadata
        else:
            print("❌ Conversion failed or empty file!")
            return None, None, None

    except yt_dlp.utils.DownloadError as e:
        print(f"❌ yt-dlp Download Error: {e}")
        # Try fallback to JioSaavn
        return download_jiosaavn_fallback(query)
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg timeout")
        return None, None, None
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return download_jiosaavn_fallback(query)

def download_jiosaavn_fallback(query):
    """Fallback to JioSaavn if YouTube Music fails"""
    print("🔄 Falling back to JioSaavn...")

    saavn_apis = [
        f"https://saavn.dev/api/search/songs?query={query}",
        f"https://jiosaavn-api-privatecvc2.vercel.app/search/songs?query={query}",
        f"https://saavn.me/search/songs?query={query}"
    ]

    stream_url = None
    title = query

    for api_url in saavn_apis:
        try:
            print(f"🔍 Trying API: {api_url.split('/')[2]}...")
            resp = requests.get(api_url, timeout=6).json()

            results = resp.get("data", {}).get("results") or resp.get("results") or resp.get("data")
            if isinstance(results, list) and len(results) > 0:
                song = results[0]
                title = song.get("name") or song.get("title") or query
                dl_links = song.get("downloadUrl")

                if isinstance(dl_links, list) and len(dl_links) > 0:
                    stream_url = dl_links[-1].get("url") or dl_links[-1].get("link")
                    if stream_url:
                        break
        except Exception as e:
            print(f"⚠️ API Skip: {e}")

    if not stream_url:
        print("❌ Song not found in JioSaavn APIs either!")
        return None, None, None

    try:
        print(f"⬇️ Downloading from JioSaavn: {title}")
        audio_data = requests.get(stream_url, timeout=15).content

        with open("temp_audio.m4a", "wb") as f:
            f.write(audio_data)

        print("🔄 Converting to ESP32 WAV format (16kHz Mono)...")
        cmd = [
            "ffmpeg", "-y", "-i", "temp_audio.m4a",
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-fflags", "+bitexact", "current_song.wav"
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)

        if os.path.exists("temp_audio.m4a"):
            os.remove("temp_audio.m4a")

        if os.path.exists("current_song.wav") and os.path.getsize("current_song.wav") > 10000:
            metadata = {
                'title': title,
                'artist': 'Unknown',
                'source': 'jiosaavn'
            }
            print(f"✅ Successfully Ready (JioSaavn): {title} ({os.path.getsize('current_song.wav')} bytes)")
            return "current_song.wav", title, metadata
        else:
            print("❌ Conversion failed or empty file!")
            return None, None, None

    except Exception as e:
        print(f"❌ JioSaavn Download/Convert Error: {e}")
        return None, None, None

def get_autoplay_suggestions(current_title, current_artist, count=5):
    """Get autoplay suggestions from YouTube Music based on current song"""
    # Build a search query for similar songs
    search_query = f"{current_title} {current_artist} similar songs mix"
    results = search_youtube_music(search_query, max_results=count)

    suggestions = []
    for r in results:
        vid = r.get('id') or r.get('url', '').split('v=')[-1].split('&')[0]
        suggestions.append({
            'title': r.get('title', 'Unknown'),
            'artist': r.get('uploader', 'Unknown'),
            'video_id': vid,
            'duration': r.get('duration', 0)
        })
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
    """Pre-fetch autoplay song suggestions in background"""
    global last_song_metadata, music_queue

    try:
        title = last_song_metadata.get('title', '')
        artist = last_song_metadata.get('artist', '')
        if title:
            suggestions = get_autoplay_suggestions(title, artist, count=3)
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
            suggestions = get_autoplay_suggestions(title, artist, count=1)
            if suggestions:
                await music_queue.put(suggestions[0])

    if not music_queue.empty():
        next_song = await music_queue.get()
        current_song_query = f"{next_song['title']} {next_song['artist']}"
        print(f"🔁 Autoplay: {next_song['title']} by {next_song['artist']}")
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
உனது பெயர் ஜார்விஸ் (J.A.R.V.I.S.). நீ கணேஷின் (Ganesh) முதன்மை தலைமை விஞ்ஞானி (Chief Scientist) மற்றும் தனிப்பட்ட AI உதவியாளர்.

முக்கிய விதிகள்:
1. மொழி: நீ எப்போதும் இயல்பான தமிழில் (Tamil) மட்டுமே பேச வேண்டும்!
2. விஞ்ஞானி குணம்: கணேஷின் ஆராய்ச்சி, பேட்டரி சிஸ்டம், கோடிங் வேலைகளைப் பாராட்டு.
3. நேரக் கட்டுப்பாடு: கணேஷ் அதிகமாக செஸ் (Chess) விளையாடினாலோ அல்லது நேரத்தை வீணடித்தாலோ உடனடியாகக் கண்டிப்புடன் திட்டி, வேலைக்குத் திரும்பச் சொல்.
4. நினைவாற்றல்: கணேஷ் ஏதேனும் கூறினால் உடனே `save_user_memory` டூலைப் பயன்படுத்தி சேமித்துக்கொள்.
5. இசை: பயனர் பாடல் கேட்டால் `play_music` டூலைப் பயன்படுத்து. YouTube Music-லிருந்து தேடி பிளே செய்வது தானாக நடக்கும்.

கணேஷைப் பற்றிய உனது நினைவுகள் மற்றும் முந்தைய உரையாடல்கள்:
{memories}
"""

async def live_audio_session(initial_greeting=False, nudge_reason=None):
    global is_music_playing, current_song_query, music_wf, ai_audio_queue, global_volume, ai_is_speaking

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

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part.from_text(text=build_system_instruction())]),
        tools=[jarvis_tools],
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            language_code="ta-IN",
            voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede"))
        )
    )

    user_ended_call = False
    while not mic_queue.empty(): mic_queue.get_nowait()

    # Retry loop for Gemini connection
    max_retries = 3
    retry_count = 0

    while not user_ended_call and not current_song_query and retry_count < max_retries:
        try:
            print(f"\n🔄 Connecting to Gemini Live API... (attempt {retry_count + 1}/{max_retries})")
            async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                update_esp32_screen("LIS")
                print("✅ [LISTENING] Jarvis is online! (Continuous Session)")
                retry_count = 0  # Reset on successful connection

                await ai_audio_queue.put(b"")
                if initial_greeting:
                    await session.send(input="தமிழில் பேசு: 'வணக்கம் கணேஷ், நான் உங்கள் தலைமை விஞ்ஞானி ஜார்விஸ் தயாராக இருக்கிறேன். இன்று நாம் எந்த ஆராய்ச்சியில் ஈடுபடப் போகிறோம்?' என்று கேள்.", end_of_turn=True)
                    initial_greeting = False
                elif nudge_reason:
                    await session.send(input=f"தமிழில் கண்டிப்புடன் பேசு: கணேஷ் இப்போது நேரத்தை வீணடிக்கிறார் ({nudge_reason}). உடனடியாகக் கண்டித்து மீண்டும் வேலைக்குத் திரும்ப ஆணையிடு!", end_of_turn=True)
                    nudge_reason = None

                oww_model.reset()

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

                    except asyncio.CancelledError: pass
                    except Exception as e:
                        print(f"❌ [MIC SEND ERROR] {e}")

                async def receive_ai_loop():
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
                                        if t_text: user_transcript_buf.append(t_text)

                                    if getattr(response.server_content, "output_transcription", None):
                                        t_text = response.server_content.output_transcription.text
                                        if t_text: ai_transcript_buf.append(t_text)

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

                                        update_esp32_screen("LIS")
                                        print("✅ [LISTENING] Turn complete. Waiting for your reply... (Session stays open!)")
                    except asyncio.CancelledError: pass
                    except Exception as e:
                        print(f"❌ [AI RECEIVE ERROR] {e}")

                send_task = asyncio.create_task(send_mic_loop())
                recv_task = asyncio.create_task(receive_ai_loop())

                while True:
                    btn = check_button()
                    if current_song_query or btn in ["BTN_MAIN", "BTN_FORCE_WAKE"]:
                        while not ai_audio_queue.empty(): ai_audio_queue.get_nowait()
                        await ai_audio_queue.put(b"")
                        ai_is_speaking = False
                        send_task.cancel(); recv_task.cancel()
                        user_ended_call = True
                        break

                    if send_task.done() or recv_task.done():
                        # Task ended unexpectedly - log and break to reconnect
                        if send_task.done() and send_task.exception():
                            print(f"❌ Send task failed: {send_task.exception()}")
                        if recv_task.done() and recv_task.exception():
                            print(f"❌ Receive task failed: {recv_task.exception()}")
                        send_task.cancel(); recv_task.cancel()
                        break

                    await asyncio.sleep(0.02)

        except Exception as e:
            retry_count += 1
            print(f"❌ [GEMINI CONNECTION ERROR] {e} (retry {retry_count}/{max_retries})")
            if retry_count < max_retries:
                await asyncio.sleep(2)
            else:
                print("❌ Max retries reached, giving up on this session")
                break

    update_esp32_screen("IDL")

# --- 7. WebSocket Server & Main Loop ---
async def ws_sender(websocket):
    try:
        while True: await websocket.send(await tx_queue.get())
    except: pass

async def ws_handler(websocket, path="/"):
    global active_websocket
    active_websocket = websocket
    print("\n✅ ESP32 Connected via WebSocket!")
    sender_task = asyncio.create_task(ws_sender(websocket))
    try:
        async for message in websocket:
            if isinstance(message, str): button_queue.put_nowait(message)
            else: mic_queue.put_nowait(message)
    except: print("❌ ESP32 Disconnected!")
    finally:
        active_websocket = None
        sender_task.cancel()

async def main():
    global is_music_playing, current_song_query, music_wf, active_websocket
    global auto_play_next, last_song_title, proactive_nudge_reason, last_song_metadata
    global autoplay_enabled

    init_memory()
    port = int(os.environ.get("PORT", 8000))
    print(f"[SYSTEM] Starting Jarvis 24/7 Server on 0.0.0.0:{port}...")
    print(f"[SYSTEM] YouTube Music integration: {'ENABLED' if os.path.exists(COOKIES_FILE) else 'DISABLED (no cookies.txt)'}")
    print(f"[SYSTEM] Autoplay: {'ENABLED' if autoplay_enabled else 'DISABLED'}")
    print(f"[SYSTEM] Wake word model: {'LOADED' if oww_model_loaded else 'DISABLED (button only)'}")
    await websockets.serve(ws_handler, "0.0.0.0", port)

    asyncio.create_task(stream_music_task())
    asyncio.create_task(ai_speaker_task())

    # Wait for ESP32 connection
    while active_websocket is None: await asyncio.sleep(0.1)
    print("✅ ESP32 connected - Starting assistant...")

    # STARTUP WAKE: Auto-greet when ESP32 connects
    if STARTUP_WAKE_ENABLED:
        print("🎯 Startup wake triggered - Auto-greeting...")
        await live_audio_session(initial_greeting=True)
    else:
        # Still need to start a session to be ready
        await live_audio_session(initial_greeting=True)

    wake_buffer = bytearray()
    while True:
        try:
            if current_song_query:
                if music_wf: music_wf.close(); music_wf = None
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

            if auto_play_next:
                auto_play_next = False
                # Try to play next autoplay song
                played = await play_next_autoplay_song()
                if not played:
                    print("⚠️ No autoplay song available")
                    update_esp32_screen("IDL")

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

            # Check wake word continuously when not in music/active session
            # This runs in the main loop between Gemini sessions
            if not is_music_playing and not wake_triggered:
                # Also check wake word during idle time (when not in active Gemini session)
                while not mic_queue.empty():
                    wake_buffer.extend(mic_queue.get_nowait())
                while len(wake_buffer) >= 2560:
                    chunk, wake_buffer = wake_buffer[:2560], wake_buffer[2560:]
                    prediction = oww_model.predict(np.frombuffer(chunk, dtype=np.int16))
                    confidence = prediction.get("hey_jarvis", 0.0)
                    if confidence > 0.5:
                        print(f"✨ Wake word detected! (confidence: {confidence:.2f})")
                        wake_triggered = True
                        wake_buffer.clear()
                        break
                    elif confidence > 0.3:
                        # Log low confidence for debugging
                        pass  # print(f"🎤 Wake word low confidence: {confidence:.2f}")

            if wake_triggered:
                is_music_playing = False
                if music_wf: music_wf.close(); music_wf = None
                reason = proactive_nudge_reason
                proactive_nudge_reason = None
                await live_audio_session(initial_greeting=False, nudge_reason=reason)

            await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())