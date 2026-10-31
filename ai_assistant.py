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

# --- 1. Gemini Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before running the assistant.")
client = genai.Client(api_key=API_KEY)
LIVE_MODEL = "gemini-3.8-live" # அதிவேக ரெஸ்பான்ஸிற்காக 2.0 Flash பயன்படுத்தப்பட்டுள்ளது

print("[SYSTEM] Loading offline 'Hey Jarvis' model...")
openwakeword.utils.download_models(["hey_jarvis"])
oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

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
proactive_nudge_reason = None

tx_queue = asyncio.Queue()      
mic_queue = asyncio.Queue()     
button_queue = asyncio.Queue()  
ai_audio_queue = asyncio.Queue()

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

def download_youtube_song(query):
    print(f"\n[MUSIC] Searching Full Song: {query}")
    for old in glob.glob("current_song.*"):
        try: os.remove(old)
        except: pass

    # --- முறை 1: JioSaavn API (320kbps முழு நீள ஒரிஜினல் தமிழ் பாடல்கள் - No Bot Block!) ---
    saavn_apis = [
        f"https://saavn.dev/api/search/songs?query={query}",
        f"https://jiosaavn-api-privatecvc2.vercel.app/search/songs?query={query}"
    ]
    
    for api_url in saavn_apis:
        try:
            resp = requests.get(api_url, timeout=8).json()
            results = resp.get("data", {}).get("results") or resp.get("results")
            if results and len(results) > 0:
                song = results[0]
                title = song.get("name") or song.get("title") or query
                
                # டவுன்லோட் லிங்க்கை எடுத்தல்
                dl_links = song.get("downloadUrl") or song.get("more_info", {}).get("encrypted_media_url")
                if isinstance(dl_links, list) and len(dl_links) > 0:
                    stream_url = dl_links[-1].get("url") or dl_links[-1].get("link")
                    
                    print(f"⬇️ Downloading Full Song: {title}")
                    # FFmpeg மூலம் நேரடியாக 16kHz Mono WAV ஆக மாற்றுதல் (2 விநாடிகளில் முடிந்துவிடும்)
                    cmd = [
                        "ffmpeg", "-y", "-i", stream_url,
                        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                        "current_song.wav"
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                    
                    if os.path.exists("current_song.wav") and os.path.getsize("current_song.wav") > 10000:
                        print(f"✅ Ready to play: {title}")
                        return "current_song.wav", title
        except Exception as e:
            print(f"⚠️ Saavn API trying next... ({e})")

    # --- முறை 2: YouTube via Piped Proxy (யூடியூப்பில் மட்டுமே உள்ள ஆடியோக்களுக்கு) ---
    print("🔄 Trying YouTube Proxy API...")
    piped_apis = [
        "https://pipedapi.kavin.rocks",
        "https://pipedapi.tokhmi.xyz",
        "https://api.piped.projectsegfau.lt"
    ]
    for base_api in piped_apis:
        try:
            s_resp = requests.get(f"{base_api}/search?q={query}+song+audio&filter=music_songs", timeout=8).json()
            items = s_resp.get("items", [])
            if items:
                vid_url = items[0].get("url", "") # e.g., /watch?v=szvt1vD0Uug
                vid_id = vid_url.split("v=")[-1]
                title = items[0].get("title", query)
                
                st_resp = requests.get(f"{base_api}/streams/{vid_id}", timeout=8).json()
                audio_streams = st_resp.get("audioStreams", [])
                if audio_streams:
                    stream_url = audio_streams[0].get("url")
                    cmd = [
                        "ffmpeg", "-y", "-i", stream_url,
                        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
                        "current_song.wav"
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40)
                    if os.path.exists("current_song.wav") and os.path.getsize("current_song.wav") > 10000:
                        print(f"✅ Ready from YouTube Proxy: {title}")
                        return "current_song.wav", title
        except Exception as e:
            continue

    print("❌ Could not download the song from any source.")
    return None, None

# --- 5. Audio Streaming Tasks ---
async def stream_music_task():
    global is_music_playing, music_wf, global_volume, auto_play_next
    audio_start_time = 0
    audio_frames_sent = 0

    while True:
        if is_music_playing:
            if not music_wf and os.path.exists("current_song.wav"):
                try:
                    music_wf = wave.open("current_song.wav", "rb")
                    audio_start_time, audio_frames_sent = time.time(), 0
                except:
                    music_wf, is_music_playing = None, False

            if music_wf:
                chunk = music_wf.readframes(512)
                if not chunk:
                    music_wf.close()
                    music_wf, is_music_playing = None, False
                    update_esp32_screen("IDL")
                    auto_play_next = True
                    # பாட்டு முடிந்ததும் மெமரியை கிளியர் செய்ய ஃபைலை நீக்குதல்
                    try: os.remove("current_song.wav")
                    except: pass
                else:
                    np_chunk = np.clip(np.frombuffer(chunk, dtype=np.int16).astype(np.float32) * global_volume, -32768, 32767).astype(np.int16)
                    if active_websocket: tx_queue.put_nowait(np_chunk.tobytes())
                    audio_frames_sent += 512
                    sleep_delay = (audio_frames_sent / 16000.0) - (time.time() - audio_start_time)
                    await asyncio.sleep(max(sleep_delay, 0.001))
                    continue
        else:
            audio_start_time = time.time() - (audio_frames_sent / 16000.0)
        await asyncio.sleep(0.02)

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

கணேஷைப் பற்றிய உனது நினைவுகள் மற்றும் முந்தைய உரையாடல்கள்:
{memories}
"""

async def live_audio_session(initial_greeting=False, nudge_reason=None):
    global is_music_playing, current_song_query, music_wf, ai_audio_queue, global_volume, ai_is_speaking

    jarvis_tools = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="play_music",
            description="Play a song or music track from YouTube",
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
    
    while not user_ended_call and not current_song_query:
        try:
            print("\n🔄 Connecting to Gemini Live API...")
            async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                update_esp32_screen("LIS")
                print("✅ [LISTENING] Jarvis is online! (Continuous Session)")
                
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
                                            function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"result": "Playing song now"}))
                                        
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
                        send_task.cancel(); recv_task.cancel()
                        break
                        
                    await asyncio.sleep(0.02)
                    
        except Exception as e: 
            print(f"❌ [GEMINI CONNECTION ERROR] {e}")
            await asyncio.sleep(2)
            
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
    global auto_play_next, last_song_title, proactive_nudge_reason
    
    init_memory()
    port = int(os.environ.get("PORT", 8000))
    print(f"[SYSTEM] Starting Jarvis 24/7 Server on 0.0.0.0:{port}...")
    await websockets.serve(ws_handler, "0.0.0.0", port)
    
    asyncio.create_task(stream_music_task())
    asyncio.create_task(ai_speaker_task())

    while active_websocket is None: await asyncio.sleep(0.1)
    
    await live_audio_session(initial_greeting=True)

    wake_buffer = bytearray()
    while True:
        try:
            if current_song_query:
                if music_wf: music_wf.close(); music_wf = None
                query, current_song_query = current_song_query, None
                update_esp32_screen("LOD")
                
                result = await asyncio.to_thread(download_youtube_song, query)
                if result and result[0]:
                    is_music_playing = True
                    last_song_title = result[1]
                    print(f"▶️ Now Playing: {last_song_title}")
                    update_esp32_screen("MUS")
                else: 
                    update_esp32_screen("IDL")

            if auto_play_next:
                auto_play_next = False
                try:
                    prompt = f"The user just listened to the Tamil song '{last_song_title}'. Suggest ONE similar Tamil song. Reply ONLY with 'Song name - Movie name'."
                    resp = await client.aio.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    current_song_query = resp.text.strip()
                    continue
                except Exception as e:
                    print(f"❌ Autoplay failed: {e}")
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

            if not is_music_playing and not wake_triggered:
                while not mic_queue.empty(): wake_buffer.extend(mic_queue.get_nowait())
                while len(wake_buffer) >= 2560:
                    chunk, wake_buffer = wake_buffer[:2560], wake_buffer[2560:]
                    if oww_model.predict(np.frombuffer(chunk, dtype=np.int16)).get("hey_jarvis", 0.0) > 0.5:
                        print("✨ Wake word detected!")
                        wake_triggered = True
                        wake_buffer.clear()
                        break

            if wake_triggered:
                is_music_playing = False
                if music_wf: music_wf.close(); music_wf = None
                reason = proactive_nudge_reason
                proactive_nudge_reason = None
                await live_audio_session(initial_greeting=False, nudge_reason=reason)
                
            await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(main())
