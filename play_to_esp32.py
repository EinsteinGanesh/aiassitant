import asyncio
import websockets
import wave
import os
from pydub import AudioSegment

# நீங்கள் பிளே செய்ய விரும்பும் ஆடியோ ஃபைலின் பெயர் (WAV அல்லது MP3)
INPUT_AUDIO_FILE = "test_audio.wav"  
PROCESSED_AUDIO_FILE = "ready_for_esp32.wav"

def prepare_audio(filename):
    print(f"🎧 Preparing '{filename}' for ESP32...")
    # ஆடியோவை 16kHz, 16-bit, Mono ஃபார்மேட்டிற்கு மாற்றுதல்
    audio = AudioSegment.from_file(filename)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(PROCESSED_AUDIO_FILE, format="wav")
    print("✅ Audio conversion successful!")

async def ws_handler(websocket, path="/"):
    print("\n✅ ESP32 Connected! Streaming audio to speaker...")
    
    try:
        with wave.open(PROCESSED_AUDIO_FILE, 'rb') as wf:
            while True:
                # 512 frames (1024 bytes) ஆகப் பிரித்து அனுப்புதல்
                chunk = wf.readframes(512)
                
                if not chunk:
                    print("🏁 Audio playback finished!")
                    break
                
                # பைனரி டேட்டாவை ESP32-க்கு அனுப்புதல்
                await websocket.send(chunk)
                
                # ஆடியோவின் வேகத்தை (Playback Speed) சீராக வைக்க சிறிய Delay
                # 512 frames @ 16000Hz = 0.032 seconds
                await asyncio.sleep(0.032)
                
    except websockets.exceptions.ConnectionClosed:
        print("❌ ESP32 Disconnected!")

async def main():
    if not os.path.exists(INPUT_AUDIO_FILE):
        print(f"❌ Error: Please put an audio file named '{INPUT_AUDIO_FILE}' in this folder.")
        return

    prepare_audio(INPUT_AUDIO_FILE)
    
    print("\n🚀 Starting Audio Streaming Server on port 8000...")
    async with websockets.serve(ws_handler, "0.0.0.0", 8000):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer Stopped.")