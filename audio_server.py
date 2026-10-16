import socket
import wave
import struct

# --- Server Configuration ---
UDP_IP = "0.0.0.0"       # எல்லா நெட்வொர்க் இன்டர்ஃபேஸ்களையும் கவனிக்கும்
UDP_PORT = 12345         # ஆடியோ டேட்டாவைப் பெறும் போர்ட்
OUTPUT_FILENAME = "test_recording.wav" # டேட்டா இந்த ஃபைலில் சேவ் ஆகும்

# ஆடியோ ஃபார்மெட் (16kHz, 16-bit, Mono)
SAMPLE_RATE = 16000
CHANNELS = 2
SAMPLE_WIDTH = 2 # 16-bit = 2 bytes

print(f"Starting Audio Server on Port {UDP_PORT}...")

# UDP Socket-ஐ உருவாக்குகிறோம்
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print("Server is listening for audio data. Press Ctrl+C to stop.")

try:
    # WAV ஃபைலை எழுதத் தயார் செய்கிறோம்
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        
        while True:
            # ESP32-லிருந்து 512 bytes அளவுள்ள டேட்டாவை வாங்குகிறோம்
            data, addr = sock.recvfrom(512)
            if data:
                wf.writeframes(data)
                print(f"Received {len(data)} bytes from {addr[0]}", end='\r')

except KeyboardInterrupt:
    print(f"\nRecording stopped. Saved as {OUTPUT_FILENAME}")
finally:
    sock.close()