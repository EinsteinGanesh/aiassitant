import socket
import wave
import array

UDP_IP = "0.0.0.0"
UDP_PORT = 12345
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit audio
OUTPUT_FILE = "my_recording.wav"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

audio_data = bytearray()

print("=" * 50)
print(f"Listening on port {UDP_PORT}... Speak into the ESP32 mic!")
print("Press 'Ctrl + C' to stop recording and save the file.")
print("=" * 50)

try:
    while True:
        data, addr = sock.recvfrom(2048)
        audio_data.extend(data)
        
        # Live Volume Meter
        samples = array.array('h', data)
        peak = max(abs(s) for s in samples) if samples else 0
        
        bars = "=" * min(40, peak // 150)
        print(f"Recording | Vol: {peak:05d} | {bars}>", end='\r')

except KeyboardInterrupt:
    print("\n\nRecording stopped by user.")

# Save to WAV file
print(f"Saving {len(audio_data)} bytes to {OUTPUT_FILE}...")
with wave.open(OUTPUT_FILE, "wb") as wf:
    wf.setnchannels(1)           # Mono
    wf.setsampwidth(SAMPLE_WIDTH)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_data)

print("Done! You can now play 'my_recording.wav'.")