import socket
import wave
import time
import sys
import array

UDP_IP = "0.0.0.0"
MIC_PORT = 12345
SPK_PORT = 12346
SAMPLE_RATE = 16000
RECORD_SECONDS = 5

sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_rx.bind((UDP_IP, MIC_PORT))
sock_rx.settimeout(5.0)  # Wait max 5 seconds for ESP32 to send data
sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("=" * 50)
print(f"PHASE 1: RECORDING ({RECORD_SECONDS} seconds)")
print("Make noise near the ESP32 mic now...")
print("=" * 50)

audio_data = bytearray()
esp32_ip = None
target_bytes = SAMPLE_RATE * 2 * RECORD_SECONDS 

try:
    while len(audio_data) < target_bytes:
        data, addr = sock_rx.recvfrom(2048)
        audio_data.extend(data)
        esp32_ip = addr[0]
        
        # Show live network progress and audio volume
        samples = array.array('h', data)
        peak = max(abs(s) for s in samples) if samples else 0
        percent = int((len(audio_data) / target_bytes) * 100)
        
        # Draw the volume bar
        sys.stdout.write(f"\rNetwork: {percent:03d}% | Vol: {peak:05d} | {'=' * (peak // 300)}>")
        sys.stdout.flush()
        
except socket.timeout:
    print("\n\n[ERROR] Network Failure: PC did not receive any audio packets from the ESP32!")
    print("Fix 1: Open command prompt, type 'ipconfig', and verify your IPv4 address is still 192.168.1.4.")
    print("Fix 2: Open Arduino Serial Monitor and verify the ESP32 actually connected to Wi-Fi.")
    sys.exit(1)

print("\n\nPHASE 2: SAVING")
filename = "system_recording.wav"
with wave.open(filename, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_data)
print(f"File saved successfully.")

print("\nPHASE 3: PLAYBACK")
print(f"Streaming back to ESP32 speakers at {esp32_ip}...")

chunk_size = 1024 
for i in range(0, len(audio_data), chunk_size):
    chunk = audio_data[i:i+chunk_size]
    sock_tx.sendto(chunk, (esp32_ip, SPK_PORT))
    time.sleep(0.031) # Precise delay to prevent audio stutter

print("\nPlayback complete! Check the speakers.")