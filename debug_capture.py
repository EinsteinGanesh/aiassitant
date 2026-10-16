import socket
import wave

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 12345))

print("5 வினாடிகள் பேசுங்கள்...")
audio_buffer = bytearray()
for _ in range(300):
    data, addr = sock.recvfrom(1024)
    audio_buffer.extend(data)

# Stereo to Mono
mono_buffer = bytearray()
for i in range(0, len(audio_buffer), 4):
    if i + 2 <= len(audio_buffer):
        mono_buffer.extend(audio_buffer[i:i+2])

# WAV ஆக சேவ் செய்கிறோம்
with wave.open("check_stt.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(mono_buffer)

print("check_stt.wav சேவ் செய்யப்பட்டது! இதைக் கேட்டுப் பாருங்கள்.")
sock.close()