import socket
import wave
import array

UDP_IP = "0.0.0.0"
UDP_PORT = 12345
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit = 2 bytes per sample

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("ESP32-ல் இருந்து ஆடியோவை வாங்க காத்திருக்கிறேன்...")
print("மைக்கில் சத்தமாகப் பேசுங்கள்... (சுமார் 8 வினாடிகள் ரெக்கார்ட் ஆகும்)\n")

audio_buffer = bytearray()

try:
    # 500 பாக்கெட்டுகள் = சுமார் 8 வினாடிகள் ரெக்கார்டிங்
    for _ in range(500):  
        data, addr = sock.recvfrom(1024)
        
        # ESP32 ஏற்கனவே 16-bit Mono ஆக அனுப்புவதால் நேரடியாக படிக்கிறோம்
        samples = array.array('h', data)
        max_volume = max(abs(s) for s in samples) if len(samples) > 0 else 0
        
        # சத்தத்தின் அளவை பார் (Bar) ஆக காட்டுகிறது
        bars = "=" * (max_volume // 200)
        print(f"Volume: {max_volume:05d} | {bars:<40}", end='\r')
        
        audio_buffer.extend(data)
except KeyboardInterrupt:
    pass

print("\n\nஆடியோ சேகரிப்பு முடிந்தது! 'ai_heard.wav' ஃபைலாக சேவ் செய்கிறோம்...")

with wave.open("ai_heard.wav", "wb") as wf:
    wf.setnchannels(1)           # Mono
    wf.setsampwidth(SAMPLE_WIDTH) # 16-bit
    wf.setframerate(SAMPLE_RATE)  # 16000 Hz
    wf.writeframes(audio_buffer)

print("ஃபைல் சேவ் ஆகிவிட்டது! இப்போது உங்கள் ஃபோல்டரில் உள்ள 'ai_heard.wav' ஃபைலை ஓபன் செய்து கேட்கவும்.")