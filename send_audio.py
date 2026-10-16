import socket
import wave
import time
import array 

# --- செட்டிங்ஸ் ---
ESP32_IP = "192.168.1.19" 
UDP_PORT = 12346
WAV_FILE = r"G:\ai assitant\file.wav"

# வால்யூமை எவ்வளவு மடங்கு அதிகரிக்க வேண்டும்? 
VOLUME_MULTIPLIER = 4.0  

print(f"Sending {WAV_FILE} to ESP32 ({ESP32_IP}) with Volume x{VOLUME_MULTIPLIER}...")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

try:
    with wave.open(WAV_FILE, 'rb') as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        
        chunk_size = 512
        start_time = time.time()
        total_samples = 0
        
        while True:
            data = wf.readframes(chunk_size)
            if not data:
                break
                
            # 1. Stereo-லிருந்து Mono-வாக மாற்றுதல்
            if channels == 2 and sampwidth == 2:
                mono_data = bytearray()
                for i in range(0, len(data), 4):
                    mono_data.extend(data[i:i+2])
                send_data = mono_data
            else:
                send_data = data
                
            # 2. வால்யூமை அதிகரித்தல் (Software Amplifier)
            if sampwidth == 2: # 16-bit ஆடியோ
                samples = array.array('h', send_data) # 'h' = signed 16-bit
                for i in range(len(samples)):
                    amplified = int(samples[i] * VOLUME_MULTIPLIER)
                    
                    # அதிக சத்தத்தால் ஸ்பீக்கர் கிழியாமல் தடுக்க (Clipping limit)
                    if amplified > 32767: amplified = 32767
                    elif amplified < -32768: amplified = -32768
                    
                    samples[i] = amplified
                send_data = samples.tobytes()
                
            # 3. ESP32-க்கு அனுப்புதல்
            sock.sendto(send_data, (ESP32_IP, UDP_PORT))
            
            # துல்லியமான டைமிங்
            total_samples += chunk_size
            expected_time = total_samples / sample_rate
            current_time = time.time() - start_time
            
            if expected_time > current_time:
                time.sleep(expected_time - current_time)

    print("Playback Finished!")
except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()