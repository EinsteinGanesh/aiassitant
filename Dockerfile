FROM python:3.11-slim

# FFmpeg மற்றும் Node.js இன்ஸ்டால் செய்ய (JavaScript challenge solver-க்காக)
RUN apt-get update && apt-get install -y ffmpeg nodejs curl

WORKDIR /app
COPY . .

# yt-dlp மற்றும் மற்ற லைப்ரரிகளை லேட்டஸ்ட் வெர்ஷனுக்கு அப்டேட் செய்ய
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "ai_assistant.py"]
