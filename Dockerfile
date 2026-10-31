FROM python:3.11-slim

# FFmpeg மற்றும் Deno (YouTube JS Challenge Solver) இன்ஸ்டால் செய்தல்
RUN apt-get update && apt-get install -y ffmpeg curl unzip \
    && curl -fsSL https://deno.land/install.sh | sh
ENV PATH="/root/.deno/bin:$PATH"

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# YouTube பிளாக்கைத் தவிர்க்க டெய்லி அப்டேட் ஆகும் yt-dlp nightly வெர்ஷன்
RUN pip install --no-cache-dir -U --pre "yt-dlp[default]"

CMD ["python", "ai_assistant.py"]
