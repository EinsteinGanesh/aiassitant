# AI Assistant with YouTube Music Integration

A voice-activated AI assistant (JARVIS) for ESP32-based Bluetooth speakers with YouTube Music playback and autoplay support.

## Features

- 🎤 **Wake Word Detection**: "Hey Jarvis" offline detection using openWakeWord
- 🤖 **AI Conversation**: Gemini Live API for natural Tamil conversation
- 🎵 **YouTube Music Playback**: Search and play songs from YouTube Music using yt-dlp
- 🔁 **Autoplay**: Automatically plays similar songs when current song ends
- 🍪 **Cookie Authentication**: Uses browser cookies for YouTube Music premium/private content
- 📱 **ESP32 WebSocket Streaming**: Real-time 16kHz mono audio streaming to ESP32
- 🧠 **Persistent Memory**: SQLite-based long-term memory and chat history
- 🏠 **Smart Home Control**: Device control via voice commands

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **ffmpeg** (required for audio conversion)
3. **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/)
4. **YouTube Cookies** (for YouTube Music access)

### Installation

```bash
# Clone the repository
git clone https://github.com/EinsteinGanesh/aiassistant.git
cd aiassistant

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Install Python dependencies
pip install -r requirements.txt
pip install -U --pre "yt-dlp[default]"
```

### Get YouTube Cookies (Required for YouTube Music)

1. Install a cookie exporter browser extension:
   - **Chrome/Edge**: [Get cookies.txt LOCALLY](https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
   - **Firefox**: [cookies.txt](https://addons.mozilla.org/en-US/firefox/addon/cookies-txt/)

2. Go to [music.youtube.com](https://music.youtube.com) and sign in

3. Export cookies for `youtube.com` and `music.youtube.com` domains

4. Save as `cookies.txt` in the project root

### Configuration

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-gemini-api-key-here"

# Run the assistant
./start.sh
# OR directly:
python ai_assistant.py
```

### Docker Deployment

```bash
# Build the image
docker build -t ai-assistant .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY="your-key" \
  -v $(pwd)/cookies.txt:/app/cookies.txt:ro \
  -v $(pwd)/jarvis_memory.db:/app/jarvis_memory.db \
  --name jarvis \
  ai-assistant
```

## Usage

### Voice Commands (Tamil)

| Command | Action |
|---------|--------|
| "ஹே ஜார்விஸ்" | Wake up the assistant |
| "இரவன் பாடல் பLAY பண்ணு" | Play "Iravan" song from YouTube Music |
| "அரிஜித் சிங் பாடல்கள் பLAY பண்ணு" | Play Arijit Singh songs |
| "நிறுத்து" / "பஆச்" | Pause/Resume music |
| "அடுத்த பாடல்" | Skip to next song (autoplay) |
| "வால்யூம் அதிகரிக்க" | Increase volume |
| "வால்யூம் குறை" | Decrease volume |

### Button Controls (ESP32)

| Button Press | Action |
|--------------|--------|
| Short Press | Wake / Pause-Resume music |
| Long Press | Force wake (bypass wake word) |
| Volume Up/Down | Adjust volume |

## Architecture

```
┌─────────────┐     WebSocket      ┌──────────────┐
│   ESP32     │ ◄─────────────────► │  AI Assistant │
│  (Speaker)  │   16kHz Mono PCM   │  (Python)     │
└─────────────┘                     └──────┬───────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
            ┌─────────────┐         ┌──────────────┐         ┌─────────────┐
            │  Gemini     │         │   yt-dlp     │         │  SQLite     │
            │  Live API   │         │ YouTube Music│         │  Memory     │
            └─────────────┘         └──────────────┘         └─────────────┘
```

## Key Components

### `ai_assistant.py` - Main Application
- WebSocket server for ESP32 communication
- Gemini Live session management
- Music playback orchestration
- Wake word detection loop

### YouTube Music Integration (`download_youtube_music_song`)
- Searches YouTube Music via yt-dlp
- Uses cookies for authentication
- Downloads best audio quality (m4a)
- Converts to 16kHz mono WAV for ESP32 via ffmpeg
- Falls back to JioSaavn if YouTube fails

### Autoplay System
- Pre-fetches similar song suggestions in background
- Uses YouTube Music search for "similar songs mix"
- Queues next song automatically when current ends
- Maintains continuous playback experience

## Troubleshooting

### "ffmpeg not found"
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### "YouTube Music not playing / Sign in required"
- Ensure `cookies.txt` is in project root
- Cookies must be from a logged-in YouTube Music session
- Re-export cookies if they expire (typically weekly)

### "No audio on ESP32"
- Check WebSocket connection (port 8000)
- Verify ESP32 IP matches server
- Check audio format: 16kHz, 16-bit, Mono PCM

### "yt-dlp extraction failed"
```bash
# Update yt-dlp to latest nightly
pip install -U --pre "yt-dlp[default]"
```

## Development

### Project Structure
```
aiassistant/
├── ai_assistant.py       # Main application
├── audio_server.py       # UDP audio test server
├── play_to_esp32.py      # Test audio streaming to ESP32
├── send_audio.py         # UDP audio sender
├── stt_test.py           # Speech-to-text test
├── voice_test.py         # Voice loopback test
├── gemini_test.py        # Gemini API test
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── start.sh              # Startup script
├── cookies.txt           # YouTube cookies (gitignored)
├── jarvis_memory.db      # SQLite memory (gitignored)
└── README.md             # This file
```

### Running Tests
```bash
# Test Gemini API
python gemini_test.py

# Test audio recording from ESP32
python stt_test.py

# Test voice loopback
python voice_test.py

# Test audio streaming to ESP32
python play_to_esp32.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with ESP32 hardware
5. Submit a pull request

## License

MIT License - Feel free to use and modify for your projects.

## Credits

- [openWakeWord](https://github.com/dscripka/openWakeWord) for wake word detection
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube Music extraction
- [Google Gemini](https://ai.google.dev/) for AI conversation
- [websockets](https://github.com/python-websockets/websockets) for real-time communication