#!/bin/bash
# Startup script for AI Assistant with YouTube Music

set -e

echo "🚀 Starting AI Assistant with YouTube Music integration..."

# Check for required environment variables
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ ERROR: GEMINI_API_KEY environment variable not set!"
    echo "Please set it: export GEMINI_API_KEY='your-api-key-here'"
    exit 1
fi

# Check for cookies.txt
if [ ! -f "cookies.txt" ]; then
    echo "⚠️  WARNING: cookies.txt not found!"
    echo "YouTube Music may not work without authentication cookies."
    echo "Please export your YouTube/Google cookies to cookies.txt"
fi

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ERROR: ffmpeg not found! Please install ffmpeg."
    exit 1
fi

# Install/update Python dependencies
echo "📦 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -U --pre "yt-dlp[default]"

# Run the assistant
echo "✅ Starting AI Assistant..."
exec python ai_assistant.py