#!/usr/bin/env python3
"""
Test script to verify YouTube Music integration works correctly.
Run this before starting the main assistant to diagnose issues.
"""

import os
import sys
import subprocess
import yt_dlp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_assistant import (
    download_youtube_music_song,
    search_youtube_music,
    get_autoplay_suggestions,
    COOKIES_FILE
)


def test_ffmpeg():
    """Check if ffmpeg is available"""
    print("🔍 Checking ffmpeg...")
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ ffmpeg found: {version_line}")
            return True
        else:
            print(f"❌ ffmpeg error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ ffmpeg NOT FOUND! Please install ffmpeg.")
        print("   Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("   macOS: brew install ffmpeg")
        print("   Windows: https://ffmpeg.org/download.html")
        return False


def test_cookies():
    """Check if cookies.txt exists and is valid"""
    print(f"\n🔍 Checking cookies file ({COOKIES_FILE})...")
    if not os.path.exists(COOKIES_FILE):
        print(f"❌ {COOKIES_FILE} not found!")
        print("   Please export YouTube cookies using a browser extension.")
        print("   See README.md for instructions.")
        return False

    # Check file size
    size = os.path.getsize(COOKIES_FILE)
    print(f"✅ cookies.txt found ({size:,} bytes)")

    # Check for YouTube domains
    with open(COOKIES_FILE, 'r') as f:
        content = f.read()
        if 'youtube.com' in content or 'music.youtube.com' in content:
            print("✅ YouTube domain cookies detected")
        else:
            print("⚠️  No YouTube domain cookies found - may not work for YouTube Music")

    # Check for important cookies
    important_cookies = ['__Secure-3PSID', 'SID', 'HSID', 'SSID', 'APISID', 'SAPISID']
    found = [c for c in important_cookies if c in content]
    if found:
        print(f"✅ Important auth cookies found: {', '.join(found)}")
    else:
        print("⚠️  No standard Google auth cookies detected - may need fresh export")

    return True


def test_yt_dlp():
    """Test yt-dlp basic functionality"""
    print("\n🔍 Testing yt-dlp...")

    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': 'in_playlist',
    }

    if os.path.exists(COOKIES_FILE):
        ydl_opts['cookiefile'] = COOKIES_FILE

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Test search
            result = ydl.extract_info("ytsearch1:test song", download=False)
            if result and 'entries' in result and result['entries']:
                entry = result['entries'][0]
                print(f"✅ yt-dlp search works: Found '{entry.get('title', 'Unknown')}'")
                return True
            else:
                print("❌ yt-dlp search returned no results")
                return False
    except Exception as e:
        print(f"❌ yt-dlp error: {type(e).__name__}: {e}")
        return False


def test_search():
    """Test YouTube Music search"""
    print("\n🔍 Testing YouTube Music search...")
    try:
        results = search_youtube_music("Arijit Singh", max_results=3)
        if results:
            print(f"✅ Search works! Found {len(results)} results:")
            for i, r in enumerate(results, 1):
                title = r.get('title', 'Unknown')
                uploader = r.get('uploader', 'Unknown')
                vid = r.get('id', 'N/A')
                print(f"   {i}. {title} by {uploader} (ID: {vid})")
            return True
        else:
            print("❌ Search returned no results")
            return False
    except Exception as e:
        print(f"❌ Search error: {type(e).__name__}: {e}")
        return False


def test_download():
    """Test downloading a song (short test)"""
    print("\n🔍 Testing song download (this may take 10-30 seconds)...")
    print("   Searching for a short test track...")

    # Use a very popular short song for testing
    try:
        wav_file, title, metadata = download_youtube_music_song("never gonna give you up rick astley")
        if wav_file and os.path.exists(wav_file):
            size = os.path.getsize(wav_file)
            duration = size / (16000 * 2)  # 16kHz, 16-bit mono
            print(f"✅ Download successful!")
            print(f"   Title: {title}")
            print(f"   File: {wav_file} ({size:,} bytes, ~{duration:.1f}s)")
            print(f"   Metadata: {metadata}")

            # Cleanup
            try:
                os.remove(wav_file)
                for f in glob.glob("temp_audio.*") + glob.glob("youtube_audio.*"):
                    os.remove(f)
            except:
                pass
            return True
        else:
            print("❌ Download failed - no WAV file produced")
            return False
    except Exception as e:
        print(f"❌ Download error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_autoplay():
    """Test autoplay suggestions"""
    print("\n🔍 Testing autoplay suggestions...")
    try:
        suggestions = get_autoplay_suggestions("Shape of You", "Ed Sheeran", count=3)
        if suggestions:
            print(f"✅ Autoplay works! Got {len(suggestions)} suggestions:")
            for i, s in enumerate(suggestions, 1):
                print(f"   {i}. {s['title']} by {s['artist']} (ID: {s['video_id']})")
            return True
        else:
            print("❌ No autoplay suggestions returned")
            return False
    except Exception as e:
        print(f"❌ Autoplay error: {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("🧪 AI Assistant - YouTube Music Integration Test")
    print("=" * 60)

    all_passed = True

    # Run all tests
    all_passed &= test_ffmpeg()
    all_passed &= test_cookies()
    all_passed &= test_yt_dlp()
    all_passed &= test_search()

    # Only test download if basics pass
    if all_passed:
        all_passed &= test_download()
        all_passed &= test_autoplay()

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! YouTube Music integration is ready.")
        print("   You can now run: python ai_assistant.py")
    else:
        print("❌ SOME TESTS FAILED. Please fix the issues above before running.")
        print("   Common fixes:")
        print("   - Install ffmpeg")
        print("   - Export fresh YouTube cookies to cookies.txt")
        print("   - Update yt-dlp: pip install -U --pre 'yt-dlp[default]'")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import glob
    sys.exit(main())