import os

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before running this test.")

print("Gemini-யைத் தொடர்புகொள்கிறோம், காத்திருக்கவும்...")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('models/gemini-3.5-flash')
    
    print("கேள்வி அனுப்புகிறோம்...")
    response = model.generate_content("Hello Gemini, are you there? Reply in one short sentence.")
    
    print("\n✅ வெற்றி! Gemini பதில்:")
    print(response.text)
    
except Exception as e:
    print("\n❌ Error வந்துள்ளது. காரணம்:")
    print(e)