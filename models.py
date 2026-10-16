import os

import google.generativeai as genai

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the GEMINI_API_KEY environment variable before running this script.")

genai.configure(api_key=API_KEY)

for model in genai.list_models():
    print(model)