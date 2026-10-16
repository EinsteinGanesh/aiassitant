# aiassitant

## Setup

Install the Python dependencies:

```powershell
python -m pip install -r requirement.txt
```

Set your Gemini API key in the environment before running the assistant:

```powershell
$env:GEMINI_API_KEY = "your-api-key"
python ai_assistant.py
```

Do not commit API keys, local recordings, generated audio, or the local SQLite database.
