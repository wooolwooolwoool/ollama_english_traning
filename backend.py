from fastapi import Body, FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import subprocess
import json
import shutil
from faster_whisper import WhisperModel
import os
import requests
import datetime

OLLAMA_URL = "http://localhost:11434/api/generate"

app = FastAPI()

# Whisper
model = WhisperModel("base", compute_type="int8")

summary = {
    "topic": "",
    "conversation_flow": "",
    "user_level": "C"
}
last_question = "What topic should we talk about today?"

PROMPT = """
You are an English tutor.

The user is learning English.

Your tasks:
1. Correct the user's sentence
2. Explain mistakes in English
3. Ask the next question in simple English
4. Update the conversation summary

IMPORTANT RULES:
- Keep the conversation connected to the previous context
- The next question must naturally follow the conversation flow
- Ask 2 to 3 short sentences for the next question (not too long)
- Focus on grammar and word usage
- Keep explanations short and clear in English
- Use simple and natural English
- When someone suggests changing the topic, change the summary

STRICT OUTPUT RULES:
- Output ONLY valid JSON in English
- No extra text
- No markdown

{
  "correction": "...",
  "explanation": "...",
  "next_question": "...",
  "summary": {
    "topic": "...",
    "conversation_flow": "...",
    "user_level": "..."
  }
}
"""

session = requests.Session()

# --- STT ---
def transcribe(path):
    segments, _ = model.transcribe(path, language="en", task="transcribe")
    return " ".join([s.text for s in segments]).strip()


def ask_llm(text):

    prompt = f"""
{PROMPT}

Current summary:
{json.dumps(summary, ensure_ascii=False)}

Current question: {last_question}

User: {text}
"""

    response = session.post(
        OLLAMA_URL,
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )

    return response.json()["response"]

# --- TTS ---
def tts(text, output="reply.wav"):
    subprocess.run([
        "piper",
        "--model", "models/en_US-lessac-low.onnx",
        "--output_file", output
    ], input=text, text=True)

@app.post("/api/voice")
async def voice(file: UploadFile = File(...)):
    with open("input.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"{datetime.datetime.now()} - Received voice input: {file.filename}")
    user_text = transcribe("input.wav")
    print(f"{datetime.datetime.now()} - Transcribed text: {user_text}")
    raw = ask_llm(user_text)
    print(f"{datetime.datetime.now()} - Generated text: {raw}")

    try:
        data = json.loads(raw)
    except:
        data = {
            "correction": user_text,
            "explanation": "Failed to parse LLM response.",
            "next_question": "Can you try again?",
            "summary": {
                "topic": "",
                "conversation_flow": "",
                "user_level": ""
            }
        }

    summary.update(data["summary"])
    global last_question
    last_question = data["next_question"]
    # TTS生成
    tts(data["explanation"] + " " + data["next_question"], "reply.wav")

    return {
        "user": user_text,
        **data,
        "audio_url": "/audio/reply.wav"
    }

@app.get("/audio/{filename}")
def get_audio(filename: str):
    if os.path.exists(filename):
        return FileResponse(filename)
    return {"error": "file not found"}

@app.post("/api/text")
async def text_api(data: dict = Body(...)):
    user_text = data.get("text", "")

    raw = ask_llm(user_text)

    try:
        result = json.loads(raw)
    except:
        result = {
            "correction": user_text,
            "explanation": "Failed to parse LLM response.",
            "next_question": "Can you try again?"
        }

    # サマリ更新
    global summary
    summary.update(result.get("summary", {}))

    # TTS
    tts(result["next_question"], "reply.wav")

    return {
        "user": user_text,
        **result,
        "audio_url": "/audio/reply.wav"
    }

@app.post("/api/reset")
def reset():
    global history, summary, last_question
    summary = {
        "topic": "",
        "conversation_flow": "",
        "user_level": "C"
    }
    last_question = "What topic should we talk about today?"
    return {"status": "ok"}

@app.get("/")
def index():
    return HTMLResponse(open("index.html", encoding="utf-8").read())