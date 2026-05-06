from fastapi import Body, FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import json
import shutil
from faster_whisper import WhisperModel
import os
import requests
import datetime
import copy
from piper import PiperVoice
import soundfile as sf

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_PATH = "models/en_US-lessac-low.onnx"
app = FastAPI()

# STT
model = WhisperModel("base", compute_type="int8")
# TTS
voice = PiperVoice.load(MODEL_PATH)

summary = {
    "topic": "",
    "conversation_flow": ""
}
last_question = "What topic should we talk about today?"
last_state = None
last_question = "What did you do yesterday?"

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
- If there is correct syntax, teach it
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
    "conversation_flow": "..."
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
        timeout=180
    )

    return response.json()["response"]

# --- TTS ---
def tts(text, output="reply.wav"):
    audio = []
    sample_rate = voice.config.sample_rate
    for chunk in voice.synthesize(text):
        audio.extend(chunk.audio_float_array)
    sf.write(output, audio, sample_rate)

def process_text(user_text):
    try:
        raw = ask_llm(user_text)
        print(f"{datetime.datetime.now()} - Generated text: {raw}")

        global last_question, last_state, summary
        data = json.loads(raw)
        last_state = {
            "summary": copy.deepcopy(summary),
            "question": copy.deepcopy(last_question)
        }
        summary.update(data["summary"])
        last_question = data["next_question"]
    except:
        data = {
            "correction": user_text,
            "explanation": "Failed to parse LLM response.",
            "next_question": "Can you try again?",
            "summary": summary
        }
    # TTS生成
    tts(data["explanation"] + " " + data["next_question"], "reply.wav")
    audio_url = "/audio/reply.wav"


    return {
        "user": user_text,
        **data,
        "audio_url": audio_url
    }

@app.post("/api/voice")
async def voice_api(file: UploadFile = File(...)):
    with open("input.wav", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"{datetime.datetime.now()} - Received voice input: {file.filename}")
    user_text = transcribe("input.wav")
    print(f"{datetime.datetime.now()} - Transcribed text: {user_text}")
    return process_text(user_text)


@app.get("/audio/{filename}")
def get_audio(filename: str):
    if os.path.exists(filename):
        return FileResponse(filename)
    return {"error": "file not found"}

@app.post("/api/text")
async def text_api(data: dict = Body(...)):
    user_text = data.get("text", "")
    return process_text(user_text)

@app.post("/api/reset")
def reset():
    global summary, last_question
    summary = {
        "topic": "",
        "conversation_flow": ""
    }
    last_question = "What topic should we talk about today?"
    return {"status": "ok"}

@app.post("/api/back")
def go_back():
    global summary, last_question, last_state

    if not last_state:
        return {"status": "no_history"}

    summary = last_state["summary"]
    last_question = last_state["question"]
    print(f"{datetime.datetime.now()} - Back to : {summary}")
    print(f"{datetime.datetime.now()} - Back to : {last_question}")

    # 一回だけ戻れるようにする
    last_state = None

    return {
        "status": "ok",
        "question": last_question
    }

@app.get("/")
def index():
    return HTMLResponse(open("index.html", encoding="utf-8").read())