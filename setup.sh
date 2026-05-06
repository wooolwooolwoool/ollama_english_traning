# Ubuntu
pip install fastapi uvicorn faster-whisper python-multipart piper-tts soundfile
ollama pull llama3.1:8B
mkdir models
cd models

# 英語音声モデル
# https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/low
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json
cd ..

