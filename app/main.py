

### Revised  by AI v2 # #######
import asyncio
import json
import logging
import io
import time
from typing import Optional

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from IPython.display import display, Audio
from sherpa_onnx import OnlineRecognizer

from kokoro import KPipeline, KModel

from src.configs.connections import ConnectionManager
from src.utils import bytes_to_audio, get_normalized_audio_energy
from src.transcription import transcribe_microphone_stream, detect_speech_in_chunk
from src.configs import conversation_gen_model


# Logging setup
logging.basicConfig(level=logging.INFO,filename='app.log',format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(funcName)s')
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals
recognizer_en: OnlineRecognizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
manager = None
pipeline = None


# ---------------------- Model Loading ---------------------- #

async def load_asr_model():
    """Load the ASR model in memory"""
    global recognizer_en
    logger.info("Loading ASR model...")

    try:
        recognizer_en = OnlineRecognizer.from_transducer(
            tokens="./models/Kroko-Streaming-ASR-Python/en_tokens.txt",
            encoder="./models/Kroko-Streaming-ASR-Python/en_encoder.onnx",
            decoder="./models/Kroko-Streaming-ASR-Python/en_decoder.onnx",
            joiner="./models/Kroko-Streaming-ASR-Python/en_joiner.onnx",
            num_threads=4,
            decoding_method="modified_beam_search",
            debug=False,
        )
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        raise


async def load_tts_model():
    """Load the TTS model in memory"""
    global pipeline
    model = KModel(
        config=r"C:\Users\akliv\OneDrive\Desktop\Akesh kumar\forks\Audio2Audio\models\Kroko-82M\config.json",
        model=r"C:\Users\akliv\OneDrive\Desktop\Akesh kumar\forks\Audio2Audio\models\Kroko-82M\kokoro-v1_0.pth",
    )
    pipeline = KPipeline(lang_code="a", model=model)


@app.on_event("startup")
async def startup_event():
    await load_asr_model()
    await load_tts_model()
    global manager
    manager = ConnectionManager()


# ---------------------- Audio Processing ---------------------- #

async def process_audio_buffer(
    buffer: list[bytes], stream_state, websocket: WebSocket
) -> Optional[str]:
    """Process accumulated audio buffer and return transcription"""
    logger.info("Processing audio buffer...")

    if not buffer:
        return None
    
    audio_data = b"".join(buffer)
    audio_np, sample_rate = bytes_to_audio(audio_data, sample_rate=44100)
    sf.write("output.wav", audio_np, sample_rate)

    current_text, stream_state = transcribe_microphone_stream(
        recognizer=recognizer_en,
        audio_chunk_with_sample_rate=(audio_np, sample_rate),
        stream_state=stream_state,
    )

    logger.info(f"Transcription: {current_text}")
    await manager.send_message(websocket, "transcription", current_text)

    return current_text, stream_state


# ---------------------- WebSocket Handler ---------------------- #

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("WebSocket connected")

    audio_chunks = []
    user_is_speaking = False
    silence_chunk_count = 0
    speaking_chunk_count = 0
    min_allowed_speaking = 40
    max_allowed_silence = 10 # 
    AUDIO_ENERGY_SPEAKING_THRESHOLD = 0.008
    is_responded = False
    stream_context = None
    

    try:
        while True:
            raw_audio = await websocket.receive_bytes()
            audio_energy = get_normalized_audio_energy(raw_audio)
            speech_detected = detect_speech_in_chunk(raw_audio, sample_rate=44100)

            # logger.info(f"Silence frames: {silence_chunk_count}, Audio energy: {audio_energy}, Speaking: {speech_detected}")

            if speech_detected:
                is_responded = False
                user_is_speaking = True
                silence_chunk_count = 0
                speaking_chunk_count += 1
                audio_chunks.append(raw_audio)

                status_msg = "Listening..." if audio_energy > AUDIO_ENERGY_SPEAKING_THRESHOLD else "Listening... Speak louder..."
                await manager.send_message(websocket, "status", status_msg)
                
                
            elif user_is_speaking:
                silence_chunk_count += 1
                audio_chunks.append(raw_audio)

                if silence_chunk_count > max_allowed_silence and not is_responded:
                    await manager.send_message(websocket, "status", "Thinking...")
                    if speaking_chunk_count < min_allowed_speaking:
                        await manager.send_message(websocket, "status", "Listening... Speak More...")
                        continue
                    
                    transcription, stream_context = await process_audio_buffer(
                        audio_chunks,
                        stream_context,
                        websocket,
                    )
                    if transcription is None or transcription == "":
                        manager.send_message(websocket, "status", "Listening...")
                        continue

                    response_text = ""
                    response = conversation_gen_model.generate_content(
                        f"User: {transcription}\nAI:", stream=True
                    )
                    for data in response:
                        response_text += data.text
                    logger.info(f"\n\nQuery: {transcription}\nLLM Response: {response_text}\n\n")
                    await manager.send_message(websocket, "llm_response", response_text)

                    generator = pipeline(response_text, voice="af_heart", split_pattern=r"(?:\r?\n)|(?<=\.)\s+(?=[A-Z])")
                    await manager.send_message(websocket, "audio_start", "Audio incoming...")

                    for i, (gs, ps, audio) in enumerate(generator):
                        logger.info(f"Sending audio chunk {i}: {gs}, {ps}")
                        sf.write(f"{i}.wav", audio, 24000)
                        logger.info(f"Audio generated for text chunk :{gs[0:10]+'...'+gs[-10:]}")
                        buf = io.BytesIO()
                        sf.write(buf, audio, 24000, format='WAV')
                        buf.seek(0)
                        await websocket.send_bytes(buf.read())

                    await manager.send_message(websocket, "audio_end", "Done playing audio")

                    # Reset
                    audio_chunks = []
                    user_is_speaking = False
                    stream_context = None
                    is_responded = True
                    await manager.send_message(websocket, "status", "Ready")

            else:
                logger.info("Not speaking, discarding audio")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")


# ---------------------- Web Routes ---------------------- #

@app.get("/")
async def get():
    with open("static/index_speech_to_text.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
