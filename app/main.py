import asyncio
import json
import logging
import io
from typing import Optional

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torchaudio

from sherpa_onnx import OnlineRecognizer
from kokoro import KPipeline, KModel

from src.configs.connections import ConnectionManager
from src.utils import bytes_to_audio, get_normalized_audio_energy
from src.transcription.voice_activity_detection import is_speech
from src.configs import conversation_gen_model

from collections import deque

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(funcName)s')
logger = logging.getLogger(__name__)
AUDIO_ENERGY_SPEAKING_THRESHOLD = 0.008
# FastAPI app setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals
recognizer_en: OnlineRecognizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
manager = None
pipeline = None
VAD_SAMPLE_RATE = 16000

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
            num_threads=5,
            decoding_method="modified_beam_search",
            debug=False,
            provider="cuda",
            device=0,
        )
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        raise

async def load_tts_model():
    """Load the TTS model in memory"""
    global pipeline
    model = KModel(
        config="./models/Kroko-82M/config.json",
        model="./models/Kroko-82M/kokoro-v1_0.pth",
    ).to('cuda')
    pipeline = KPipeline(lang_code="a", model=model, device='cuda')

@app.on_event("startup")
async def startup_event():
    await load_asr_model()
    await load_tts_model()
    global manager
    manager = ConnectionManager()

import re

def extract_sentences(text: str):
    """
    Splits text into sentences. Returns a list of complete sentences
    and the remaining unfinished text.
    """
    # Matches sentences ending in ., ?, !, or a newline
    pattern = re.compile(r'([^.!?\n]+[.!?\n]+)')
    matches = pattern.findall(text)
    
    if not matches:
        return [], text
        
    matched_len = sum(len(m) for m in matches)
    remainder = text[matched_len:]
    
    return [m.strip() for m in matches if m.strip()], remainder

async def generate_and_send_response(websocket: WebSocket, prompt: str, voice: str, pipeline, manager):
    try:
        await manager.send_message(websocket, "audio_start", "Audio incoming...")
        
        # Start Gemini API call asynchronously with streaming
        response = await conversation_gen_model.generate_content_async(prompt, stream=True)
        
        text_buffer = ""
        loop = asyncio.get_running_loop()
        
        async for chunk in response:
            try:
                if chunk.text:
                    text_buffer += chunk.text
                    sentences, text_buffer = extract_sentences(text_buffer)
                    for sentence in sentences:
                        logger.info(f"Generated sentence: {sentence}")
                        await manager.send_message(websocket, "llm_response_chunk", sentence)
                        
                        def run_tts(text_to_speak, v_name):
                            return list(pipeline(text_to_speak, voice=v_name))
                        
                        # Generate audio for the sentence in thread pool
                        tts_results = await loop.run_in_executor(None, run_tts, sentence, voice)
                        for gs, ps, audio in tts_results:
                            await asyncio.sleep(0)
                            buf = io.BytesIO()
                            sf.write(buf, audio, 24000, format='WAV')
                            buf.seek(0)
                            await websocket.send_bytes(buf.read())
            except Exception as inner_e:
                logger.error(f"Error processing text chunk: {inner_e}")
                
        # Send remaining text
        if text_buffer.strip():
            sentence = text_buffer.strip()
            logger.info(f"Generated final sentence: {sentence}")
            await manager.send_message(websocket, "llm_response_chunk", sentence)
            
            def run_tts(text_to_speak, v_name):
                return list(pipeline(text_to_speak, voice=v_name))
                
            tts_results = await loop.run_in_executor(None, run_tts, sentence, voice)
            for gs, ps, audio in tts_results:
                await asyncio.sleep(0)
                buf = io.BytesIO()
                sf.write(buf, audio, 24000, format='WAV')
                buf.seek(0)
                await websocket.send_bytes(buf.read())
                
        await manager.send_message(websocket, "audio_end", "Done playing audio")
    except asyncio.CancelledError:
        logger.info("AI response streaming task received cancellation signal.")
        await manager.send_message(websocket, "audio_stop", "Playback stopped.")
    except Exception as e:
        logger.error(f"Error in generate_and_send_response: {e}")
        await manager.send_message(websocket, "error", f"Response error: {str(e)}")

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("WebSocket connected")

    stream_state = None
    user_is_speaking = False
    silence_chunk_count = 0
    speaking_chunk_count = 0
    
    # Default configs
    current_voice = "af_heart"
    max_allowed_silence = 8    # ~1 sec at 128ms per chunk
    min_allowed_speaking = 3   # ~380ms at 128ms per chunk
    chunk_duration_ms = 128.0  # assumed 2048 samples at 16kHz
    audio_energy_speaking_threshold = 0.020
    
    ai_response_task = None
    barge_in_consecutive_chunks = 0

    POST_SPEECH_BUFFER_CHUNKS = 10
    post_speech_buffer = deque(maxlen=POST_SPEECH_BUFFER_CHUNKS)

    try:
        while True:
            # We receive websocket message (which can be bytes or text)
            message = await websocket.receive()
            
            if "bytes" in message:
                raw_audio = message["bytes"]
                post_speech_buffer.append(raw_audio)
                
                audio_np = np.frombuffer(raw_audio, dtype=np.int16)
                # No resampling needed since client streams 16kHz natively
                resampled_audio = audio_np
                
                # VAD framing
                VAD_FRAME_SIZE_MS = 30 # ms
                vad_frame_size_samples = int(VAD_SAMPLE_RATE * VAD_FRAME_SIZE_MS / 1000)

                webrtcvad_speech_detected = False
                for i in range(0, len(resampled_audio) - vad_frame_size_samples + 1, vad_frame_size_samples):
                    vad_frame = resampled_audio[i:i + vad_frame_size_samples]
                    if is_speech(vad_frame.tobytes(), VAD_SAMPLE_RATE):
                        webrtcvad_speech_detected = True
                        break

                audio_energy = get_normalized_audio_energy(raw_audio)
                speech_detected = webrtcvad_speech_detected or (audio_energy > audio_energy_speaking_threshold)

                if speech_detected:
                    # User is speaking. If AI is speaking/generating, cancel it immediately (barge-in)!
                    if ai_response_task and not ai_response_task.done():
                        # While AI is speaking, be conservative to avoid false barge-ins:
                        # 1. Require webrtcvad_speech_detected (actual voice frequencies), ignoring energy-only triggers.
                        # 2. Require consecutive chunks to filter out transient noises/clicks.
                        if webrtcvad_speech_detected:
                            barge_in_consecutive_chunks += 1
                        else:
                            barge_in_consecutive_chunks = 0

                        if barge_in_consecutive_chunks >= 2:
                            logger.info(f"Cancelling AI response task due to user speech detection (barge-in). Consecutive chunks: {barge_in_consecutive_chunks}")
                            ai_response_task.cancel()
                            barge_in_consecutive_chunks = 0
                    else:
                        barge_in_consecutive_chunks = 0
                    
                    user_is_speaking = True
                    silence_chunk_count = 0
                    speaking_chunk_count += 1
                    
                    if stream_state is None:
                        stream_state = recognizer_en.create_stream()
                    
                    audio_np_asr, _ = bytes_to_audio(raw_audio, sample_rate=16000)
                    stream_state.accept_waveform(16000, audio_np_asr.tolist())
                    
                    while recognizer_en.is_ready(stream_state):
                        recognizer_en.decode_streams([stream_state])
                    
                    interim_text = recognizer_en.get_result(stream_state)
                    if interim_text:
                        await manager.send_message(websocket, "interim_transcription", interim_text)

                else:
                    barge_in_consecutive_chunks = 0
                    if user_is_speaking:
                        silence_chunk_count += 1
                        if silence_chunk_count > max_allowed_silence:
                            user_is_speaking = False
                            
                            if speaking_chunk_count < min_allowed_speaking:
                                logger.info(f"Utterance too short ({speaking_chunk_count} chunks). Discarding.")
                                speaking_chunk_count = 0
                                stream_state = None
                                await manager.send_message(websocket, "status", "Listening... Speak more...")
                                continue

                            speaking_chunk_count = 0
                            if stream_state:
                                # Feed the post-speech buffer to ASR
                                for buffered_chunk in post_speech_buffer:
                                    audio_np_asr, _ = bytes_to_audio(buffered_chunk, sample_rate=16000)
                                    stream_state.accept_waveform(16000, audio_np_asr.tolist())
                                post_speech_buffer.clear()

                                stream_state.input_finished()
                                while recognizer_en.is_ready(stream_state):
                                    recognizer_en.decode_streams([stream_state])
                                
                                final_transcription = recognizer_en.get_result(stream_state)
                                await manager.send_message(websocket, "transcription", final_transcription)

                                if final_transcription:
                                    prompt = f"User: {final_transcription}\nAI:"
                                    if ai_response_task and not ai_response_task.done():
                                        ai_response_task.cancel()
                                    
                                    ai_response_task = asyncio.create_task(
                                        generate_and_send_response(
                                            websocket, prompt, current_voice, pipeline, manager
                                        )
                                    )
                                stream_state = None
                            
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "user_interrupted":
                        if ai_response_task and not ai_response_task.done():
                            logger.info("Cancelling AI response task due to client interruption signal.")
                            ai_response_task.cancel()
                            
                    elif msg_type == "update_settings":
                        if "voice" in data:
                            current_voice = data["voice"]
                            logger.info(f"Updated current voice to: {current_voice}")
                        if "max_silence_ms" in data:
                            max_silence_ms = data["max_silence_ms"]
                            max_allowed_silence = max(1, int(max_silence_ms / chunk_duration_ms))
                            logger.info(f"Updated max_allowed_silence to {max_allowed_silence} chunks ({max_silence_ms}ms)")
                        if "min_speaking_ms" in data:
                            min_speaking_ms = data["min_speaking_ms"]
                            min_allowed_speaking = max(1, int(min_speaking_ms / chunk_duration_ms))
                            logger.info(f"Updated min_allowed_speaking to {min_allowed_speaking} chunks ({min_speaking_ms}ms)")
                        if "energy_threshold" in data:
                            audio_energy_speaking_threshold = data["energy_threshold"]
                            logger.info(f"Updated audio_energy_speaking_threshold to {audio_energy_speaking_threshold}")
                except Exception as text_e:
                    logger.error(f"Error parsing text message: {text_e}")
                        
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket endpoint exception: {e}")
        try:
            manager.disconnect(websocket)
        except Exception:
            pass

@app.get("/")
async def get():
    with open("static/index_speech_to_text.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)