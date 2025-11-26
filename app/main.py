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
    )
    pipeline = KPipeline(lang_code="a", model=model, device='cuda')

@app.on_event("startup")
async def startup_event():
    await load_asr_model()
    await load_tts_model()
    global manager
    manager = ConnectionManager()

async def send_ai_audio(websocket: WebSocket, text: str, pipeline, manager):
    try:
        generator = pipeline(text, voice="af_heart", split_pattern=r"(?:\r?\n)|(?<=\.)\s+(?=[A-Z])")
        await manager.send_message(websocket, "audio_start", "Audio incoming...")
        for i, (gs, ps, audio) in enumerate(generator):
            await asyncio.sleep(0)
            logger.info(f"Attempting to send audio chunk {i}. Task cancelled: {asyncio.current_task().cancelled()}")
            buf = io.BytesIO()
            sf.write(buf, audio, 24000, format='WAV')
            buf.seek(0)
            await websocket.send_bytes(buf.read())
        await manager.send_message(websocket, "audio_end", "Done playing audio")
    except asyncio.CancelledError:
        logger.info("AI audio task received cancellation signal.")
        logger.info("AI audio playback was cancelled by user barge-in.")
        await manager.send_message(websocket, "audio_stop", "Playback stopped.")

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("WebSocket connected")

    stream_state = None
    user_is_speaking = False
    silence_chunk_count = 0
    speaking_chunk_count = 0 # Initialize speaking chunk count
    max_allowed_silence = 20 # Increased to allow for more natural pauses
    min_allowed_speaking = 10 # Decreased to allow shorter utterances
    ai_audio_task = None

    POST_SPEECH_BUFFER_CHUNKS = 10 # Number of chunks to buffer after speech ends
    post_speech_buffer = deque(maxlen=POST_SPEECH_BUFFER_CHUNKS)

    try:
        while True:
            raw_audio = await websocket.receive_bytes()
            raw_audio = await websocket.receive_bytes()
            post_speech_buffer.append(raw_audio) # Always append to buffer

            audio_np = np.frombuffer(raw_audio, dtype=np.int16)
            
            audio_tensor = torch.from_numpy(audio_np).float()
            if audio_tensor.ndim > 1:
                audio_tensor = audio_tensor.mean(axis=1)
            
            resampler = torchaudio.transforms.Resample(
                orig_freq=44100, 
                new_freq=VAD_SAMPLE_RATE
            )
            resampled_audio = resampler(audio_tensor).numpy().astype(np.int16)
            
            # VAD framing
            VAD_FRAME_SIZE_MS = 30 # ms
            vad_frame_size_samples = int(VAD_SAMPLE_RATE * VAD_FRAME_SIZE_MS / 1000)
            vad_frame_size_bytes = vad_frame_size_samples * 2 # 2 bytes per sample for int16

            webrtcvad_speech_detected = False
            # Process audio in VAD-compatible frames
            for i in range(0, len(resampled_audio) - vad_frame_size_samples + 1, vad_frame_size_samples):
                vad_frame = resampled_audio[i:i + vad_frame_size_samples]
                if is_speech(vad_frame.tobytes(), VAD_SAMPLE_RATE):
                    webrtcvad_speech_detected = True
                    break

            audio_energy = get_normalized_audio_energy(raw_audio)
            speech_detected = webrtcvad_speech_detected or (audio_energy > AUDIO_ENERGY_SPEAKING_THRESHOLD)

            if speech_detected:
                if ai_audio_task and not ai_audio_task.done():
                    logger.info("Attempting to cancel AI audio task due to user interruption.")
                    ai_audio_task.cancel()
                    logger.info(f"AI audio task state after cancel: done={ai_audio_task.done()}, cancelled={ai_audio_task.cancelled()}")
                
                user_is_speaking = True
                silence_chunk_count = 0
                speaking_chunk_count += 1 # Increment speaking chunk count
                if stream_state is None:
                    stream_state = recognizer_en.create_stream()
                
                audio_np_asr, _ = bytes_to_audio(raw_audio, sample_rate=44100)
                stream_state.accept_waveform(44100, audio_np_asr.tolist())
                
                # Ensure decoding happens for the latest waveform
                while recognizer_en.is_ready(stream_state):
                    recognizer_en.decode_streams([stream_state])
                
                # Get and send interim text after processing the current chunk
                interim_text = recognizer_en.get_result(stream_state)
                if interim_text: # Only send if not empty
                    logger.debug(f"Sending interim_transcription: {interim_text}")
                    await manager.send_message(websocket, "interim_transcription", interim_text)

            elif user_is_speaking:
                logger.debug(f"Silence frames: {silence_chunk_count}, Audio energy: {audio_energy:.4f}, Speaking: {speech_detected}, Speaking chunks: {speaking_chunk_count}")
                silence_chunk_count += 1
                if silence_chunk_count > max_allowed_silence:
                    user_is_speaking = False
                    # Check if the utterance was long enough
                    if speaking_chunk_count < min_allowed_speaking:
                        logger.info(f"Utterance too short ({speaking_chunk_count} chunks). Discarding.")
                        speaking_chunk_count = 0 # Reset for next utterance
                        stream_state = None # Reset ASR stream
                        await manager.send_message(websocket, "status", "Listening... Speak more...")
                        continue # Skip to next audio chunk

                    speaking_chunk_count = 0 # Reset for next utterance
                    if stream_state:
                        # Feed the post-speech buffer to the ASR stream
                        for buffered_chunk in post_speech_buffer:
                            audio_np_asr, _ = bytes_to_audio(buffered_chunk, sample_rate=44100)
                            stream_state.accept_waveform(44100, audio_np_asr.tolist())
                        post_speech_buffer.clear() # Clear the buffer after use

                        stream_state.input_finished()
                        while recognizer_en.is_ready(stream_state):
                            recognizer_en.decode_streams([stream_state])
                        
                        final_transcription = recognizer_en.get_result(stream_state)
                        await manager.send_message(websocket, "transcription", final_transcription)

                        if final_transcription:
                            response_text = ""
                            response = conversation_gen_model.generate_content(
                                f"User: {final_transcription}\nAI:", stream=True
                            )
                            for data in response:
                                response_text += data.text
                            
                            await manager.send_message(websocket, "llm_response", response_text)
                            ai_audio_task = asyncio.create_task(
                                send_ai_audio(websocket, response_text, pipeline, manager)
                            )
                        stream_state = None

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")

@app.get("/")
async def get():
    with open("static/index_speech_to_text.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)