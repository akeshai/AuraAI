import asyncio
import json
import logging
import io
import time
from collections import deque

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Preload CUDA libraries from virtualenv before loading ASR/TTS engines
from src.utils import preload_cuda_libraries
preload_cuda_libraries()

from src.configs.connections import ConnectionManager
from src.utils import bytes_to_audio, get_normalized_audio_energy, extract_sentences, ensure_models_exist
from src.transcription.voice_activity_detection import is_speech
from src.transcription import StreamingASR
from src.speech_generation import TTSManager
from src.configs import conversation_gen_model

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - %(lineno)d - %(funcName)s'
)
logger = logging.getLogger(__name__)

# FastAPI app setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Globals (Managers initialized on startup)
asr_manager: StreamingASR = None
tts_manager: TTSManager = None
manager: ConnectionManager = None

VAD_SAMPLE_RATE = 16000

@app.on_event("startup")
async def startup_event():
    """Load models and connection manager on startup"""
    global asr_manager, tts_manager, manager
    
    # Ensure all required models exist before loading
    ensure_models_exist()
    
    # Initialize ASR
    asr_manager = StreamingASR(
        tokens_path="./models/Kroko-Streaming-ASR-Python/en_tokens.txt",
        encoder_path="./models/Kroko-Streaming-ASR-Python/en_encoder.onnx",
        decoder_path="./models/Kroko-Streaming-ASR-Python/en_decoder.onnx",
        joiner_path="./models/Kroko-Streaming-ASR-Python/en_joiner.onnx",
        num_threads=5,
        decoding_method="modified_beam_search",
        debug=False,
        provider="cpu",
        device=0,
    )
    
    # Initialize TTS
    tts_manager = TTSManager(
        model_path="./models/Kroko-82M/kokoro-v1.0.onnx",
        voices_path="./models/Kroko-82M/voices-v1.0.bin"
    )
    
    # Initialize connection manager
    manager = ConnectionManager()

async def generate_and_send_response(
    websocket: WebSocket,
    prompt: str,
    voice: str,
    tts_mgr: TTSManager,
    conn_mgr: ConnectionManager,
    speech_end_time: float,
    asr_done_time: float,
    conversation_history: list
):
    """
    Asynchronously triggers response generation from Gemini, splits the text stream 
    into sentences, runs speech generation (TTS) on each sentence, and streams audio to client.
    """
    ai_response_text = ""
    ai_response_appended = False
    try:
        await conn_mgr.send_message(websocket, "audio_start", "Audio incoming...")
        
        # Start Gemini API call asynchronously with streaming
        response = await conversation_gen_model.generate_content_async(prompt, stream=True)
        
        text_buffer = ""
        loop = asyncio.get_running_loop()
        
        first_token_time = None
        first_audio_sent_time = None
        
        async for chunk in response:
            try:
                if chunk.text:
                    if first_token_time is None:
                        first_token_time = time.time()
                        
                    text_buffer += chunk.text
                    ai_response_text += chunk.text
                    sentences, text_buffer = extract_sentences(text_buffer)
                    for sentence in sentences:
                        logger.info(f"Generated sentence: {sentence}")
                        await conn_mgr.send_message(websocket, "llm_response_chunk", sentence)
                        
                        # Generate audio for the sentence in the thread pool using TTSManager
                        tts_results = await loop.run_in_executor(
                            None, tts_mgr.generate_speech, sentence, voice
                        )
                        for gs, ps, audio in tts_results:
                            await asyncio.sleep(0)
                            buf = io.BytesIO()
                            sf.write(buf, audio, 24000, format='WAV')
                            buf.seek(0)
                            
                            if first_audio_sent_time is None:
                                first_audio_sent_time = time.time()
                                asr_ms = int((asr_done_time - speech_end_time) * 1000)
                                llm_ms = int((first_token_time - asr_done_time) * 1000)
                                tts_ms = int((first_audio_sent_time - first_token_time) * 1000)
                                total_ms = int((first_audio_sent_time - speech_end_time) * 1000)
                                
                                metrics = {
                                    "asr_ms": asr_ms,
                                    "llm_ms": llm_ms,
                                    "tts_ms": tts_ms,
                                    "total_ms": total_ms
                                }
                                await conn_mgr.send_message(websocket, "latency_metrics", json.dumps(metrics))
                            
                            await websocket.send_bytes(buf.read())
            except Exception as inner_e:
                logger.error(f"Error processing text chunk: {inner_e}")
                
        # Send remaining text in buffer
        if text_buffer.strip():
            sentence = text_buffer.strip()
            logger.info(f"Generated final sentence: {sentence}")
            await conn_mgr.send_message(websocket, "llm_response_chunk", sentence)
            
            tts_results = await loop.run_in_executor(
                None, tts_mgr.generate_speech, sentence, voice
            )
            for gs, ps, audio in tts_results:
                await asyncio.sleep(0)
                buf = io.BytesIO()
                sf.write(buf, audio, 24000, format='WAV')
                buf.seek(0)
                
                if first_audio_sent_time is None:
                    first_audio_sent_time = time.time()
                    if first_token_time is None:
                        first_token_time = first_audio_sent_time
                    asr_ms = int((asr_done_time - speech_end_time) * 1000)
                    llm_ms = int((first_token_time - asr_done_time) * 1000)
                    tts_ms = int((first_audio_sent_time - first_token_time) * 1000)
                    total_ms = int((first_audio_sent_time - speech_end_time) * 1000)
                    
                    metrics = {
                        "asr_ms": asr_ms,
                        "llm_ms": llm_ms,
                        "tts_ms": tts_ms,
                        "total_ms": total_ms
                    }
                    await conn_mgr.send_message(websocket, "latency_metrics", json.dumps(metrics))
                    
                await websocket.send_bytes(buf.read())
                
        await conn_mgr.send_message(websocket, "audio_end", "Done playing audio")
        
        # Append completed AI response to conversation history
        if not ai_response_appended and ai_response_text.strip():
            conversation_history.append({"role": "ai", "text": ai_response_text.strip()})
            ai_response_appended = True
            
    except asyncio.CancelledError:
        logger.info("AI response streaming task received cancellation signal.")
        await conn_mgr.send_message(websocket, "audio_stop", "Playback stopped.")
        # Append partial AI response to conversation history on user interruption
        if not ai_response_appended and ai_response_text.strip():
            conversation_history.append({"role": "ai", "text": ai_response_text.strip()})
            ai_response_appended = True
    except Exception as e:
        logger.error(f"Error in generate_and_send_response: {e}")
        await conn_mgr.send_message(websocket, "error", f"Response error: {str(e)}")

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("WebSocket connected")

    stream_state = None
    user_is_speaking = False
    silence_chunk_count = 0
    speaking_chunk_count = 0
    
    # Store conversation history (last 10 messages)
    conversation_history = []
    
    # Default configs
    current_voice = "af_heart"
    max_allowed_silence = 8    # ~1 sec at 128ms per chunk
    min_allowed_speaking = 3   # ~380ms at 128ms per chunk
    chunk_duration_ms = 128.0  # assumed 2048 samples at 16kHz
    audio_energy_speaking_threshold = 0.030
    
    ai_response_task = None
    ai_response_start_time = 0.0

    POST_SPEECH_BUFFER_CHUNKS = 10
    post_speech_buffer = deque(maxlen=POST_SPEECH_BUFFER_CHUNKS)

    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                raw_audio = message["bytes"]
                post_speech_buffer.append(raw_audio)
                
                audio_np = np.frombuffer(raw_audio, dtype=np.int16)
                resampled_audio = audio_np
                
                # VAD framing (30ms frames)
                VAD_FRAME_SIZE_MS = 30
                vad_frame_size_samples = int(VAD_SAMPLE_RATE * VAD_FRAME_SIZE_MS / 1000)

                webrtcvad_speech_detected = False
                speech_frames = 0
                total_frames = 0
                for i in range(0, len(resampled_audio) - vad_frame_size_samples + 1, vad_frame_size_samples):
                    vad_frame = resampled_audio[i:i + vad_frame_size_samples]
                    total_frames += 1
                    if is_speech(vad_frame.tobytes(), VAD_SAMPLE_RATE):
                        speech_frames += 1
                
                if total_frames > 0 and (speech_frames / total_frames) >= 0.5:
                    webrtcvad_speech_detected = True

                audio_energy = get_normalized_audio_energy(raw_audio)
                speech_detected = webrtcvad_speech_detected or (audio_energy > audio_energy_speaking_threshold)
                
                logger.info(
                    f"VAD: {webrtcvad_speech_detected} | "
                    f"Energy: {audio_energy:.4f} (threshold: {audio_energy_speaking_threshold}) | "
                    f"Speech Detected: {speech_detected}"
                )

                if speech_detected:
                    user_is_speaking = True
                    silence_chunk_count = 0
                    speaking_chunk_count += 1
                    
                    if stream_state is None:
                        stream_state = asr_manager.create_stream()
                    
                    interim_text = asr_manager.process_chunk(stream_state, raw_audio)
                    if interim_text:
                        await manager.send_message(websocket, "interim_transcription", interim_text)

                else:
                    if user_is_speaking:
                        silence_chunk_count += 1
                        if silence_chunk_count > max_allowed_silence:
                            user_is_speaking = False
                            speech_end_time = time.time()
                            
                            if speaking_chunk_count < min_allowed_speaking:
                                logger.info(f"Utterance too short ({speaking_chunk_count} chunks). Discarding.")
                                speaking_chunk_count = 0
                                stream_state = None
                                await manager.send_message(websocket, "status", "Listening... Speak more...")
                                continue

                            speaking_chunk_count = 0
                            if stream_state:
                                # Finalize transcription using ASR manager
                                final_transcription = asr_manager.finalize_stream(
                                    stream_state, post_speech_buffer
                                )
                                post_speech_buffer.clear()
                                asr_done_time = time.time()
                                await manager.send_message(websocket, "transcription", final_transcription)

                                if final_transcription:
                                    # Append user query to history
                                    conversation_history.append({"role": "user", "text": final_transcription})
                                    
                                    # Limit history size to the last 10 messages (approx 5 conversation turns)
                                    while len(conversation_history) > 10:
                                        conversation_history.pop(0)
                                        
                                    # Construct prompt with historical context
                                    prompt = ""
                                    for msg in conversation_history:
                                        role_label = "User" if msg["role"] == "user" else "AI"
                                        prompt += f"{role_label}: {msg['text']}\n"
                                    prompt += "AI:"
                                    
                                    if ai_response_task and not ai_response_task.done():
                                        ai_response_task.cancel()
                                    
                                    ai_response_task = asyncio.create_task(
                                        generate_and_send_response(
                                            websocket,
                                            prompt,
                                            current_voice,
                                            tts_manager,
                                            manager,
                                            speech_end_time,
                                            asr_done_time,
                                            conversation_history
                                        )
                                    )
                                    ai_response_start_time = time.time()
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
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)