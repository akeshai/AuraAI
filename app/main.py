import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# from transformers import pipeline
import torch
import numpy as np
from typing import Optional
from sherpa_onnx import OnlineRecognizer
import time
from src.configs.connections import ConnectionManager

from src.utils import bytes_to_audio

from src.transcription import transcribe_microphone_stream
from src.transcription import detect_speech_in_chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files (including our index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for ASR pipeline
recognizer_en: OnlineRecognizer = None

device = "cuda" if torch.cuda.is_available() else "cpu"


async def load_asr_model():
    """Load the ASR model in memory"""
    global recognizer_en
    logger.info("Loading ASR model...")

    try:
        # Initialize the recognizer
        recognizer_en = OnlineRecognizer.from_transducer(
            tokens="./models/Kroko-Streaming-ASR-Python/en_tokens.txt",
            encoder="./models/Kroko-Streaming-ASR-Python/en_encoder.onnx",
            decoder="./models/Kroko-Streaming-ASR-Python/en_decoder.onnx",
            joiner="./models/Kroko-Streaming-ASR-Python/en_joiner.onnx",
            num_threads=1,
            decoding_method="modified_beam_search",
            debug=False,
        )
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load models when application starts"""
    await load_asr_model()
    global manager
    manager = ConnectionManager()


async def process_audio_buffer(
    buffer: list[bytes], stream_state, websocket: WebSocket
) -> Optional[str]:
    """Process accumulated audio buffer and return transcription"""
    logger.info("Processing audio buffer...")
    if not buffer:
        return None

    audio_data = b"".join(buffer)

    # Convert to numpy array (assuming 44100 Hz from client)
    audio_np, sample_rate = bytes_to_audio(audio_data, sample_rate=44100)
    # print('Audio shape',audio_np.shape)
    # Run ASR
    # write audio data to audio file
    import soundfile as sf

    sf.write("output.wav", audio_np, sample_rate)  # Save with the assumed sample rate

    current_text, stream_state = transcribe_microphone_stream(
        recognizer=recognizer_en,
        audio_chunk_with_sample_rate=(audio_np, sample_rate),
        stream_state=stream_state,
        #   chunk_length_s = 5  # Process in 5-second chunks
        # ,stride_length_s = 1
    )

    print("Transcription:", current_text)

    # Send transcription back to client
    await manager.send_message(websocket, "transcription", current_text)

    return stream_state

    # except Exception as e:
    #     logger.error(f"Error processing audio: {e}")
    #     return None


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("WebSocket connected")

    # Audio processing state
    audio_buffer = []
    is_speaking = False
    silence_frames = 0
    max_silence_frames = 5  # Adjust based on your VAD needs
    logger.info("Processing audio...")
    try:
        stream_state = None
        while True:
            data = await websocket.receive_bytes()

            audio_energy = np.frombuffer(data, dtype=np.int16).var()
            speaking_now = detect_speech_in_chunk(data, sample_rate=44100)
            print(
                "Silence frames:",
                silence_frames,
                "Audio energy: ",
                audio_energy,
                "Speaking now:",
                speaking_now,
            )

            if speaking_now:
                is_speaking = True
                silence_frames = 0
                audio_buffer.append(data)
                print(
                    "Length of audio buffer",
                    len(audio_buffer),
                )

            elif is_speaking:
                silence_frames += 1
                audio_buffer.append(data)
                # If we've had enough silence frames, process the utterance
                if silence_frames > max_silence_frames:
                    stream_state = await process_audio_buffer(
                        audio_buffer,
                        stream_state,
                        websocket,
                    )
                    audio_buffer = []
                    is_speaking = False
                    logger.info(f"Transcription: {stream_state}")
                    # Removed redundant sending of transcription here
            else:
                # Not speaking, discard the audio
                logger.info("Not speaking, discarding audio")
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    # except Exception as e:
    #     manager.disconnect(websocket)
    #     logger.error(f"WebSocket error: {e}")


@app.get("/")
async def get():
    """Serve the HTML interface"""
    with open(r"static/index_speech_to_text.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run("speech_to_text:app", host="0.0.0.0", port=8000, reload=True)
