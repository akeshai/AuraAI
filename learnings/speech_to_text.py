import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from transformers import pipeline
import torch
import numpy as np
from typing import Optional
print('Tensor',torch.from_numpy(np.array([1, 2, 3]))
)
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files (including our index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for ASR pipeline
asr_pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"


async def load_asr_model():
    """Load the ASR model in memory"""
    global asr_pipeline
    logger.info("Loading ASR model...")

    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="facebook/wav2vec2-base-960h",  # You can change this to other models
            device=device,
            # padding=True
        )
        logger.info(f"ASR model loaded on device: {device}")
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load models when application starts"""
    await load_asr_model()


class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, websocket: WebSocket, message_type: str, text: str):
        message = {"type": message_type, "text": text}
        await websocket.send_text(json.dumps(message))


manager = ConnectionManager()


def bytes_to_audio(buffer: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Convert bytes to numpy array of floats"""
    # Convert bytes to int16 numpy array
    audio_int16 = np.frombuffer(buffer, dtype=np.int16)

    # Convert to float32 in range [-1, 1]
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    return {
        "raw": audio_float32,
        "sampling_rate": sample_rate,  # Most models use 16kHz
    }


# async def process_audio_buffer(buffer: list, websocket: WebSocket) -> Optional[str]:
#     """Process accumulated audio buffer and return transcription"""
#     logger.info("Processing audio buffer...")
#     if not buffer:
#         return None

#     try:
#         # Combine all audio chunks
#         audio_data = b"".join(buffer)

#         # Convert to numpy array
#         audio_np = bytes_to_audio(audio_data)
#         # print('Audio shape',audio_np.shape)
#         # Run ASR
#         # write audio data to audio file
#         import soundfile as sf
#         sf.write('output.wav', audio_np['raw'], 16000)

#         result = asr_pipeline(
#             audio_np,
#             chunk_length_s=5,  # Process in 5-second chunks
#             stride_length_s=1,
#         )

#         transcription = result.get("text", "")
#         logger.info(f"Transcription: {transcription}")

#         # Send transcription back to client
#         await manager.send_message(websocket, "transcription", transcription)

#         return transcription

#     except Exception as e:
#         logger.error(f"Error processing audio: {e}")
#         return None

async def process_audio_buffer(buffer: list, websocket: WebSocket) -> Optional[str]:
    """Process accumulated audio buffer and return transcription"""
    logger.info("Processing audio buffer...")
    if not buffer:
        return None

    # try:
        # Combine all audio chunks
    audio_data = b"".join(buffer)

    # Convert to numpy array (assuming 44100 Hz from client)
    audio_np = bytes_to_audio(audio_data, sample_rate=44100)
    # print('Audio shape',audio_np.shape)
    # Run ASR
    # write audio data to audio file
    # import soundfile as sf
    # sf.write('output.wav', audio_np['raw'], 44100)  # Save with the assumed sample rate

    result = asr_pipeline(audio_np,
                        #   chunk_length_s = 5  # Process in 5-second chunks
        # ,stride_length_s = 1  
        )

    transcription = result.get("text", "")
    logger.info(f"Transcription: {transcription}")

    # Send transcription back to client
    await manager.send_message(websocket, "transcription", transcription)

    return transcription

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
        while True:
            # Receive audio data from client
            data = await websocket.receive_bytes()

            # Simple voice activity detection (you can replace with more sophisticated VAD)
            audio_energy = np.frombuffer(data, dtype=np.int16).var()
            # print(audio_energy)
            speaking_now = audio_energy > 120000  # Simple threshold
            # speaking_now = True
            print("Silence frames", silence_frames, "Audio energy", audio_energy)

            if speaking_now:
                is_speaking = True
                silence_frames = 0
                audio_buffer.append(data)
                print("Length of audio buffer", len(audio_buffer), )
                
            elif is_speaking:
                silence_frames += 1
                audio_buffer.append(data)
                # If we've had enough silence frames, process the utterance
                if silence_frames > max_silence_frames:
                    transcription = await process_audio_buffer(audio_buffer, websocket)
                    audio_buffer = []
                    is_speaking = False
                    logger.info(f"Transcription: {transcription}")
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
