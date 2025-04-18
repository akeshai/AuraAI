
import numpy as np

def bytes_to_audio(buffer: bytes, sample_rate: int = 16000) -> (np.ndarray, int):
    """Convert bytes to numpy array of floats 
    returns tuple of (np.ndarray of audio, sample_rate)"""
    # Convert bytes to int16 numpy array
    audio_int16 = np.frombuffer(buffer, dtype=np.int16)

    # Convert to float32 in range [-1, 1]
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    return (audio_float32,
         sample_rate,  # Most models use 16kHz
    )