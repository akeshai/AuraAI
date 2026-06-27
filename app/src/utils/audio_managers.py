import numpy as np

def bytes_to_audio(buffer: bytes, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    """
    Convert raw PCM bytes to a numpy array of float32 normalized in [-1, 1].
    
    Args:
        buffer (bytes): Raw audio byte stream.
        sample_rate (int): Audio sample rate (default 16000).
        
    Returns:
        tuple[np.ndarray, int]: Float32 audio waveform and the sample rate.
    """
    # Convert bytes to int16 numpy array
    audio_int16 = np.frombuffer(buffer, dtype=np.int16)

    # Convert to float32 in range [-1, 1]
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    return audio_float32, sample_rate

def get_normalized_audio_energy(chunk: bytes, dtype=np.int16) -> float:
    """
    Calculate the normalized Root Mean Square (RMS) energy of an audio chunk.
    
    Args:
        chunk (bytes): Raw audio bytes (typically 16-bit PCM).
        dtype: Numpy dtype of the input (default: int16).
        
    Returns:
        float: Normalized energy (0.0 to 1.0).
    """
    try:
        # Convert raw bytes to numpy array
        audio_np = np.frombuffer(chunk, dtype=dtype)

        if len(audio_np) == 0:
            return 0.0

        # Normalize to [-1, 1] range
        max_val = np.iinfo(dtype).max
        audio_float = audio_np.astype(np.float32) / max_val

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_float**2))

        # Return as normalized value (0.0 to 1.0)
        return float(rms)

    except Exception as e:
        # Silent failure to prevent console log flooding
        return 0.0