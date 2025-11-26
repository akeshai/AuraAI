import webrtcvad

# Assume audio passed to this function is ALWAYS 16kHz, 16-bit PCM
# The resampling should happen *before* calling this.
vad = webrtcvad.Vad(3) # Set aggressiveness level

def is_speech(chunk: bytes, sample_rate: int) -> bool:
    """
    Checks for speech in a raw audio chunk.
    NOTE: `webrtcvad` requires sample rates of 8000, 16000, 32000, or 48000
    and frame durations of 10, 20, or 30 ms.
    """
    # This function should be simple. The complexity of resampling and buffering
    # belongs in the main application logic.
    return vad.is_speech(chunk, sample_rate)