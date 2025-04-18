# import torchaudio
# import soundfile
# from speechbrain.inference.VAD import VAD

# import numpy as np
# import webrtcvad
# import pyaudio
# # import noisereduce

# VAD_LEVEL: int = 3
# vad = webrtcvad.Vad(VAD_LEVEL)

# import torch


# def detect_speech_in_chunk(chunk: bytes, sample_rate: int = 16000, ) -> bool:
#     """
#     Detect if speech is present in an audio chunk using webrtcvad.
    
#     Arguments:
#         chunk (bytes): Raw audio data as bytes
#         sample_rate (int): Sample rate of the audio (default: 16000)
#         vad_level (int): VAD aggressiveness level 0-3 (default: 3)
#     Returns:
#         bool: True if speech detected, False otherwise
#     """
#     # try:
        
#     # resample bytes to 16000 frames if frames greater than 16000
#     if sample_rate != 16000:
#         waveform = torch.from_numpy(np.frombuffer(chunk, dtype=np.int16)).float().unsqueeze(0)
#         resampler = torchaudio.transforms.Resample(
#             orig_freq=sample_rate, new_freq=16000
#         )
#         waveform = resampler(waveform)
#         chunk = waveform.squeeze(0).numpy()
#         sample_rate = 16000
    
#     frame_duration = 10  
#     frame_size = int(sample_rate * frame_duration / 1000)
    
#     # If chunk size doesn't match expected size, return False
#     if len(chunk) != frame_size * 2:  # *2 because of 16-bit audio
#         return False
#     # Detect speech
#     return vad.is_speech(bytes(chunk), sample_rate)
        
#     # except Exception as e:
#     #     print(f"Error detecting speech: {e}")
#     #     return False



# if __name__ == "__main__":
#     sample_rate = 44100 
#     frame_duration = 10  # in milliseconds
#     frame_size = int(sample_rate * frame_duration / 1000)  # Number of samples per frame
#     frame_bytes = frame_size * 2  # 2 bytes per sample (16-bit audio)
#     p = pyaudio.PyAudio()
#     stream = p.open(
#         format=pyaudio.paInt16,
#         channels=1,
#         rate=sample_rate,
#         input=True,
#         frames_per_buffer=frame_size,
#     )
#     vad = webrtcvad.Vad(3)  # Aggressiveness: 0-3 (0 = least aggressive, 3 = most)
#     try:
#         while True:
#             data = stream.read(frame_size, exception_on_overflow=False)
#             is_speech = detect_speech_in_chunk(data, sample_rate=sample_rate)
#             print("Speech detected" if is_speech else "Silence")

#     except KeyboardInterrupt:
#         print("Stopped.")
#     finally:
#         stream.stop_stream()
#         stream.close()
#         p.terminate()










# ######################################







import torchaudio
import soundfile
import webrtcvad
import pyaudio
import numpy as np
import torch

VAD_LEVEL: int = 3
vad = webrtcvad.Vad(VAD_LEVEL)

def detect_speech_in_chunk(chunk: bytes, sample_rate: int = 16000) -> bool:
    """
    Detect if speech is present in an audio chunk using webrtcvad.
    
    Arguments:
        chunk (bytes): Raw audio data as bytes
        sample_rate (int): Sample rate of the audio (default: 16000)
    Returns:
        bool: True if speech detected, False otherwise
    """
    try:
        # Convert bytes to numpy array (16-bit PCM)
        audio_np = np.frombuffer(chunk, dtype=np.int16)
        
        # Convert to float32 for resampling
        audio_tensor = torch.from_numpy(audio_np).float()
        
        # Resample if needed
        if sample_rate != 16000:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=16000
            )
            audio_tensor = resampler(audio_tensor)
            audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension
            sample_rate = 16000
        
        # Convert back to int16 for VAD
        audio_np = audio_tensor.numpy().astype(np.int16)
        
        # Frame parameters for VAD
        frame_duration = 10  # ms
        frame_size = int(sample_rate * frame_duration / 1000)
        
        # Check if we have enough data
        if len(audio_np) < frame_size:
            return False
            
        # Take the first frame (or you could process multiple frames)
        frame = audio_np[:frame_size].tobytes()
        
        return vad.is_speech(frame, sample_rate)
        
    except Exception as e:
        print(f"Error detecting speech: {e}")
        return False

if __name__ == "__main__":
    sample_rate = 44100  # Try with both 44100 and 16000
    frame_duration = 30  # ms - using slightly longer frame for better reliability
    frame_size = int(sample_rate * frame_duration / 1000)
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frame_size,
    )
    
    try:
        while True:
            data = stream.read(frame_size, exception_on_overflow=False)
            is_speech = detect_speech_in_chunk(data, sample_rate=sample_rate)
            print("Speech detected" if is_speech else "Silence")

    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()