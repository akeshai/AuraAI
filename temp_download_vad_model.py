import torchaudio
import soundfile
from speechbrain.inference.VAD import VAD

# VAD_model = VAD.from_hparams(source=r"C:\Users\akliv\OneDrive\Desktop\Akesh kumar\forks\Audio2Audio\models\vad-crdnn-libriparty")
# import torch

# # audio_file = r"C:\Users\akliv\OneDrive\Desktop\Akesh kumar\forks\Audio2Audio\data\testing_audios\ocean_noice_voice.m4a"
# audio_file = r"C:\Users\akliv\OneDrive\Desktop\Akesh kumar\forks\Audio2Audio\data\testing_audios\example_vad.wav"
# import numpy as np


# Print the output
# VAD.save_boundaries(boundaries)


# from pyaudio import PyAudio, paInt16

# p = PyAudio()
# print( p.get_default_input_device_info())
# stream = p.open(format=paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024,)
# stream_state = None
# import webrtcvad
# vad = webrtcvad.Vad(2)

# # while True:
# #     data = stream.read(1024)
# #     print(data)
# #     # sample_rate, audio_chunk = (16000, np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
# #     # result = vad.is_speech(audio_chunk,sample_rate)
# #     print(result)


# sample_rate = 16000
# frame_duration = 10  # ms
# frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
# print ('Contains speech: %s' % (vad.is_speech(frame, sample_rate)
# ))

import numpy as np
import webrtcvad
import pyaudio
import noisereduce

# Initialize PyAudio
p = pyaudio.PyAudio()

# Print default input device info (optional)
# print(p.get_default_input_device_info())

# # Audio settings
# sample_rate = 16000
# frame_duration = 10  # in milliseconds
# frame_size = int(sample_rate * frame_duration / 1000)  # Number of samples per frame
# frame_bytes = frame_size * 2  # 2 bytes per sample (16-bit audio)

# # Open audio stream
# stream = p.open(
#     format=pyaudio.paInt16,
#     channels=1,
#     rate=sample_rate,
#     input=True,
#     frames_per_buffer=frame_size,
# )

# # Create a VAD object
# vad = webrtcvad.Vad(3)  # Aggressiveness: 0-3 (0 = least aggressive, 3 = most)

# print("Listening... Press Ctrl+C to stop.")

# try:
#     while True:
#         data = stream.read(frame_size, exception_on_overflow=False)

#         # Check if this frame contains speech
#         is_speech = vad.is_speech(data, sample_rate)
#         print("Speech detected" if is_speech else "Silence")

# except KeyboardInterrupt:
#     print("Stopped.")
# finally:
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

def detect_speech_in_chunk(chunk: bytes, sample_rate: int = 16000, vad_level: int = 3) -> bool:
    """
    Detect if speech is present in an audio chunk using webrtcvad.
    
    Arguments:
        chunk (bytes): Raw audio data as bytes
        sample_rate (int): Sample rate of the audio (default: 16000)
        vad_level (int): VAD aggressiveness level 0-3 (default: 3)
    Returns:
        bool: True if speech detected, False otherwise
    """
    try:
        vad = webrtcvad.Vad(vad_level)
        
        frame_duration = 10  
        frame_size = int(sample_rate * frame_duration / 1000)
        
        # If chunk size doesn't match expected size, return False
        if len(chunk) != frame_size * 2:  # *2 because of 16-bit audio
            return False
            
        # Detect speech
        return vad.is_speech(chunk, sample_rate)
        
    except Exception as e:
        print(f"Error detecting speech: {e}")
        return False



if __name__ == "__main__":
    sample_rate = 44100
    frame_duration = 10  # in milliseconds
    frame_size = int(sample_rate * frame_duration / 1000)  # Number of samples per frame
    frame_bytes = frame_size * 2  # 2 bytes per sample (16-bit audio)
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=frame_size,
    )
    vad = webrtcvad.Vad(3)  # Aggressiveness: 0-3 (0 = least aggressive, 3 = most)
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
