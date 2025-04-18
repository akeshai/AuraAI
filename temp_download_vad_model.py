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
print(p.get_default_input_device_info())

# Audio settings
sample_rate = 16000
frame_duration = 10  # in milliseconds
frame_size = int(sample_rate * frame_duration / 1000)  # Number of samples per frame
frame_bytes = frame_size * 2  # 2 bytes per sample (16-bit audio)

# Open audio stream
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=sample_rate,
    input=True,
    frames_per_buffer=frame_size,
)

# Create a VAD object
vad = webrtcvad.Vad(3)  # Aggressiveness: 0-3 (0 = least aggressive, 3 = most)

print("Listening... Press Ctrl+C to stop.")

try:
    while True:
        data = stream.read(frame_size, exception_on_overflow=False)

        # Check if this frame contains speech
        is_speech = vad.is_speech(data, sample_rate)
        print("Speech detected" if is_speech else "Silence")

except KeyboardInterrupt:
    print("Stopped.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
