# # import torch
# # import numpy as np

# # # print('Tensor',torch.from_numpy(np.array([1, 2, 3])))

# # from transformers import pipeline


# # from pydub import AudioSegment
# # import soundfile as sf
# # audio = sf.read("output.wav")
# # # audio = AudioSegment.from_wav("output.wav")
# # # logger.info(f"ASR model loaded on device: {device}")
# # print('Audio',audio)

# # asr_pipeline = pipeline(
# #     "automatic-speech-recognition",
# #     model="facebook/wav2vec2-base-960h",  # You can change this to other models
# #     device='cpu',
# #     # padding=True
# # )
# # results = asr_pipeline(audio[0])
# # # except Exception as e:
# # # logger.error(f"Failed to load ASR model: {e}")

# # print(results)


# ###################################################

# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# #  from datasets import load_dataset
# import torch

# import soundfile as sf
# # audio = sf.read("output.wav",samplerate=16000)
# # print(audio)
# # lower sample rate 
# import librosa    
# audio = librosa.load('en1.mp3', sr=16000)
# print(audio)
# # playaudio
# import sounddevice as sd
# sd.play(audio[0], audio[1])

# # load model and tokenizer
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# # retrieve logits
# input_values = processor(audio[0], return_tensors="pt", padding=True,sampling_rate=audio[1],).input_values  # Batch size 1

# logits = model(input_values).logits

# # take argmax and decode
# predicted_ids = torch.argmax(logits, dim=-1)
# transcription = processor.batch_decode(predicted_ids)
# print('Transcription',transcription)

# import vader
# import soundfile as sf

# audio = sf.read("output.wav")
# print(audio)
# segments = vader.vad("output.wav")

# print('Segment',segments)

# from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
# model = load_silero_vad()
# wav = read_audio(r'C:\Users\akliv\OneDrive\Desktop\Akesh kumar\forks\Audio2Audio\recording.wav')
# # speech_timestamps = get_speech_timestamps(
# #   wav,
# #   model,
# #   return_seconds=True,  # Return speech timestamps in seconds (default is samples)
# # )

# from pyaudio import PyAudio, paInt16
# import numpy as np
# p = PyAudio()
# # print(p.get_default_input_device_info())
# stream = p.open(
#     format=paInt16,
#     channels=1,
#     rate=16000,
#     input=True,
#     frames_per_buffer=1624,
# )
# while True:
#     data = stream.read(1624)
#     # print(data),
#     sample_rate, audio_chunk = (16000, np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
#     result = get_speech_timestamps(audio_chunk,model)
#     print(result)
# audio_tensor / torch.max(torch.abs(audio_tensor))

import sounddevice as sd
import numpy as np
import torch
from collections import deque

# Load Silero VAD
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
(get_speech_timestamps, _, _, _, _) = utils

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.032  # seconds
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
ROLLING_WINDOW_DURATION = 0.5  # seconds
ROLLING_WINDOW_SIZE = int(SAMPLE_RATE * ROLLING_WINDOW_DURATION)

rolling_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)

def audio_callback(indata, frames, time, status):
    rolling_buffer.extend(indata[:, 0])  # mono audio

# Start microphone stream
with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE):
    print("Listening (Press Ctrl+C to stop)...")
    while True:
        if len(rolling_buffer) == ROLLING_WINDOW_SIZE:
            audio_np = np.array(rolling_buffer, dtype=np.float32)
            audio_tensor = torch.from_numpy(audio_np)

            speech_segments = get_speech_timestamps(audio_tensor, model, sampling_rate=SAMPLE_RATE)
            if speech_segments:
                print("🔊 Speech detected")
            else:
                print("🛑 Silence")
        sd.sleep(int(CHUNK_DURATION * 1000))  # wait for next chunk
