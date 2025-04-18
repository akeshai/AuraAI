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