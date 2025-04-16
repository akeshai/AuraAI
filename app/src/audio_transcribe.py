import numpy as np
import gradio as gr
import torchaudio
import torch
from sherpa_onnx import OnlineRecognizer
import time

# Initialize the recognizer
recognizer_en = OnlineRecognizer.from_transducer(
    tokens="./models/Kroko-Streaming-ASR-Python/en_tokens.txt",
    encoder="./models/Kroko-Streaming-ASR-Python/en_encoder.onnx",
    decoder="./models/Kroko-Streaming-ASR-Python/en_decoder.onnx",
    joiner="./models/Kroko-Streaming-ASR-Python/en_joiner.onnx",
    num_threads=1,
    decoding_method="modified_beam_search",
    debug=False
)


def transcribe_audio_online_streaming(file, language):
    """Generator for file transcription"""
    if file is None:
        yield "Please upload an audio file."
        return

    # try:
    match language:
        case "English":
            recognizer = recognizer_en

            
    waveform, sample_rate = torchaudio.load(file.name)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    waveform_np = waveform.numpy()[0]

    # Add 0.5 seconds of silence padding at the beginning and end
    pad_duration = 0.5  # seconds
    pad_samples = int(pad_duration * sample_rate)
    pad_start = np.zeros(pad_samples, dtype=np.float32)
    pad_end = np.zeros(pad_samples, dtype=np.float32)
    waveform_np = np.concatenate([pad_start, waveform_np, pad_end])
    
    total_samples = waveform_np.shape[0]
    
    s = recognizer.create_stream()
    chunk_size = 4000  # 0.25-second chunks
    offset = 0
    
    while offset < total_samples:
        end = offset + chunk_size
        chunk = waveform_np[offset:end]
        s.accept_waveform(sample_rate, chunk.tolist())
        
        while recognizer.is_ready(s):
            recognizer.decode_streams([s])
            
        yield recognizer.get_result(s)
        offset += chunk_size

    # Final processing
    tail_paddings = np.zeros(int(0.66 * sample_rate), dtype=np.float32)
    s.accept_waveform(sample_rate, tail_paddings.tolist())
    s.input_finished()
    
    while recognizer.is_ready(s):
        recognizer.decode_streams([s])
    
    current_text = recognizer.get_result(s)
    if isinstance(current_text, (list, np.ndarray)):
        current_text = " ".join(map(str, current_text))
    elif isinstance(current_text, bytes):
        current_text = current_text.decode("utf-8", errors="ignore")

    yield current_text

    # except Exception as e:
    #     yield f"Error: {e}"

def transcribe_microphone_stream(audio_chunk, stream_state, language):
    """Real-time microphone streaming transcription"""
    try:
        match language:
            case "English":
                recognizer = recognizer_en

        if audio_chunk is None:  # End of stream
            if stream_state is not None:
                # Flush remaining audio
                tail_paddings = np.zeros(int(0.66 * 16000), dtype=np.float32)
                stream_state.accept_waveform(16000, tail_paddings.tolist())
                stream_state.input_finished()
                while recognizer.is_ready(stream_state):
                    recognizer.decode_streams([stream_state])
                final_text = recognizer.get_result(stream_state)
                return final_text, None
            return "", None

        sample_rate, waveform_np = audio_chunk
        if len(waveform_np.shape) > 1:
            waveform_np = waveform_np.mean(axis=1)

        # Normalize if needed
        if waveform_np.dtype != np.float32:
            waveform_np = waveform_np.astype(np.float32)

        if np.max(np.abs(waveform_np)) > 1.0:
            waveform_np = waveform_np / np.max(np.abs(waveform_np))

        # Resample if needed
        if sample_rate != 16000:
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            waveform_np = waveform.squeeze(0).numpy()
            sample_rate = 16000

        waveform_np = np.clip(waveform_np, -1.0, 1.0)

        # Initialize stream if first chunk
        if stream_state is None:
            stream_state = recognizer.create_stream()

        # Process audio chunk
        stream_state.accept_waveform(sample_rate, waveform_np.tolist())
        
        # Decode available frames
        while recognizer.is_ready(stream_state):
            recognizer.decode_streams([stream_state])
        
        current_text = recognizer.get_result(stream_state)

        if isinstance(current_text, (list, np.ndarray)):
            current_text = " ".join(map(str, current_text))
        elif isinstance(current_text, bytes):
            current_text = current_text.decode("utf-8", errors="ignore")
        
        return current_text, stream_state

    except Exception as e:
        print(f"Stream error: {e}")
        return str(e), stream_state

for text in transcribe_audio_online_streaming(open(r'C:\Users\akliv\OneDrive\Desktop\Akesh kumar\forks\Audio2Audio\data\en1.mp3'),"English"):
    print(text)