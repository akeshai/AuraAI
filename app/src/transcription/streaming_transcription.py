from sherpa_onnx import OnlineRecognizer
# from io import BufferedWriter
import numpy as np
import torchaudio
import torch

def transcribe_microphone_stream(
    recognizer: OnlineRecognizer, audio_chunk_with_sample_rate: tuple[int, np.ndarray], stream_state=None
):
    """
    This function transcribes audio chunks from a microphone stream using the provided recognizer.
    params:
        recognizer: OnlineRecognizer
        audio_chunk: tuple(int, np.ndarray) tuple of sample_rate and waveform_np
        stream_state: object    
        
    returns:
        current_text: str
        stream_state: object
    """
    # try:
    if audio_chunk_with_sample_rate is None:  # End of stream
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
    waveform_np, sample_rate = audio_chunk_with_sample_rate
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
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
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

    # except Exception as e:
    #     print(f"Stream error: {e}")
        # return str(e), stream_state



if __name__ == "__main__":
    # Initialize the recognizer
    import wave
    recognizer_en = OnlineRecognizer.from_transducer(
        tokens="./models/Kroko-Streaming-ASR-Python/en_tokens.txt",
        encoder="./models/Kroko-Streaming-ASR-Python/en_encoder.onnx",
        decoder="./models/Kroko-Streaming-ASR-Python/en_decoder.onnx",
        joiner="./models/Kroko-Streaming-ASR-Python/en_joiner.onnx",
        num_threads=3,
        decoding_method="modified_beam_search",
        debug=True,
    )
    
    from pyaudio import PyAudio, paInt16
    pya = PyAudio()
    stream = pya.open(format=paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1624)
    stream_state = None
    audio_chunks = []
    
    try:
        while True:
            data = stream.read(1624)
            # Convert to numpy array and normalize
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_chunk = (16000, audio_data.astype(np.float32) / 32768.0)
            
            # Save the raw audio data (before normalization) for WAV file
            audio_chunks.append(audio_data)
            
            # Process with recognizer
            text, stream_state = transcribe_microphone_stream(recognizer_en, audio_chunk, stream_state)
            print(text, stream_state)
            
    except KeyboardInterrupt:
        print("\nStopping recording...")
    
    finally:
        # Clean up
        stream.stop_stream()
        stream.close()
        pya.terminate()
        
        # Save the recorded audio to a WAV file
        if audio_chunks:
            # Combine all chunks
            audio_data = np.concatenate(audio_chunks)
            
            # Save as WAV file
            filename = "recording.wav"
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pya.get_sample_size(paInt16))
                wf.setframerate(16000)
                wf.writeframes(audio_data.tobytes())
            
            print(f"Audio saved to {filename}")