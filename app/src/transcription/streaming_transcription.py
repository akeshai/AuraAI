from sherpa_onnx import OnlineRecognizer
# from io import BufferedWriter
import numpy as np
import torchaudio
import torch

def transcribe_microphone_stream(
    recognizer: OnlineRecognizer, audio_chunk: tuple(int, np.ndarray), stream_state
):
    """
    This function transcribes audio chunks from a microphone stream using the provided recognizer.
    params:
        recognizer: OnlineRecognizer
        audio_chunk: tuple(int, np.ndarray) tuple of sample_rate and waveform_np
        stream_state: object
    """
    try:
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

    except Exception as e:
        print(f"Stream error: {e}")
        return str(e), stream_state
