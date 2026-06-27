import logging
import numpy as np
from sherpa_onnx import OnlineRecognizer
from src.utils.audio_managers import bytes_to_audio

logger = logging.getLogger(__name__)

class StreamingASR:
    """
    Manages the online streaming Automatic Speech Recognition (ASR) model.
    Encapsulates sherpa-onnx OnlineRecognizer loading, stream lifecycle, and decoding.
    """
    def __init__(
        self,
        tokens_path: str = "./models/Kroko-Streaming-ASR-Python/en_tokens.txt",
        encoder_path: str = "./models/Kroko-Streaming-ASR-Python/en_encoder.onnx",
        decoder_path: str = "./models/Kroko-Streaming-ASR-Python/en_decoder.onnx",
        joiner_path: str = "./models/Kroko-Streaming-ASR-Python/en_joiner.onnx",
        num_threads: int = 5,
        decoding_method: str = "modified_beam_search",
        debug: bool = False,
        provider: str = "cpu",
        device: int = 0
    ):
        logger.info("Initializing StreamingASR model...")
        try:
            self.recognizer = OnlineRecognizer.from_transducer(
                tokens=tokens_path,
                encoder=encoder_path,
                decoder=decoder_path,
                joiner=joiner_path,
                num_threads=num_threads,
                decoding_method=decoding_method,
                debug=debug,
                provider=provider,
                device=device,
            )
            logger.info(f"StreamingASR model loaded successfully using provider '{provider}'.")
        except Exception as e:
            logger.error(f"Failed to load ASR models: {e}")
            raise

    def create_stream(self):
        """
        Creates a new transcription stream.
        """
        return self.recognizer.create_stream()

    def process_chunk(self, stream_state, raw_audio: bytes) -> str:
        """
        Processes a raw audio chunk, decodes it, and returns the interim/current transcription text.
        """
        audio_np_asr, _ = bytes_to_audio(raw_audio, sample_rate=16000)
        stream_state.accept_waveform(16000, audio_np_asr.tolist())
        
        while self.recognizer.is_ready(stream_state):
            self.recognizer.decode_streams([stream_state])
            
        current_text = self.recognizer.get_result(stream_state)
        return self._clean_transcription(current_text)

    def finalize_stream(self, stream_state, post_speech_buffer) -> str:
        """
        Feeds the trailing post-speech buffer chunks, triggers ASR end-of-input,
        runs final decoding, and returns the final transcription.
        """
        # Feed the post-speech buffer to ASR
        for buffered_chunk in post_speech_buffer:
            audio_np_asr, _ = bytes_to_audio(buffered_chunk, sample_rate=16000)
            stream_state.accept_waveform(16000, audio_np_asr.tolist())
            
        stream_state.input_finished()
        
        while self.recognizer.is_ready(stream_state):
            self.recognizer.decode_streams([stream_state])
            
        final_text = self.recognizer.get_result(stream_state)
        return self._clean_transcription(final_text)

    def _clean_transcription(self, text) -> str:
        """Helper to ensure transcription returns a clean string."""
        if not text:
            return ""
        if isinstance(text, (list, np.ndarray)):
            text = " ".join(map(str, text))
        elif isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        return text.strip()