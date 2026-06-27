import os
import logging
from kokoro_onnx import Kokoro

logger = logging.getLogger(__name__)

class TTSManager:
    """
    Manages the Text-to-Speech (TTS) pipeline using Kokoro ONNX.
    Encapsulates Kokoro ONNX model loading and speech generation.
    """
    def __init__(
        self,
        config_path: str = None,  # Kept for backward compatibility but unused
        model_path: str = "./models/Kroko-82M/kokoro-v1.0.onnx",
        voices_path: str = "./models/Kroko-82M/voices-v1.0.bin",
        device: str = None        # Kept for backward compatibility but unused
    ):
        logger.info(f"Initializing TTSManager with ONNX model from {model_path}...")
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX Model file not found at: {model_path}")
            if not os.path.exists(voices_path):
                raise FileNotFoundError(f"Voices file not found at: {voices_path}")
                
            import onnxruntime as ort
            # Explicitly configure providers to enable GPU (CUDA) execution
            session = ort.InferenceSession(
                model_path, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.kokoro = Kokoro.from_session(session, voices_path)
            # Log active session providers to verify GPU execution status
            active_providers = self.kokoro.sess.get_providers()
            logger.info(f"TTSManager ONNX loaded successfully. Active providers: {active_providers}")
        except Exception as e:
            logger.error(f"Failed to load TTS ONNX model: {e}")
            raise

    def generate_speech(self, text: str, voice: str) -> list:
        """
        Generates speech audio for a given sentence.
        Returns a list of tuples: [(None, None, audio_array)] to match format expected by main.py.
        """
        # Determine language code based on voice name prefixes (American vs British)
        lang = "en-us"
        if voice.startswith("b"):
            lang = "en-gb"

        try:
            samples, sample_rate = self.kokoro.create(
                text,
                voice=voice,
                speed=1.0,
                lang=lang
            )
            # Returns a list of tuples containing (graphemes, phonemes, audio)
            # main.py only extracts the audio numpy array from the third element
            return [(None, None, samples)]
        except Exception as e:
            logger.error(f"TTS generation error for text '{text}' with voice '{voice}': {e}")
            return []
