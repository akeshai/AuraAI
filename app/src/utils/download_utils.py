import os
import logging
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))

# Models config
ASR_DIR = os.path.join(MODELS_DIR, "Kroko-Streaming-ASR-Python")
ASR_FILES = {
    "en_tokens.txt": "https://huggingface.co/spaces/Banafo/Kroko-Streaming-ASR-Python/resolve/main/en_tokens.txt",
    "en_encoder.onnx": "https://huggingface.co/spaces/Banafo/Kroko-Streaming-ASR-Python/resolve/main/en_encoder.onnx",
    "en_decoder.onnx": "https://huggingface.co/spaces/Banafo/Kroko-Streaming-ASR-Python/resolve/main/en_decoder.onnx",
    "en_joiner.onnx": "https://huggingface.co/spaces/Banafo/Kroko-Streaming-ASR-Python/resolve/main/en_joiner.onnx"
}

TTS_DIR = os.path.join(MODELS_DIR, "Kroko-82M")
TTS_FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
}

def download_file(url: str, dest_path: str):
    """
    Downloads a file with a tqdm progress bar in the terminal.
    """
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, "wb") as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            size = f.write(data)
            bar.update(size)

def ensure_models_exist():
    """
    Checks for required model files and downloads them if they do not exist.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Check ASR files
    print("\n--- Verifying ASR Model Files ---")
    for filename, url in ASR_FILES.items():
        dest = os.path.join(ASR_DIR, filename)
        if not os.path.exists(dest):
            logger.info(f"ASR file {filename} missing. Starting download...")
            try:
                download_file(url, dest)
            except Exception as e:
                logger.error(f"Failed to download ASR file {filename}: {e}")
                raise RuntimeError(f"ASR model download failed: {e}")
        else:
            print(f"ASR file '{filename}' already exists. Skipping.")
            
    # 2. Check TTS files
    print("\n--- Verifying TTS Model Files ---")
    for filename, url in TTS_FILES.items():
        dest = os.path.join(TTS_DIR, filename)
        if not os.path.exists(dest):
            logger.info(f"TTS file {filename} missing. Starting download...")
            try:
                download_file(url, dest)
            except Exception as e:
                logger.error(f"Failed to download TTS file {filename}: {e}")
                raise RuntimeError(f"TTS model download failed: {e}")
        else:
            print(f"TTS file '{filename}' already exists. Skipping.")
            
    print("\nAll required model files are present and verified!\n")
