import numpy as np
from scipy import signal

import numpy as np
import time
def bytes_to_audio(buffer: bytes, sample_rate: int = 16000) -> (np.ndarray, int):
    """Convert bytes to numpy array of floats 
    returns tuple of (np.ndarray of audio, sample_rate)"""
    # Convert bytes to int16 numpy array
    audio_int16 = np.frombuffer(buffer, dtype=np.int16)

    # Convert to float32 in range [-1, 1]
    audio_float32 = audio_int16.astype(np.float32) / 32768.0

    return (audio_float32,
         sample_rate,  # Most models use 16kHz
    )
    
import numpy as np

def get_normalized_audio_energy(chunk: bytes, dtype=np.int16) -> float:
    """
    Calculate the normalized root mean square (RMS) energy of an audio chunk.
    
    Args:
        chunk (bytes): Raw audio bytes (typically 16-bit PCM)
        dtype: Numpy dtype of the input (default: int16)
        
    Returns:
        float: Normalized energy (0.0 to 1.0)
    """
    try:
        # Convert raw bytes to numpy array
        audio_np = np.frombuffer(chunk, dtype=dtype)

        if len(audio_np) == 0:
            return 0.0

        # Normalize to [-1, 1] range
        max_val = np.iinfo(dtype).max
        audio_float = audio_np.astype(np.float32) / max_val

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_float**2))

        # Return as normalized value (0.0 to 1.0)
        return float(rms)

    except Exception as e:
        print(f"Error computing energy: {e}")
        return 0.0

if __name__ == "__main__":
     import pandas as pd
     import time
     import numpy as np
     import matplotlib.pyplot as plt
     import seaborn as sns
     import keyboard  # You may need to install with: pip install keyboard
     from pyaudio import PyAudio, paInt16

     # Mock of your get_normalized_audio_energy function
   
     # Audio setup
     pya = PyAudio()
     print(pya.get_default_input_device_info())
    #  exit()
    #  pya.get_device_info_by_index()
    
     stream = pya.open(format=paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1624,input_device_index=3)

     # Tracking variables
     records = []
     labels = {'s': 'speaking', 'n': 'noise', 'd': 'silence'}
     current_label = 'unlabeled'
     start_time = time.time()

     print("Press 's' for speaking, 'n' for noise, 'd' for silence. Ctrl+C to stop.")

     try:
          while True:
               # Read audio chunk
               data = stream.read(1624)
               energy = get_normalized_audio_energy(data)

               # Check key press without blocking
               for key in labels:
                    if keyboard.is_pressed(key):
                         current_label = labels[key]

               # Append energy and label
               records.append({
                    'timestamp': time.time() - start_time,
                    'energy': energy,
                    'label': current_label
               })

               print(f"Duration: {time.time() - start_time:.2f}s | Energy: {energy:.4f} | Label: {current_label}", end="\r")

     except KeyboardInterrupt:
          print("\nStopping recording...")

     # Save to CSV
     df = pd.DataFrame(records)
     filename_base = f"data/experiments/energies/energy{np.random.randint(100)}{np.random.randint(100)}"
     df.to_csv(f"{filename_base}.csv", index=False)

     # Save energy plot
     sns.lineplot(data=df, x='timestamp', y='energy',hue='label')
     plt.title('Audio Energy Over Time')
     plt.xlabel('Frame')
     plt.ylabel('Energy')
     plt.savefig(f"{filename_base}.png")