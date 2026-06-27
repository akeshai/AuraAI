import os
import glob
import ctypes
import logging

logger = logging.getLogger(__name__)

def preload_cuda_libraries():
    """
    Finds and preloads all CUDA/cuDNN shared libraries (.so) installed via nvidia-pip packages.
    Resolves 'Failed to load shared library' errors in ONNX Runtime when running on GPU.
    """
    try:
        import nvidia
        base_dir = os.path.dirname(nvidia.__file__)
        so_files = glob.glob(os.path.join(base_dir, "**", "lib", "*.so*"), recursive=True)
        
        if not so_files:
            return

        loaded = set()
        # Make multiple passes to resolve dependency load order
        for _ in range(3):
            for so_file in so_files:
                if so_file not in loaded:
                    try:
                        ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
                        loaded.add(so_file)
                    except Exception:
                        pass
        
        logger.info(f"Preloaded {len(loaded)} NVIDIA dynamic libraries for CUDA execution.")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error while preloading NVIDIA libraries: {e}")
