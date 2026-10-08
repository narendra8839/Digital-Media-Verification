"""Device selection and CUDA verification utilities."""

import os
import logging
import torch

logger = logging.getLogger("dmv_api.device")


def get_optimal_device() -> str:
    """Safely detect and return the optimal compute device ('cuda:0' or 'cpu').

    Order of precedence:
    1. DMV_DEVICE environment variable (e.g. 'cuda:0', 'cpu', 'cuda:1').
    2. 'cuda:0' if torch.cuda.is_available() and tensor allocation succeeds.
    3. Fallback to 'cpu' if CUDA is unavailable or initialization fails.

    Returns:
        String device identifier ('cuda:0' or 'cpu').
    """
    env_device = os.environ.get("DMV_DEVICE", "").strip()
    if env_device:
        logger.info(f"Using explicitly configured device via DMV_DEVICE: '{env_device}'")
        return env_device

    if torch.cuda.is_available():
        try:
            device_str = "cuda:0"
            # Test actual memory allocation on cuda:0 to verify runtime stability
            test_tensor = torch.zeros(1, device=device_str)
            del test_tensor
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA accelerator detected and verified: {device_name} -> Using '{device_str}'")
            return device_str
        except Exception as exc:
            logger.warning(
                f"torch.cuda.is_available() is True, but test tensor allocation failed ({exc}). "
                "Falling back safely to 'cpu'."
            )
            return "cpu"

    logger.info("CUDA accelerator not detected or not supported in this runtime. Using 'cpu'.")
    return "cpu"
