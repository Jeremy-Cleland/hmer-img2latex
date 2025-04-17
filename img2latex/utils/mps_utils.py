"""
Utilities for working with Apple Metal Performance Shaders (MPS) on macOS.
"""

import platform
import time
from typing import Optional, Tuple, Union

import torch

from img2latex.utils.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def is_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    Returns:
        bool: True if MPS is available, False otherwise
    """
    if platform.system() != "Darwin":
        return False

    try:
        if not torch.backends.mps.is_available():
            return False
        if not torch.backends.mps.is_built():
            return False
        # Actually try to create a tensor to confirm MPS works
        torch.zeros(1).to(torch.device("mps"))
        return True
    except (AttributeError, AssertionError, RuntimeError):
        return False


def get_mps_device() -> Optional[torch.device]:
    """
    Get MPS device if available.

    Returns:
        Optional[torch.device]: MPS device if available, None otherwise
    """
    if is_mps_available():
        return torch.device("mps")
    return None


def set_device(device_name: str = None) -> torch.device:
    """
    Set the device to use for training based on availability.

    Args:
        device_name: Device to use (mps, cuda, cpu). If None, it will try to use MPS if available,
                    or fall back to CPU.

    Returns:
        torch.device: The selected device
    """
    if device_name == "mps" and is_mps_available():
        device = torch.device("mps")
        logger.info("Using MPS (Metal Performance Shaders) device")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        if device_name in ("mps", "cuda") and device_name != "cpu":
            logger.warning(
                f"Requested device '{device_name}' is not available, falling back to CPU"
            )
        device = torch.device("cpu")
        logger.info("Using CPU device")

    return device


def get_device_info(device: torch.device) -> Tuple[str, Union[str, None]]:
    """
    Get information about the device.

    Args:
        device: PyTorch device

    Returns:
        Tuple[str, Union[str, None]]: Device type and name if available
    """
    device_name = None

    if device.type == "mps":
        device_type = "MPS (Metal Performance Shaders)"
        # No direct way to get the GPU name in PyTorch for MPS
        try:
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
            )
            output = result.stdout
            for line in output.split("\n"):
                if "Chipset Model" in line:
                    device_name = line.split(":")[1].strip()
                    break
        except Exception:
            pass
    elif device.type == "cuda":
        device_type = "CUDA"
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(
                device.index if device.index else 0
            )
    else:
        device_type = "CPU"
        try:
            import platform

            device_name = platform.processor()
        except Exception:
            pass

    return device_type, device_name


def synchronize() -> None:
    """
    Wait for all MPS or CUDA operations to complete.
    Useful before timing operations or when measuring memory usage.
    """
    if is_mps_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache(force_gc: bool = False) -> None:
    """
    Empty the MPS cache to free up memory.
    This should be called between large operations or after validation epochs.

    Args:
        force_gc: Whether to also run Python's garbage collector
    """
    if is_mps_available():
        # Empty the MPS cache
        torch.mps.empty_cache()

        # Optionally run Python's garbage collector
        if force_gc:
            import gc

            gc.collect()
            # Empty cache again after GC
            torch.mps.empty_cache()


def deep_clean_memory() -> None:
    """
    Perform a deep cleaning of memory on MPS devices.
    This function is more aggressive than empty_cache and should be used
    before/after major operations that could cause memory pressure.
    """
    if not is_mps_available():
        return

    # First run Python's garbage collector
    import gc

    gc.collect()

    # Ensure all MPS operations are completed
    torch.mps.synchronize()

    # Empty MPS cache
    torch.mps.empty_cache()

    # Short sleep to let system process the cleanup
    import time

    time.sleep(0.1)

    # Create and immediately delete tensors to help force memory reclamation
    try:
        # Create a temporary tensor to help flush memory allocations
        temp_tensor = torch.ones((1, 1), device="mps")
        del temp_tensor
    except Exception:
        pass

    # Run GC and empty cache once more
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()

    # Final GC pass
    gc.collect()


def set_manual_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set the seed for MPS and other devices.

    This is a device-specific extension of set_seed that handles
    MPS-specific seeding when available.

    Args:
        seed: Seed number to set
        deterministic: Whether to set deterministic algorithms in torch
    """
    # Set random libraries seeds
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    # Set standard torch seed
    torch.manual_seed(seed)

    # Set MPS seed if available
    if is_mps_available():
        # MPS doesn't have a separate seed setting function,
        # but we can ensure tensors are created deterministically
        # by setting the global seed and creating a test tensor
        torch.mps.manual_seed(seed)
        # Create a test tensor to ensure proper seeding
        _ = torch.randn(1, device="mps")
        synchronize()
        logger.debug(f"Set MPS random seed to {seed}")

    # Set CUDA seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Set deterministic behavior if requested
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Set deterministic algorithms for ops with non-deterministic implementations
            import os

            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=True)

        logger.debug(f"Set CUDA random seed to {seed} (deterministic={deterministic})")

    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Optimize model for inference on MPS."""
    model.eval()  # Set to evaluation mode

    # Find MultiheadAttention modules and optimize them
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            # On Apple Silicon, we can use a more efficient implementation
            module._qkv_same_embed_dim = True

    return model


def batch_size_finder(
    model, input_shape=(1, 64, 512), target_shape=128, start_batch=64, device=None
):
    """Find optimal batch size for given model on Apple Silicon hardware."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    batch_size = start_batch
    optimal_batch = 1

    while batch_size >= 1:
        try:
            # Test with increasingly large batch sizes
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            dummy_target = torch.ones(
                batch_size, target_shape, dtype=torch.long, device=device
            )

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    model(dummy_input, dummy_target)

                # Time the forward pass
                torch.mps.synchronize()
                start_time = time.time()
                for _ in range(5):
                    model(dummy_input, dummy_target)
                torch.mps.synchronize()
                end_time = time.time()

            duration = end_time - start_time
            throughput = 5 * batch_size / duration

            # Success - record and try larger
            optimal_batch = batch_size
            print(f"Batch size {batch_size}: {throughput:.2f} samples/sec")
            batch_size += 16

            # Clean up memory
            torch.mps.empty_cache()

        except RuntimeError:
            # Memory error, reduce batch size
            batch_size = max(1, batch_size // 2)
            torch.mps.empty_cache()

            # If we've tried this batch size before, we're done
            if batch_size <= optimal_batch:
                break

    # Return optimal batch size with a small safety margin
    return max(1, int(optimal_batch * 0.9))


def limit_mps_memory(fraction: float):
    """Set a memory fraction limit for the MPS device.

    Args:
        fraction (float): The fraction of recommended max memory to allow (0 to disable).
                          Values between 0 and 2 are valid.
    """
    if not is_mps_available():
        logger.warning("Attempted to limit MPS memory, but MPS is not available.")
        return

    if not (0.0 <= fraction <= 2.0):
        logger.error(
            f"Invalid MPS memory fraction: {fraction}. Must be between 0.0 and 2.0."
        )
        return

    if fraction == 0.0:
        logger.info("MPS memory limit disabled (fraction set to 0).")
        # PyTorch docs say 0 means unlimited, which is the default behavior
        # We don't need to explicitly call set_per_process_memory_fraction(0)
        # unless the default behavior changes in future PyTorch versions.
        return

    try:
        # Set environment variables for improved MPS memory behavior
        import os

        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        os.environ["PYTORCH_PREFER_CHANNELS_LAST"] = "1"

        # Set memory fraction limit
        torch.mps.set_per_process_memory_fraction(fraction)
        recommended_max = torch.mps.recommended_max_memory()
        limit_bytes = int(recommended_max * fraction)
        logger.info(
            f"Set MPS process memory fraction limit to {fraction:.2f}. "
            f"Estimated limit: {limit_bytes / (1024**3):.2f} GB / "
            f"{recommended_max / (1024**3):.2f} GB (Recommended Max)"
        )

        # Perform initial clean to ensure we start with a clean slate
        deep_clean_memory()

    except Exception as e:
        logger.error(f"Failed to set MPS memory fraction limit: {e}", exc_info=True)


# --- Seeding ---
def set_seed(seed: int):
    """Set random seeds for reproducibility across libraries."""
    set_manual_seed(seed)


# --- Validation Optimization ---
def optimize_for_validation(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply optimizations specifically for validation on MPS devices.
    This function configures the model for maximum validation efficiency.

    Args:
        model: The model to optimize

    Returns:
        The optimized model
    """
    model.eval()  # Ensure in evaluation mode

    # Apply contiguous memory optimizations to MultiheadAttention modules
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module._qkv_same_embed_dim = True
            # Ensure all parameters are contiguous for faster access
            for name, param in module.named_parameters():
                if param.requires_grad:
                    param.data = param.data.contiguous()

    return model


def temporarily_quantize_model(model, dtype=torch.float16):
    """
    Temporarily convert model parameters to a more memory-efficient dtype.
    This is useful during validation to reduce memory usage.

    Args:
        model: The model to quantize
        dtype: Data type to convert to (default: torch.float16)

    Returns:
        dict: Original dtypes to restore later
    """
    original_dtypes = {}

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_dtypes[name] = param.data.dtype
                param.data = param.data.to(dtype)

    return original_dtypes


def restore_model_dtypes(model, original_dtypes):
    """
    Restore model parameters to their original data types after quantization.

    Args:
        model: The model to restore
        original_dtypes: Dictionary mapping parameter names to original dtypes
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in original_dtypes and param.requires_grad:
                param.data = param.data.to(original_dtypes[name])
