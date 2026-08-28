"""Centralized device resolution for TPU / CUDA / CPU.

Usage:
    from utils.device_utils import resolve_device, resolve_all_devices, gpu_count, mark_step, XLA_AVAILABLE

    device = resolve_device()           # returns primary torch.device or xla device
    devices = resolve_all_devices()     # returns list of all available GPU devices
    ...
    loss.backward()
    optimizer.step()
    mark_step()                         # no-op on CUDA/CPU, flushes XLA graph on TPU
"""

import torch

# --- XLA availability flag ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


def gpu_count() -> int:
    """Return number of available CUDA GPUs (0 if none or XLA)."""
    if XLA_AVAILABLE:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def resolve_device(config: dict | None = None):
    """Return the primary device object.

    Priority: TPU (XLA) > CUDA (gpu:0) > CPU.
    The *config* dict is updated in-place with the resolved device so that
    downstream code (LocalTrainer, metrics, etc.) can call
    ``config["device"]`` and get the correct object.
    """
    if XLA_AVAILABLE:
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config is not None:
        config["device"] = device

    return device


def resolve_all_devices(config: dict | None = None) -> list:
    """Return list of all available GPU devices, or [cpu] as fallback.

    On a dual-T4 Kaggle instance this returns [cuda:0, cuda:1].
    On a single-GPU machine it returns [cuda:0].
    On CPU-only it returns [cpu].
    """
    if XLA_AVAILABLE:
        return [xm.xla_device()]
    n = gpu_count()
    if n > 0:
        return [torch.device(f"cuda:{i}") for i in range(n)]
    return [torch.device("cpu")]


def mark_step():
    """Flush the XLA lazy-evaluation graph.

    Must be called after every ``optimizer.step()`` on TPU to prevent the
    computation graph from growing unboundedly.  No-op on CUDA / CPU.
    """
    if XLA_AVAILABLE:
        xm.mark_step()


def set_xla_seed(seed: int):
    """Set the XLA random seed (TPU-only). No-op on CUDA / CPU."""
    if XLA_AVAILABLE:
        xm.set_rng_state(seed)


def device_name(device) -> str:
    """Human-readable device name for logging."""
    if XLA_AVAILABLE and str(device).startswith("xla"):
        return f"TPU ({device})"
    if str(device).startswith("cuda"):
        try:
            n = torch.cuda.device_count()
            names = " + ".join(torch.cuda.get_device_name(i) for i in range(n))
            return f"CUDA x{n} ({names})"
        except Exception:
            return f"CUDA ({device})"
    return "CPU"
