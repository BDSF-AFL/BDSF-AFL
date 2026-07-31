"""Centralized device resolution for TPU / CUDA / CPU.

Usage:
    from utils.device_utils import resolve_device, mark_step, XLA_AVAILABLE

    device = resolve_device()          # returns torch.device or xla device
    ...
    loss.backward()
    optimizer.step()
    mark_step()                        # no-op on CUDA/CPU, flushes XLA graph on TPU
"""

import torch

# --- XLA availability flag ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


def resolve_device(config: dict | None = None):
    """Return the best available device object.

    Priority: TPU (XLA) > CUDA > CPU.
    The *config* dict is updated in-place with the resolved device so that
    downstream code (LocalTrainer, metrics, etc.) can call
    ``config["device"]`` and get the correct object.
    """
    if XLA_AVAILABLE:
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if config is not None:
        config["device"] = device

    return device


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
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            return f"CUDA ({device})"
    return f"CPU"
