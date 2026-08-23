import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable, Optional

from utils.device_utils import mark_step

# GradScaler for AMP mixed-precision (no-op on CPU or older GPUs)
try:
    from torch.amp import autocast, GradScaler
    _AMP_AVAILABLE = True
except ImportError:
    _AMP_AVAILABLE = False


class LocalTrainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, config: dict):
        self.config = config
        self.device = config.get("device", "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.local_epochs = config.get("local_epochs", 1)
        self.local_lr = float(config.get("local_lr", config.get("lr", 0.01)))
        self.lr_schedule = config.get("lr_schedule", "cosine")
        self.lr_min = float(config.get("lr_min", 0.001))
        self.lr_decay = float(config.get("lr_decay", 0.995))
        self.total_rounds = int(config.get("total_rounds", config.get("rounds", 300)))
        self.weight_decay = float(config.get("weight_decay", 1e-4))
        self.fedprox_mu = float(config.get("fedprox_mu", 0.0))
        self.criterion = nn.CrossEntropyLoss()

        # Enable AMP (FP16 mixed precision) on CUDA — ~2x throughput on T4/A100
        _is_cuda = isinstance(self.device, torch.device) and self.device.type == "cuda"
        _is_cuda = _is_cuda or (isinstance(self.device, str) and "cuda" in self.device)
        self._use_amp = _AMP_AVAILABLE and _is_cuda
        self._scaler = GradScaler(device="cuda") if self._use_amp else None
        self._stream = torch.cuda.Stream(device=self.device) if _is_cuda else None

    def _get_current_lr(self, current_round: int) -> float:
        """Calculates dynamic learning rate for the current global round."""
        if self.lr_schedule == "cosine":
            t = min(current_round, self.total_rounds)
            return self.lr_min + 0.5 * (self.local_lr - self.lr_min) * (1.0 + math.cos(math.pi * t / max(1, self.total_rounds)))
        elif self.lr_schedule == "exponential":
            return max(self.lr_min, self.local_lr * (self.lr_decay ** current_round))
        elif self.lr_schedule == "step":
            step_size = max(1, int(self.config.get("lr_step_size", 50)))
            return max(self.lr_min, self.local_lr * (0.5 ** (current_round // step_size)))
        else:
            return self.local_lr

    def train(self, W_global: torch.Tensor, current_round: int = 0) -> torch.Tensor:
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                return self._train_impl(W_global, current_round=current_round)
        return self._train_impl(W_global, current_round=current_round)

    def _train_impl(self, W_global: torch.Tensor, current_round: int = 0) -> torch.Tensor:
        self._load_weights(W_global)
        lr_t = self._get_current_lr(current_round)
        optimizer = optim.SGD(self.model.parameters(), lr=lr_t, weight_decay=self.weight_decay)
        self.model.train()

        # Cache reference parameter tensors on device for fast vectorized FedProx computation
        ref_params = [p.detach().clone() for p in self.model.parameters()] if self.fedprox_mu > 0.0 else []

        for epoch in range(self.local_epochs):
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                if self._use_amp:
                    with autocast(device_type="cuda"):
                        outputs = self.model(inputs)
                        ce_loss = self.criterion(outputs, labels)
                        if self.fedprox_mu > 0.0:
                            prox_term = sum(torch.sum((p - ref_p) ** 2) for p, ref_p in zip(self.model.parameters(), ref_params))
                            loss = ce_loss + (self.fedprox_mu / 2.0) * prox_term
                        else:
                            loss = ce_loss
                    self._scaler.scale(loss).backward()
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self.model(inputs)
                    ce_loss = self.criterion(outputs, labels)
                    if self.fedprox_mu > 0.0:
                        prox_term = sum(torch.sum((p - ref_p) ** 2) for p, ref_p in zip(self.model.parameters(), ref_params))
                        loss = ce_loss + (self.fedprox_mu / 2.0) * prox_term
                    else:
                        loss = ce_loss
                    loss.backward()
                    optimizer.step()

                # Flush XLA graph after each step (no-op on CUDA/CPU)
                mark_step()

        W_local = self._get_flat_weights()
        delta_W = W_local - W_global.cpu()

        # VRAM cleanup
        del ref_params, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return delta_W

    def _load_weights(self, W: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for param in self.model.parameters():
                numel = param.numel()
                param.copy_(W[offset:offset + numel].reshape(param.shape).to(self.device))
                offset += numel

    def _get_flat_weights(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.model.parameters()]).cpu()
