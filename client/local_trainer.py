import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable

from utils.device_utils import mark_step

# GradScaler for AMP mixed-precision (no-op on CPU or older GPUs)
try:
    from torch.amp import autocast, GradScaler
    _AMP_AVAILABLE = True
except ImportError:
    _AMP_AVAILABLE = False


class LocalTrainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, config: dict):
        self.device = config.get("device", "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.local_epochs = config.get("local_epochs", 5)
        self.local_lr = config.get("local_lr", 0.01)
        self.fedprox_mu = config.get("fedprox_mu", 0.0)
        self.criterion = nn.CrossEntropyLoss()
        # Enable AMP (FP16 mixed precision) on CUDA — ~2x throughput on T4/A100
        _is_cuda = isinstance(self.device, torch.device) and self.device.type == "cuda"
        _is_cuda = _is_cuda or (isinstance(self.device, str) and "cuda" in self.device)
        self._use_amp = _AMP_AVAILABLE and _is_cuda
        self._scaler = GradScaler(device="cuda") if self._use_amp else None
        self._stream = torch.cuda.Stream(device=self.device) if _is_cuda else None

    def train(self, W_global: torch.Tensor) -> torch.Tensor:
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                return self._train_impl(W_global)
        return self._train_impl(W_global)

    def _train_impl(self, W_global: torch.Tensor) -> torch.Tensor:
        self._load_weights(W_global)
        W_ref = W_global.clone().to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr)
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for inputs, labels in self.dataloader:
                # non_blocking=True overlaps CPU→GPU transfer with GPU compute
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                if self._use_amp:
                    with autocast(device_type="cuda"):
                        outputs = self.model(inputs)
                        ce_loss = self.criterion(outputs, labels)
                        if self.fedprox_mu > 0:
                            prox_term = 0.0
                            offset = 0
                            for param in self.model.parameters():
                                numel = param.numel()
                                ref_param = W_ref[offset:offset + numel].reshape(param.shape)
                                prox_term += torch.norm(param - ref_param) ** 2
                                offset += numel
                            loss = ce_loss + (self.fedprox_mu / 2.0) * prox_term
                        else:
                            loss = ce_loss
                    self._scaler.scale(loss).backward()
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    outputs = self.model(inputs)
                    ce_loss = self.criterion(outputs, labels)
                    if self.fedprox_mu > 0:
                        prox_term = 0.0
                        offset = 0
                        for param in self.model.parameters():
                            numel = param.numel()
                            ref_param = W_ref[offset:offset + numel].reshape(param.shape)
                            prox_term += torch.norm(param - ref_param) ** 2
                            offset += numel
                        loss = ce_loss + (self.fedprox_mu / 2.0) * prox_term
                    else:
                        loss = ce_loss
                    loss.backward()
                    optimizer.step()

                # Flush XLA graph after each step (no-op on CUDA/CPU)
                mark_step()
                
        W_local = self._get_flat_weights()
        delta_W = W_local - W_global.cpu()
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
