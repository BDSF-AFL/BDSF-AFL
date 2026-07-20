import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Callable

class LocalTrainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, config: dict):
        self.device = config.get("device", "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.local_epochs = config.get("local_epochs", 5)
        self.local_lr = config.get("local_lr", 0.01)
        self.fedprox_mu = config.get("fedprox_mu", 0.0)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, W_global: torch.Tensor) -> torch.Tensor:
        self._load_weights(W_global)
        W_ref = W_global.clone().to(self.device)
        optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr)
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
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
                    prox_term = (self.fedprox_mu / 2.0) * prox_term
                    loss = ce_loss + prox_term
                else:
                    loss = ce_loss
                    
                loss.backward()
                optimizer.step()
                
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
