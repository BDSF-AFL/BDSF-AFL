import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List

class DataPartitioner:
    def __init__(self, config: dict):
        self.dataset_name = config.get("dataset", "CIFAR10")
        self.N = config.get("N_clients", config.get("N", 20))
        self.dirichlet_alpha = config.get("dirichlet_alpha", 0.1)
        self.seed = config.get("seed", 42)
        self.batch_size = config.get("batch_size", 32)
        
        # Set transforms
        if self.dataset_name == "MNIST":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:  # Default to CIFAR10
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

    def partition(self) -> List[DataLoader]:
        """Partitions the training dataset and returns N DataLoaders."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Load train dataset
        if self.dataset_name == "MNIST":
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
            targets = train_dataset.targets.numpy()
        else:
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
            targets = np.array(train_dataset.targets)
            
        num_samples = len(train_dataset)
        
        if self.dirichlet_alpha == float('inf') or self.dirichlet_alpha is None:
            # IID split: shuffle and split equally
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            splits = np.array_split(indices, self.N)
            client_indices = [split.tolist() for split in splits]
        else:
            # Dirichlet split
            client_indices = self._dirichlet_split(targets, self.dirichlet_alpha)
            
        loaders = []
        for i in range(self.N):
            subset = Subset(train_dataset, client_indices[i])
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            loaders.append(loader)
            
        return loaders

    def _dirichlet_split(self, targets: np.ndarray, alpha: float) -> List[List[int]]:
        """Splits the indices based on target classes and Dirichlet distribution."""
        classes = np.unique(targets)
        client_indices = [[] for _ in range(self.N)]
        
        for c in classes:
            class_idx = np.where(targets == c)[0]
            np.random.shuffle(class_idx)
            
            # Sample proportions from Dirichlet
            proportions = np.random.dirichlet([alpha] * self.N)
            # Convert proportions to counts
            proportions = np.cumsum(proportions * len(class_idx)).astype(int)
            
            # Split indices based on counts
            splits = np.split(class_idx, proportions[:-1])
            for i, split in enumerate(splits):
                client_indices[i].extend(split.tolist())
                
        return client_indices

    def get_test_loader(self) -> DataLoader:
        """Returns the test set DataLoader."""
        if self.dataset_name == "MNIST":
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        else:
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)
            
        return DataLoader(test_dataset, batch_size=256, shuffle=False)
