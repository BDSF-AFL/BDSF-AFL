import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List

# pin_memory speeds CPU→GPU transfers when CUDA is available
_PIN_MEMORY = torch.cuda.is_available()

class DataPartitioner:
    def __init__(self, config: dict = None, dataset_name: str = None, N_clients: int = None, alpha: float = None, batch_size: int = None, seed: int = None, data_dir: str = None, **kwargs):
        cfg = config or {}
        self.dataset_name = dataset_name or cfg.get("dataset", "CIFAR10")
        self.data_dir = self._resolve_data_dir(cfg, data_dir)
        self.N = N_clients or cfg.get("N_clients", cfg.get("N", 20))
        self.dirichlet_alpha = alpha if alpha is not None else cfg.get("dirichlet_alpha", 0.1)
        self.seed = seed if seed is not None else cfg.get("seed", 42)
        self.batch_size = batch_size or cfg.get("batch_size", 32)
        
        # Set transforms
        if self.dataset_name == "MNIST":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:  # Default to CIFAR10
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

    def get_dataloaders(self):
        """Convenience method returning (train_loaders, test_loader)."""
        return self.partition(), self.get_test_loader()

    def _resolve_data_dir(self, config: dict = None, data_dir: str = None) -> str:
        """Resolves existing data directory, checking Kaggle paths before fallback."""
        cfg = config or {}
        configured_dir = data_dir or cfg.get("data_dir")
        if configured_dir:
            if configured_dir.endswith("cifar-10-batches-py") and os.path.exists(configured_dir):
                return os.path.dirname(configured_dir)
            if os.path.exists(configured_dir):
                return configured_dir

        if self.dataset_name == "MNIST":
            candidates = [
                "./data/MNIST",
                "./data",
                "data/MNIST",
                "data",
                "/kaggle/input/mnist-dataset",
                "/kaggle/input/mnist",
                "/kaggle/input/mnist-data",
                "/kaggle/input/digit-recognizer",
            ]
        else:  # CIFAR10
            candidates = [
                "./data/cifar-10-batches-py",
                "./data",
                "data/cifar-10-batches-py",
                "data",
                "/kaggle/input/datasets/pankrzysiu/cifar10-python/cifar-10-batches-py",
                "/kaggle/input/datasets/pankrzysiu/cifar10-python",
                "/kaggle/input/cifar10-python/cifar-10-batches-py",
                "/kaggle/input/cifar10-python",
                "/kaggle/input/cifar10/cifar-10-batches-py",
                "/kaggle/input/cifar10",
            ]

        for cand in candidates:
            if os.path.exists(cand):
                if cand.endswith("cifar-10-batches-py"):
                    return os.path.dirname(cand)
                if self.dataset_name == "CIFAR10":
                    if os.path.exists(os.path.join(cand, "cifar-10-batches-py")):
                        return cand
                elif self.dataset_name == "MNIST":
                    if os.path.exists(os.path.join(cand, "MNIST")):
                        return cand
                    if cand.endswith("MNIST"):
                        return os.path.dirname(cand)
                return cand

        return configured_dir if configured_dir else "./data"

    def partition(self) -> List[DataLoader]:
        """Partitions the training dataset and returns N DataLoaders."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Load train dataset
        if self.dataset_name == "MNIST":
            mnist_path = os.path.join(self.data_dir, "MNIST")
            should_download = not os.path.exists(mnist_path)
            train_dataset = datasets.MNIST(root=self.data_dir, train=True, download=should_download, transform=self.transform)
            targets = train_dataset.targets.numpy()
        else:
            cifar_path = os.path.join(self.data_dir, "cifar-10-batches-py")
            should_download = not os.path.exists(cifar_path)
            train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, download=should_download, transform=self.transform)
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
            loader = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=0,      # no subprocess workers — eliminates 40-worker CPU spike
                pin_memory=_PIN_MEMORY,  # enables fast CPU→GPU page-locked transfers
                persistent_workers=False,
            )
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
            mnist_path = os.path.join(self.data_dir, "MNIST")
            should_download = not os.path.exists(mnist_path)
            test_dataset = datasets.MNIST(root=self.data_dir, train=False, download=should_download, transform=self.transform)
        else:
            cifar_path = os.path.join(self.data_dir, "cifar-10-batches-py")
            should_download = not os.path.exists(cifar_path)
            test_dataset = datasets.CIFAR10(root=self.data_dir, train=False, download=should_download, transform=self.transform)
            
        return DataLoader(
            test_dataset,
            batch_size=512,          # larger batch → GPU stays full during eval
            shuffle=False,
            num_workers=0,
            pin_memory=_PIN_MEMORY,
        )
