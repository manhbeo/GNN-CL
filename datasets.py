#TODO: datasets-> projection -> graph-loss -> main(to train)

from graph_pertubation import (EdgePerturbationTransform, SubgraphTransform, FeatureMaskTransform,
                               FeatureDropoutTransform, DiffusionTransform)
import torch
import numpy as np
import pytorch_lightning as pl
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.utils import degree

def get_augmentation_transforms(dataset_name):
    """
    Get appropriate augmentation transforms for the given dataset.
    """
    common_transforms = {
        # Topological transforms
        'edge_perturbation': EdgePerturbationTransform(add_prob=0.1, drop_prob=0.1),
        'subgraph': SubgraphTransform(ratio=0.8),

        # Feature transforms
        'feature_mask': FeatureMaskTransform(mask_prob=0.2),
        'feature_dropout': FeatureDropoutTransform(dropout_prob=0.2),
        'diffusion': DiffusionTransform(alpha=0.2)
    }

    # Dataset-specific transforms
    if dataset_name in ['NCI1', 'PROTEINS', 'DD', 'MUTAG']:
        # Biological networks - prefer topology augmentations
        transform1 = Compose([
            common_transforms['edge_perturbation'],
            common_transforms['feature_mask']
        ])
        transform2 = Compose([
            common_transforms['subgraph'],
            common_transforms['diffusion']
        ])
    elif dataset_name in ['COLLAB', 'IMDB-B']:
        # Social networks - more feature augmentations
        transform1 = Compose([
            common_transforms['edge_perturbation'],
            common_transforms['feature_dropout']
        ])
        transform2 = Compose([
            common_transforms['subgraph'],
            common_transforms['feature_mask']
        ])
    elif dataset_name in ['RDT-B', 'RDT-M5K']:
        # Reddit graphs - balance of both
        transform1 = Compose([
            common_transforms['edge_perturbation'],
            common_transforms['diffusion']
        ])
        transform2 = Compose([
            common_transforms['subgraph'],
            common_transforms['feature_dropout']
        ])
    else:
        # Default augmentations
        transform1 = Compose([
            common_transforms['edge_perturbation'],
            common_transforms['feature_mask']
        ])
        transform2 = Compose([
            common_transforms['subgraph'],
            common_transforms['feature_dropout']
        ])

    return transform1, transform2


class TUDatasetPairTransform(torch.utils.data.Dataset):
    """
    Creates two augmented views of each graph in the TU dataset.
    Implements the proper PyTorch Dataset interface.
    """

    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()

        # Create two views of the graph
        view1 = self.transform1(data.clone())
        view2 = self.transform2(data.clone())

        # Extract necessary attributes
        x_i, edge_index_i = view1.x, view1.edge_index
        x_j, edge_index_j = view2.x, view2.edge_index
        fine_label = data.y

        return (x_i, edge_index_i, x_j, edge_index_j), fine_label


class GraphContrastiveDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for graph contrastive learning.
    Supports various TU datasets and applies transformations to create two views.
    """

    def __init__(
            self,
            dataset_name,
            batch_size=32,
            num_workers=4,
            root='data',
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root = root
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        # Set random seeds for reproducibility
        self.save_hyperparameters(ignore=['root'])
        pl.seed_everything(seed, workers=True)


    def prepare_data(self):
        # Download the data if it doesn't exist
        TUDataset(root=self.root, name=self.dataset_name)

    def setup(self, stage=None):
        # Load dataset
        dataset = TUDataset(root=self.root, name=self.dataset_name)

        # For some datasets, we need to add fake node features if they don't exist
        if not hasattr(dataset[0], 'x') or dataset[0].x is None:
            # For datasets without node features, use one-hot encoding of node degrees
            max_degree = 0
            for data in dataset:
                edge_index = data.edge_index
                if edge_index.size(1) > 0:  # Skip empty graphs
                    d = degree(edge_index[0], num_nodes=data.num_nodes).long()
                    max_degree = max(max_degree, d.max().item())

            # Add one for zero degree nodes
            max_degree = max_degree + 1

            for i, data in enumerate(dataset):
                edge_index = data.edge_index
                num_nodes = data.num_nodes
                if edge_index.size(1) > 0:  # Skip empty graphs
                    d = degree(edge_index[0], num_nodes=num_nodes).long()
                    one_hot = torch.zeros((num_nodes, max_degree), dtype=torch.float)
                    one_hot.scatter_(1, d.unsqueeze(1), 1)
                else:
                    one_hot = torch.zeros((num_nodes, max_degree), dtype=torch.float)
                    one_hot[:, 0] = 1  # All nodes have degree 0

                dataset[i].x = one_hot

        # Get the transforms for this dataset
        transform1, transform2 = get_augmentation_transforms(self.dataset_name)

        # Create dataset with pair transform
        paired_dataset = TUDatasetPairTransform(dataset, transform1, transform2)

        # Split dataset into train, val, test
        num_samples = len(paired_dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        train_idx = int(self.train_ratio * num_samples)
        val_idx = int((self.train_ratio + self.val_ratio) * num_samples)

        train_indices = indices[:train_idx]
        val_indices = indices[train_idx:val_idx]
        test_indices = indices[val_idx:]

        # Create proper subset datasets instead of lists
        self.train_dataset = torch.utils.data.Subset(paired_dataset, train_indices)
        self.val_dataset = torch.utils.data.Subset(paired_dataset, val_indices)
        self.test_dataset = torch.utils.data.Subset(paired_dataset, test_indices)

        # Store feature dimension
        sample_x = dataset[0].x
        self.num_features = sample_x.size(1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def get_num_features(self):
        return self.num_features


# Dataset factory function
def get_dataset(dataset_name, **kwargs):
    """
    Factory function to get the appropriate dataset module.

    Args:
        dataset_name: Name of the dataset (NCI1, PROTEINS, DD, MUTAG, COLLAB, RDT-B, RDT-M5K, IMDB-B)
        **kwargs: Additional arguments to pass to the dataset module

    Returns:
        GraphContrastiveDataModule instance
    """
    # Map from requested names to TUDataset names
    dataset_mapping = {
        'NCI1': 'NCI1',
        'PROTEINS': 'PROTEINS',
        'DD': 'DD',
        'MUTAG': 'MUTAG',
        'COLLAB': 'COLLAB',
        'RDT-B': 'REDDIT-BINARY',
        'RDT-M5K': 'REDDIT-MULTI-5K',
        'IMDB-B': 'IMDB-BINARY'
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(dataset_mapping.keys())}")

    return GraphContrastiveDataModule(
        dataset_name=dataset_mapping[dataset_name],
        **kwargs
    )
