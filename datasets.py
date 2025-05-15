import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch_geometric.datasets import TUDataset, PPI, ZINC, MoleculeNet
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from chem_loader import MoleculeDataset
from bio_loader import BioDataset
from chem_batch import BatchMasking as ChemBatchMasking
from bio_batch import BatchMasking as BioBatchMasking
from torch_geometric.data import Batch


class GraphDataModule(pl.LightningDataModule):
    def __init__(
            self,
            dataset_name,
            root="data",
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            mode="unsupervised",  # "unsupervised", "transfer", "semi_supervised"
            pretrain_dataset=None,  # For transfer learning: "PPI-306K" or "ZINC-2M"
            label_rate=0.1,  # For semi-supervised learning: 0.01 for 1%, 0.1 for 10%
            random_state=42
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mode = mode
        self.pretrain_dataset = pretrain_dataset
        self.label_rate = label_rate
        self.random_state = random_state

        # Set up specific dataset parameters
        self.setup_dataset_params()

    def setup_dataset_params(self):
        """Set specific parameters based on dataset name"""
        # Unsupervised datasets from TU Dataset collection
        self.tu_datasets = ["NCI1", "PROTEINS", "DD", "MUTAG", "COLLAB", "RDT-B", "RDT-M5K", "IMDB-B"]

        # Datasets for finetuning with ZINC-2M pretraining
        self.molecule_datasets = ["Tox21", "ToxCast", "Sider", "ClinTox", "MUV", "HIV", "BBBP", "Bace"]

        # Check dataset type
        self.dataset_type = None
        if self.dataset_name in self.tu_datasets:
            self.dataset_type = "tu"
        elif self.dataset_name in self.molecule_datasets:
            self.dataset_type = "molecule"
        elif self.dataset_name == "PPI":
            self.dataset_type = "ppi"
        elif self.dataset_name == "ZINC":
            self.dataset_type = "zinc"
        elif self.dataset_name == "PPI-306K":
            self.dataset_type = "ppi_large"
        elif self.dataset_name == "ZINC-2M":
            self.dataset_type = "zinc_large"
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def prepare_data(self):
        """Download datasets if not available"""
        # This method is called only once and on a single process
        if self.dataset_type == "tu":
            # TU datasets
            TUDataset(root=os.path.join(self.root, self.dataset_name), name=self.dataset_name)
        elif self.dataset_type == "ppi":
            # PPI dataset
            PPI(root=os.path.join(self.root, 'PPI'))
        elif self.dataset_type == "zinc":
            # ZINC dataset
            ZINC(root=os.path.join(self.root, 'ZINC'))
        elif self.dataset_type == "molecule":
            # MoleculeNet datasets
            MoleculeNet(root=os.path.join(self.root, self.dataset_name), name=self.dataset_name)

        # For large datasets, we assume they are already downloaded and preprocessed

    def setup(self, stage=None):
        """Load datasets based on stage"""
        if stage == 'fit' or stage is None:
            # Load the full dataset
            self.dataset = self._load_dataset()

            # Check if dataset has explicit train/val/test splits
            self.has_explicit_split = hasattr(self.dataset, 'train_mask') and hasattr(self.dataset, 'val_mask')

            if self.has_explicit_split:
                # Use predefined splits
                train_idx = torch.where(self.dataset.train_mask)[0].tolist()
                val_idx = torch.where(self.dataset.val_mask)[0].tolist()

                self.train_dataset = [self.dataset[i] for i in train_idx]
                self.val_dataset = [self.dataset[i] for i in val_idx]
            else:
                # Split into train and validation
                # For semi-supervised, we'll apply label masking to the training set
                train_size = int(0.8 * len(self.dataset))
                val_size = len(self.dataset) - train_size

                # Create train/val splits with fixed random seed for reproducibility
                generator = torch.Generator().manual_seed(self.random_state)
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.dataset, [train_size, val_size], generator=generator
                )

            # For semi-supervised learning, apply label masking to training set
            if self.mode == "semi_supervised":
                self.apply_semi_supervised_masking()

    def apply_semi_supervised_masking(self):
        """Create a mask for labeled data in semi-supervised setting"""
        # Create a mask to track which nodes have labels
        # We don't actually modify the dataset, as eval.py handles the label rate
        num_train_samples = len(self.train_dataset)
        num_labeled = max(1, int(num_train_samples * self.label_rate))

        # Generate random indices for labeled samples
        np.random.seed(self.random_state)
        labeled_indices = np.random.choice(num_train_samples, num_labeled, replace=False)

        # Create a boolean mask
        self.labeled_mask = np.zeros(num_train_samples, dtype=bool)
        self.labeled_mask[labeled_indices] = True

        print(f"Applied {self.label_rate * 100:.1f}% label rate: {num_labeled}/{num_train_samples} samples labeled")

    def _load_dataset(self):
        """Load specific dataset based on name and type"""
        if self.dataset_type == "tu":
            return TUDataset(root=os.path.join(self.root, self.dataset_name), name=self.dataset_name)

        elif self.dataset_type == "molecule":
            return MoleculeNet(root=os.path.join(self.root, self.dataset_name), name=self.dataset_name)

        elif self.dataset_type == "ppi":
            train_dataset = PPI(root=os.path.join(self.root, 'PPI'), split='train')
            val_dataset = PPI(root=os.path.join(self.root, 'PPI'), split='val')
            combined_dataset = train_dataset + val_dataset
            return combined_dataset

        elif self.dataset_type == "zinc":
            train_dataset = ZINC(root=os.path.join(self.root, 'ZINC'), split='train')
            val_dataset = ZINC(root=os.path.join(self.root, 'ZINC'), split='val')

            combined_dataset = train_dataset + val_dataset
            return combined_dataset

        elif self.dataset_type == "ppi_large":
            # Using the custom BioDataset class
            return BioDataset(root=os.path.join(self.root, 'PPI-306K'), data_type='unsupervised')

        elif self.dataset_type == "zinc_large":
            # Using the custom MoleculeDataset class
            return MoleculeDataset(root=os.path.join(self.root, 'ZINC-2M'), dataset='zinc_standard_agent')

    def _collate_fn(self, batch):
        """Custom collate function to handle different data formats"""
        if len(batch) == 0:
            return batch

        # Check batch type and select appropriate collation
        if self.dataset_type == "ppi_large":
            return BioBatchMasking.from_data_list(batch)
        elif self.dataset_type == "zinc_large":
            return ChemBatchMasking.from_data_list(batch)
        else:
            # Default PyG collation
            return Batch.from_data_list(batch)

    def process_batch_for_model(self, batch):
        """Process batch into format expected by model (x, edge_index, labels)"""
        # Different datasets have different formats, so we need to standardize
        if self.dataset_type in ["tu", "molecule"]:
            # TU datasets and MoleculeNet typically have x, edge_index, y
            x = batch.x
            edge_index = batch.edge_index
            if hasattr(batch, 'y'):
                y = batch.y
                # Handle different y shapes
                if y.dim() > 1 and y.size(1) == 1:
                    y = y.squeeze(1)
            else:
                y = torch.zeros(batch.num_nodes, dtype=torch.long)

            return x, edge_index, y

        elif self.dataset_type in ["ppi", "ppi_large"]:
            # PPI datasets
            x = batch.x
            edge_index = batch.edge_index
            if hasattr(batch, 'y'):
                y = batch.y
            else:
                y = torch.zeros(batch.num_nodes, dtype=torch.long)

            return x, edge_index, y

        elif self.dataset_type in ["zinc", "zinc_large"]:
            # ZINC datasets
            x = batch.x
            edge_index = batch.edge_index
            if hasattr(batch, 'y'):
                y = batch.y
            else:
                y = torch.zeros(batch.num_nodes, dtype=torch.long)

            return x, edge_index, y

        else:
            # Default handling
            x = batch.x if hasattr(batch, 'x') else None
            edge_index = batch.edge_index if hasattr(batch, 'edge_index') else None
            y = batch.y if hasattr(batch, 'y') else torch.zeros(batch.num_nodes, dtype=torch.long)

            return x, edge_index, y

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

    def get_entire_dataset(self):
        """Return the entire dataset for evaluation purposes"""
        return self.dataset


class TransferDataModule(pl.LightningDataModule):
    """Specialized data module for transfer learning scenarios"""

    def __init__(
            self,
            pretrain_dataset,
            finetune_dataset,
            root="data",
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            random_state=42
    ):
        super().__init__()
        self.pretrain_dataset_name = pretrain_dataset
        self.finetune_dataset_name = finetune_dataset
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.random_state = random_state

        # Create individual data modules for pretraining and finetuning
        self.pretrain_dm = GraphDataModule(
            dataset_name=pretrain_dataset,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mode="unsupervised",
            random_state=random_state
        )

        self.finetune_dm = GraphDataModule(
            dataset_name=finetune_dataset,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mode="unsupervised",  # We handle fine-tuning evaluation ourselves
            random_state=random_state
        )

    def prepare_data(self):
        """Download datasets if not available"""
        self.pretrain_dm.prepare_data()
        self.finetune_dm.prepare_data()

    def setup(self, stage=None):
        """Setup datasets for pretraining or finetuning based on stage"""
        if stage == 'pretrain' or stage is None:
            self.pretrain_dm.setup(stage='fit')

        if stage == 'finetune' or stage is None:
            self.finetune_dm.setup(stage='fit')

    def pretrain_dataloader(self):
        """Return dataloader for pretraining"""
        return self.pretrain_dm.train_dataloader()

    def pretrain_val_dataloader(self):
        """Return validation dataloader for pretraining"""
        return self.pretrain_dm.val_dataloader()

    def finetune_train_dataloader(self):
        """Return dataloader for finetuning training"""
        return self.finetune_dm.train_dataloader()

    def finetune_val_dataloader(self):
        """Return dataloader for finetuning validation"""
        return self.finetune_dm.val_dataloader()

    def get_finetune_dataset(self):
        """Return the entire finetuning dataset for evaluation purposes"""
        return self.finetune_dm.get_entire_dataset()


class SemiSupervisedDataModule(GraphDataModule):
    """Specialized data module for semi-supervised learning"""

    def __init__(
            self,
            dataset_name,
            root="data",
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            label_rate=0.1,  # 0.01 for 1%, 0.1 for 10%
            random_state=42
    ):
        super().__init__(
            dataset_name=dataset_name,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mode="semi_supervised",
            label_rate=label_rate,
            random_state=random_state
        )

    def get_labeled_mask(self):
        """Return the mask indicating which training samples are labeled"""
        return self.labeled_mask
