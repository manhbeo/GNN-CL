import os
from typing import Callable, Optional, Tuple, Any

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, Subset
from torch_geometric.datasets import TUDataset, ZINC, MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data

from chem_loader import MoleculeDataset
from chem_batch import BatchMasking as ChemBatchMasking


def identity_augment(data: Data) -> Data:
    return data.clone()


class ContrastiveDataset(Dataset):
    """
    Wrap a graph dataset and return two augmented views of the same graph.
    """

    def __init__(
        self,
        base_dataset,
        aug1: Optional[Callable[[Data], Data]] = None,
        aug2: Optional[Callable[[Data], Data]] = None,
    ):
        self.base_dataset = base_dataset
        self.aug1 = aug1 if aug1 is not None else identity_augment
        self.aug2 = aug2 if aug2 is not None else identity_augment

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        view1 = self.aug1(data.clone())
        view2 = self.aug2(data.clone())
        return view1, view2


class GraphDataModule(pl.LightningDataModule):
    TU_UNSUPERVISED = {
        "nci1", "proteins", "dd", "mutag", "collab", "rdt-b", "rdt-m5k", "imdb-b"
    }

    TU_SEMISUPERVISED = {
        "nci1", "proteins", "dd", "collab", "rdt-b", "rdt-m5k"
    }

    MOLECULENET_DOWNSTREAM = {
        "tox21", "toxcast", "sider", "clintox", "muv", "hiv", "bbbp", "bace"
    }

    TU_NAME_MAP = {
        "nci1": "NCI1",
        "proteins": "PROTEINS",
        "dd": "DD",
        "mutag": "MUTAG",
        "collab": "COLLAB",
        "rdt-b": "REDDIT-BINARY",
        "rdt-m5k": "REDDIT-MULTI-5K",
        "imdb-b": "IMDB-BINARY",
    }

    MOLECULENET_NAME_MAP = {
        "tox21": "Tox21",
        "toxcast": "ToxCast",
        "sider": "SIDER",
        "clintox": "ClinTox",
        "muv": "MUV",
        "hiv": "HIV",
        "bbbp": "BBBP",
        "bace": "BACE",
    }

    def __init__(
        self,
        dataset_name: str,
        root: str = ".",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        mode: str = "unsupervised",   # "unsupervised", "semi_supervised", "supervised"
        label_rate: float = 0.1,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
        zinc_subset: bool = False,
        persistent_workers: Optional[bool] = None,
        drop_last_train: bool = True,
        train_aug1: Optional[Callable[[Data], Data]] = None,
        train_aug2: Optional[Callable[[Data], Data]] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mode = mode
        self.label_rate = label_rate
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.zinc_subset = zinc_subset
        self.drop_last_train = drop_last_train
        self.train_aug1 = train_aug1 if train_aug1 is not None else identity_augment
        self.train_aug2 = train_aug2 if train_aug2 is not None else identity_augment
        self.persistent_workers = (
            persistent_workers if persistent_workers is not None else num_workers > 0
        )

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_pair_dataset = None

        self.labeled_mask = None
        self.labeled_train_dataset = None
        self.unlabeled_train_dataset = None

    # ----------------------------
    # Helpers
    # ----------------------------

    def _is_tu(self) -> bool:
        return self.dataset_name in self.TU_UNSUPERVISED or self.dataset_name in self.TU_SEMISUPERVISED

    def _is_moleculenet(self) -> bool:
        return self.dataset_name in self.MOLECULENET_DOWNSTREAM

    def _is_snap_chem(self) -> bool:
        return self.dataset_name == "zinc-2m"

    def _is_regular_zinc(self) -> bool:
        return self.dataset_name == "zinc"

    def _random_split_dataset(self, dataset) -> Tuple[Subset, Optional[Subset], Optional[Subset]]:
        n = len(dataset)
        n_val = int(n * self.val_ratio)
        n_test = int(n * self.test_ratio)
        n_train = n - n_val - n_test

        if n_train <= 0:
            raise ValueError(
                f"Invalid split sizes for dataset length {n}: "
                f"train={n_train}, val={n_val}, test={n_test}"
            )

        generator = torch.Generator().manual_seed(self.random_state)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )
        return train_ds, val_ds, test_ds

    def _batch_graphs(self, batch_list):
        if len(batch_list) == 0:
            return batch_list

        if self._is_snap_chem():
            return ChemBatchMasking.from_data_list(batch_list)

        return Batch.from_data_list(batch_list)

    def _collate_pair(self, batch):
        view1_list = [x[0] for x in batch]
        view2_list = [x[1] for x in batch]
        return self._batch_graphs(view1_list), self._batch_graphs(view2_list)

    def _collate_single(self, batch):
        return self._batch_graphs(batch)

    # ----------------------------
    # Dataset loading
    # ----------------------------

    def prepare_data(self):
        if self._is_tu():
            TUDataset(
                root=os.path.join(self.root, self.dataset_name),
                name=self.TU_NAME_MAP[self.dataset_name],
            )

        elif self._is_regular_zinc():
            ZINC(root=os.path.join(self.root, "zinc"), split="train", subset=self.zinc_subset)
            ZINC(root=os.path.join(self.root, "zinc"), split="val", subset=self.zinc_subset)
            ZINC(root=os.path.join(self.root, "zinc"), split="test", subset=self.zinc_subset)

        elif self._is_moleculenet():
            MoleculeNet(
                root=os.path.join(self.root, self.dataset_name),
                name=self.MOLECULENET_NAME_MAP[self.dataset_name],
            )

        elif self._is_snap_chem():
            MoleculeDataset(
                root=os.path.join(self.root, "chem", "dataset", "zinc_standard_agent"),
                dataset="zinc_standard_agent",
            )

        else:
            raise ValueError(f"Unsupported dataset_name for graph-level setting: {self.dataset_name}")

    def setup(self, stage: Optional[str] = None):
        if stage not in (None, "fit", "validate", "test", "predict"):
            return

        if self.train_dataset is not None:
            return

        if self._is_regular_zinc():
            self.train_dataset = ZINC(
                root=os.path.join(self.root, "zinc"),
                split="train",
                subset=self.zinc_subset,
            )
            self.val_dataset = ZINC(
                root=os.path.join(self.root, "zinc"),
                split="val",
                subset=self.zinc_subset,
            )
            self.test_dataset = ZINC(
                root=os.path.join(self.root, "zinc"),
                split="test",
                subset=self.zinc_subset,
            )
            self.dataset = self.train_dataset

        elif self._is_snap_chem():
            dataset = MoleculeDataset(
                root=os.path.join(self.root, "chem", "dataset", "zinc_standard_agent"),
                dataset="zinc_standard_agent",
            )
            self.dataset = dataset
            self.train_dataset = dataset
            self.val_dataset = None
            self.test_dataset = None

        elif self._is_tu():
            dataset = TUDataset(
                root=os.path.join(self.root, self.dataset_name),
                name=self.TU_NAME_MAP[self.dataset_name],
            )
            self.dataset = dataset

            if self.mode == "unsupervised":
                self.train_dataset = dataset
                self.val_dataset = None
                self.test_dataset = None
            else:
                self.train_dataset, self.val_dataset, self.test_dataset = self._random_split_dataset(dataset)

        elif self._is_moleculenet():
            dataset = MoleculeNet(
                root=os.path.join(self.root, self.dataset_name),
                name=self.MOLECULENET_NAME_MAP[self.dataset_name],
            )
            self.dataset = dataset
            self.train_dataset, self.val_dataset, self.test_dataset = self._random_split_dataset(dataset)

        else:
            raise ValueError(f"Unsupported dataset_name for graph-level setting: {self.dataset_name}")

        if self.mode == "semi_supervised":
            self._apply_semi_supervised_split()

        self.train_pair_dataset = ContrastiveDataset(
            self.train_dataset,
            aug1=self.train_aug1,
            aug2=self.train_aug2,
        )

    def _apply_semi_supervised_split(self):
        if self.dataset_name not in self.TU_SEMISUPERVISED:
            raise ValueError(
                "Semi-supervised mode is only configured here for the TU semi-supervised datasets."
            )

        num_train = len(self.train_dataset)
        num_labeled = max(1, int(num_train * self.label_rate))

        generator = torch.Generator().manual_seed(self.random_state)
        perm = torch.randperm(num_train, generator=generator)
        labeled_local_indices = perm[:num_labeled].tolist()
        unlabeled_local_indices = perm[num_labeled:].tolist()

        self.labeled_mask = torch.zeros(num_train, dtype=torch.bool)
        self.labeled_mask[labeled_local_indices] = True

        if isinstance(self.train_dataset, Subset):
            base_dataset = self.train_dataset.dataset
            base_indices = list(self.train_dataset.indices)

            labeled_base_indices = [base_indices[i] for i in labeled_local_indices]
            unlabeled_base_indices = [base_indices[i] for i in unlabeled_local_indices]

            self.labeled_train_dataset = Subset(base_dataset, labeled_base_indices)
            self.unlabeled_train_dataset = Subset(base_dataset, unlabeled_base_indices)
        else:
            self.labeled_train_dataset = Subset(self.train_dataset, labeled_local_indices)
            self.unlabeled_train_dataset = Subset(self.train_dataset, unlabeled_local_indices)

        self.train_dataset = self.labeled_train_dataset

    # ----------------------------
    # Batch formatting
    # ----------------------------

    def process_batch_for_model(self, batch: Any):
        """
        Single-view batch:
            returns x, edge_index, batch_index, y, edge_attr

        Two-view batch:
            returns (x1, ei1, b1, y1, ea1), (x2, ei2, b2, y2, ea2)
        """
        if isinstance(batch, tuple) and len(batch) == 2:
            return self.process_batch_for_model(batch[0]), self.process_batch_for_model(batch[1])

        x = batch.x if hasattr(batch, "x") else None
        edge_index = batch.edge_index if hasattr(batch, "edge_index") else None
        batch_index = batch.batch if hasattr(batch, "batch") else None
        edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
        y = batch.y if hasattr(batch, "y") else None

        if y is not None and y.dim() > 1 and y.size(-1) == 1:
            y = y.squeeze(-1)

        return x, edge_index, batch_index, y, edge_attr

    # ----------------------------
    # Dataloaders
    # ----------------------------

    def train_dataloader(self):
        return DataLoader(
            self.train_pair_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last_train,
            collate_fn=self._collate_pair,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_single,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_single,
        )

    # ----------------------------
    # Convenience getters
    # ----------------------------

    def get_entire_dataset(self):
        return self.dataset

    def get_labeled_mask(self):
        return self.labeled_mask

    def get_unlabeled_train_dataset(self):
        return self.unlabeled_train_dataset


class TransferDataModule(pl.LightningDataModule):
    VALID_PRETRAIN = {"zinc-2m"}
    VALID_DOWNSTREAM = {"tox21", "toxcast", "sider", "clintox", "muv", "hiv", "bbbp", "bace"}

    def __init__(
        self,
        pretrain_dataset: str,
        finetune_dataset: str,
        root: str = ".",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        random_state: int = 42,
        zinc_subset: bool = False,
        pretrain_aug1: Optional[Callable[[Data], Data]] = None,
        pretrain_aug2: Optional[Callable[[Data], Data]] = None,
        finetune_aug1: Optional[Callable[[Data], Data]] = None,
        finetune_aug2: Optional[Callable[[Data], Data]] = None,
    ):
        super().__init__()

        pretrain_dataset = pretrain_dataset.lower()
        finetune_dataset = finetune_dataset.lower()

        if pretrain_dataset not in self.VALID_PRETRAIN:
            raise ValueError(f"Invalid graph-level pretrain dataset: {pretrain_dataset}")

        if finetune_dataset not in self.VALID_DOWNSTREAM:
            raise ValueError(
                f"Invalid graph-level downstream dataset: {finetune_dataset}"
            )

        self.pretrain_dm = GraphDataModule(
            dataset_name=pretrain_dataset,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mode="unsupervised",
            random_state=random_state,
            zinc_subset=zinc_subset,
            train_aug1=pretrain_aug1,
            train_aug2=pretrain_aug2,
        )

        self.finetune_dm = GraphDataModule(
            dataset_name=finetune_dataset,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mode="supervised",
            random_state=random_state,
            zinc_subset=zinc_subset,
            train_aug1=finetune_aug1,
            train_aug2=finetune_aug2,
        )

    def prepare_data(self):
        self.pretrain_dm.prepare_data()
        self.finetune_dm.prepare_data()

    def setup(self, stage: Optional[str] = None):
        self.pretrain_dm.setup("fit")
        self.finetune_dm.setup("fit")

    def pretrain_dataloader(self):
        return self.pretrain_dm.train_dataloader()

    def pretrain_val_dataloader(self):
        return self.pretrain_dm.val_dataloader()

    def finetune_train_dataloader(self):
        return self.finetune_dm.train_dataloader()

    def finetune_val_dataloader(self):
        return self.finetune_dm.val_dataloader()

    def finetune_test_dataloader(self):
        return self.finetune_dm.test_dataloader()

    def get_finetune_dataset(self):
        return self.finetune_dm.get_entire_dataset()


class SemiSupervisedDataModule(GraphDataModule):
    def __init__(
        self,
        dataset_name: str,
        root: str = ".",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        label_rate: float = 0.1,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
        train_aug1: Optional[Callable[[Data], Data]] = None,
        train_aug2: Optional[Callable[[Data], Data]] = None,
    ):
        dataset_name = dataset_name.lower()
        if dataset_name not in GraphDataModule.TU_SEMISUPERVISED:
            raise ValueError(
                f"{dataset_name} is not in the configured TU semi-supervised set: "
                f"{sorted(GraphDataModule.TU_SEMISUPERVISED)}"
            )

        super().__init__(
            dataset_name=dataset_name,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            mode="semi_supervised",
            label_rate=label_rate,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
            train_aug1=train_aug1,
            train_aug2=train_aug2,
        )