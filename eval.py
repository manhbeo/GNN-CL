import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader


# =========================================================
# Basic utilities
# =========================================================

def get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_torch_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_long_labels(y: torch.Tensor) -> torch.Tensor:
    """
    Convert graph labels to class indices for multiclass classification.
    """
    if y.dim() > 1 and y.size(-1) == 1:
        y = y.squeeze(-1)
    return y.long()


def normalize_binary_targets(y: torch.Tensor) -> torch.Tensor:
    """
    Normalize binary / multilabel targets to {0,1} when possible.
    Supports:
    - {-1, 1}
    - {0, 1}
    """
    y = y.float()
    unique_vals = torch.unique(y[torch.isfinite(y)])
    unique_set = set(unique_vals.detach().cpu().tolist())

    if unique_set.issubset({-1.0, 1.0}):
        return (y + 1.0) / 2.0
    return y


def infer_task_type_from_labels(y: torch.Tensor) -> str:
    """
    Infer graph-level task type from batched labels.

    Returns one of:
    - 'multiclass'
    - 'binary'
    - 'multilabel'
    """
    if y.dim() == 1:
        unique_vals = torch.unique(y[torch.isfinite(y)])
        unique_set = set(unique_vals.detach().cpu().tolist())

        if unique_set.issubset({0.0, 1.0, -1.0}):
            return "binary"
        return "multiclass"

    if y.dim() == 2 and y.size(1) == 1:
        unique_vals = torch.unique(y[torch.isfinite(y)])
        unique_set = set(unique_vals.detach().cpu().tolist())
        if unique_set.issubset({0.0, 1.0, -1.0}):
            return "binary"
        return "multiclass"

    return "multilabel"


# =========================================================
# Metrics
# =========================================================

def multiclass_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    return acc


def binary_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    y = normalize_binary_targets(y).float()
    return (pred == y).float().mean().item()


def _binary_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> Optional[float]:
    """
    Pure PyTorch AUROC for binary labels.
    Returns None if AUC is undefined.
    """
    y_true = y_true.detach().float()
    y_score = y_score.detach().float()

    mask = torch.isfinite(y_true) & torch.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if y_true.numel() == 0:
        return None

    y_true = normalize_binary_targets(y_true)
    pos = (y_true == 1).sum().item()
    neg = (y_true == 0).sum().item()

    if pos == 0 or neg == 0:
        return None

    sorted_idx = torch.argsort(y_score)
    ranks = torch.zeros_like(sorted_idx, dtype=torch.float)
    ranks[sorted_idx] = torch.arange(1, y_score.numel() + 1, device=y_score.device, dtype=torch.float)

    pos_ranks = ranks[y_true == 1].sum()
    auc = (pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    return auc.item()


def multilabel_mean_auc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Mean AUROC across tasks, ignoring undefined tasks.
    """
    probs = torch.sigmoid(logits)
    y = normalize_binary_targets(y)

    aucs = []
    for t in range(y.size(1)):
        auc = _binary_auc(y[:, t], probs[:, t])
        if auc is not None:
            aucs.append(auc)

    if len(aucs) == 0:
        return float("nan")
    return float(sum(aucs) / len(aucs))


# =========================================================
# Stratified splitting in pure PyTorch
# =========================================================

def stratified_kfold_indices(
    y: torch.Tensor,
    n_splits: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    Returns list of folds, each fold is a list of indices.
    Works for single-label classification only.
    """
    y = to_long_labels(y).cpu()
    generator = torch.Generator().manual_seed(seed)

    classes = torch.unique(y)
    folds = [[] for _ in range(n_splits)]

    for c in classes:
        idx = torch.where(y == c)[0]
        idx = idx[torch.randperm(idx.numel(), generator=generator)]

        chunks = torch.chunk(idx, n_splits)
        for k in range(n_splits):
            if k < len(chunks):
                folds[k].extend(chunks[k].tolist())

    return folds


def stratified_subset_indices(
    y: torch.Tensor,
    fraction: float,
    seed: int = 42,
) -> List[int]:
    """
    Stratified subset sampling for single-label classification.
    """
    y = to_long_labels(y).cpu()
    generator = torch.Generator().manual_seed(seed)
    selected = []

    for c in torch.unique(y):
        idx = torch.where(y == c)[0]
        idx = idx[torch.randperm(idx.numel(), generator=generator)]
        k = max(1, int(math.floor(idx.numel() * fraction)))
        selected.extend(idx[:k].tolist())

    return selected


def stratified_train_val_split_from_indices(
    y: torch.Tensor,
    indices: List[int],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Stratified split of a provided index list into train / val.
    """
    y_sub = y[indices]
    y_sub = to_long_labels(y_sub)

    train_part = []
    val_part = []
    generator = torch.Generator().manual_seed(seed)

    for c in torch.unique(y_sub):
        class_local = torch.where(y_sub == c)[0]
        class_local = class_local[torch.randperm(class_local.numel(), generator=generator)]

        n_val = max(1, int(math.floor(class_local.numel() * val_ratio))) if class_local.numel() > 1 else 0
        val_local = class_local[:n_val]
        train_local = class_local[n_val:]

        val_part.extend([indices[i.item()] for i in val_local])
        train_part.extend([indices[i.item()] for i in train_local])

    return train_part, val_part


# =========================================================
# Graph embedding extraction
# =========================================================

@torch.no_grad()
def extract_graph_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    data_module,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract graph-level embeddings h and labels y from a single-view dataloader.
    Assumes model returns graph embeddings through:
        model(..., return_projection=False)
    or:
        model.encode(...)
    """
    device = get_device(device)
    model = model.to(device)
    model.eval()

    all_h = []
    all_y = []

    for batch in dataloader:
        x, edge_index, batch_index, y, edge_attr = data_module.process_batch_for_model(batch)

        x = x.to(device)
        edge_index = edge_index.to(device)
        batch_index = batch_index.to(device)
        edge_attr = edge_attr.to(device) if edge_attr is not None else None

        if hasattr(model, "encode"):
            h = model.encode(x, edge_index, batch_index, edge_attr=edge_attr)
        else:
            h = model(x, edge_index, batch_index, edge_attr=edge_attr, return_projection=False)

        all_h.append(h.detach().cpu())
        if y is None:
            raise ValueError("Labels are required for evaluation but y is None.")
        all_y.append(y.detach().cpu())

    H = torch.cat(all_h, dim=0)
    Y = torch.cat(all_y, dim=0)
    return H, Y


# =========================================================
# Linear SVM in PyTorch
# =========================================================

class LinearSVM(nn.Module):
    """
    Multiclass linear SVM using Crammer-Singer hinge loss.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def loss(self, x: torch.Tensor, y: torch.Tensor, weight_decay: float = 1e-4) -> torch.Tensor:
        scores = self.forward(x)
        correct_scores = scores.gather(1, y.unsqueeze(1))
        margins = F.relu(scores - correct_scores + 1.0)
        margins.scatter_(1, y.unsqueeze(1), 0.0)
        data_loss = margins.sum(dim=1).mean()
        reg_loss = 0.5 * weight_decay * self.fc.weight.pow(2).sum()
        return data_loss + reg_loss


def train_linear_svm(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    epochs: int = 300,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> LinearSVM:
    device = get_device(device)
    set_torch_seed(seed)

    x_train = x_train.to(device)
    y_train = to_long_labels(y_train).to(device)

    num_classes = int(torch.max(y_train).item()) + 1
    model = LinearSVM(x_train.size(1), num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -1.0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(x_train, y_train, weight_decay=weight_decay)
        loss.backward()
        optimizer.step()

        if x_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                logits = model(x_val.to(device))
                acc = multiclass_accuracy(logits.cpu(), to_long_labels(y_val))
            if acc > best_metric:
                best_metric = acc
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


# =========================================================
# Fine-tuning head for graph-level downstream evaluation
# =========================================================

class GraphClassifier(nn.Module):
    """
    Wrap a pretrained graph encoder with a graph-level prediction head.
    """
    def __init__(self, backbone: nn.Module, emb_dim: int, out_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(emb_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self.backbone, "encode"):
            h = self.backbone.encode(x, edge_index, batch_index, edge_attr=edge_attr)
        else:
            h = self.backbone(x, edge_index, batch_index, edge_attr=edge_attr, return_projection=False)
        return self.head(h)


def infer_embedding_dim(
    model: nn.Module,
    dataloader: DataLoader,
    data_module,
    device: Optional[torch.device] = None,
) -> int:
    H, _ = extract_graph_embeddings(model, dataloader, data_module, device=device)
    return H.size(1)


def build_single_view_loader(dataset, data_module, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=getattr(data_module, "pin_memory", True),
        persistent_workers=(num_workers > 0),
        collate_fn=data_module._collate_single,
    )


def evaluate_model_on_loader(
    model: nn.Module,
    dataloader: DataLoader,
    data_module,
    task_type: str,
    device: Optional[torch.device] = None,
) -> float:
    device = get_device(device)
    model.eval()
    all_logits = []
    all_y = []

    with torch.no_grad():
        for batch in dataloader:
            x, edge_index, batch_index, y, edge_attr = data_module.process_batch_for_model(batch)

            x = x.to(device)
            edge_index = edge_index.to(device)
            batch_index = batch_index.to(device)
            edge_attr = edge_attr.to(device) if edge_attr is not None else None

            logits = model(x, edge_index, batch_index, edge_attr=edge_attr)
            all_logits.append(logits.cpu())
            all_y.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)

    if task_type == "multiclass":
        return multiclass_accuracy(logits, to_long_labels(y))
    elif task_type == "binary":
        return _binary_auc(normalize_binary_targets(y.view(-1)), torch.sigmoid(logits.view(-1)))
    elif task_type == "multilabel":
        return multilabel_mean_auc_from_logits(logits, y)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def train_finetune_classifier(
    backbone: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    data_module,
    task_type: str,
    out_dim: int,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None,
) -> GraphClassifier:
    device = get_device(device)

    emb_dim = infer_embedding_dim(backbone, train_loader, data_module, device=device)
    model = GraphClassifier(copy.deepcopy(backbone), emb_dim=emb_dim, out_dim=out_dim).to(device)

    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_metric = -float("inf")

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            x, edge_index, batch_index, y, edge_attr = data_module.process_batch_for_model(batch)

            x = x.to(device)
            edge_index = edge_index.to(device)
            batch_index = batch_index.to(device)
            edge_attr = edge_attr.to(device) if edge_attr is not None else None
            y = y.to(device)

            logits = model(x, edge_index, batch_index, edge_attr=edge_attr)

            if task_type == "multiclass":
                loss = F.cross_entropy(logits, to_long_labels(y))
            elif task_type == "binary":
                target = normalize_binary_targets(y).view_as(logits).float()
                loss = F.binary_cross_entropy_with_logits(logits, target)
            elif task_type == "multilabel":
                target = normalize_binary_targets(y).float()
                valid_mask = torch.isfinite(target)
                if valid_mask.sum() == 0:
                    continue
                loss = F.binary_cross_entropy_with_logits(logits[valid_mask], target[valid_mask])
            else:
                raise ValueError(f"Unknown task_type: {task_type}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            metric = evaluate_model_on_loader(model, val_loader, data_module, task_type, device=device)
            if metric is not None and not math.isnan(metric) and metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


# =========================================================
# Main evaluator
# =========================================================

class GraphLevelEvaluator:
    """
    Graph-level evaluator for:
    - unsupervised evaluation via linear SVM on frozen embeddings
    - semi-supervised / transfer evaluation via fine-tuning a graph classifier

    This evaluator is graph-level only.
    It is not compatible with regular PyG PPI, which is node-level.
    """

    def __init__(
        self,
        model: nn.Module,
        data_module,
        mode: str,
        label_rate: float = 0.1,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        random_state: int = 42,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.data_module = data_module
        self.mode = mode
        self.label_rate = label_rate
        self.batch_size = batch_size if batch_size is not None else data_module.batch_size
        self.num_workers = num_workers if num_workers is not None else data_module.num_workers
        self.random_state = random_state
        self.device = get_device(device)

        dataset_name = getattr(data_module, "dataset_name", None)
        if dataset_name == "ppi":
            raise ValueError(
                "This eval.py is graph-level only, but regular PyG 'ppi' is a node-level task."
            )

    def evaluate_unsupervised(self) -> Dict[str, object]:
        """
        Train SimGRACE externally on the whole dataset, then:
        - extract graph embeddings
        - fit a downstream linear SVM
        - 10-fold cross-validation
        """
        dataset = self.data_module.get_entire_dataset()
        loader = build_single_view_loader(
            dataset=dataset,
            data_module=self.data_module,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        H, Y = extract_graph_embeddings(self.model, loader, self.data_module, device=self.device)
        task_type = infer_task_type_from_labels(Y)
        if task_type != "multiclass":
            raise ValueError("Unsupervised SVM protocol here is implemented for graph classification only.")

        folds = stratified_kfold_indices(Y, n_splits=10, seed=self.random_state)

        scores = []
        for k in range(10):
            test_idx = folds[k]
            train_idx = [i for j in range(10) if j != k for i in folds[j]]

            x_train = H[train_idx]
            y_train = Y[train_idx]
            x_test = H[test_idx]
            y_test = Y[test_idx]

            svm = train_linear_svm(
                x_train=x_train,
                y_train=y_train,
                lr=1e-2,
                weight_decay=1e-4,
                epochs=300,
                seed=self.random_state + k,
                device=self.device,
            )

            svm.eval()
            with torch.no_grad():
                logits = svm(x_test.to(self.device)).cpu()
            acc = multiclass_accuracy(logits, to_long_labels(y_test))
            scores.append(acc)

        scores_t = torch.tensor(scores)
        return {
            "metric": "accuracy",
            "mean": scores_t.mean().item(),
            "std": scores_t.std(unbiased=False).item(),
            "fold_scores": scores,
        }

    def evaluate_semi_or_transfer(self, freeze_backbone: bool = False) -> Dict[str, object]:
        """
        Semi-supervised / transfer evaluation.

        Protocol:
        - if explicit split exists in the data module:
            fine-tune on partial training data, evaluate on val/test
        - otherwise:
            K-fold with K = 1 / label_rate
            one fold is labeled training set each time
        """
        if self.data_module.train_dataset is None:
            self.data_module.setup("fit")

        if self.data_module.val_dataset is not None or self.data_module.test_dataset is not None:
            return self._evaluate_explicit_split(freeze_backbone=freeze_backbone)
        return self._evaluate_implicit_split(freeze_backbone=freeze_backbone)

    def _evaluate_explicit_split(self, freeze_backbone: bool = False) -> Dict[str, object]:
        train_dataset = self.data_module.train_dataset
        val_dataset = self.data_module.val_dataset
        test_dataset = self.data_module.test_dataset

        # Build a single-view loader on full train split to inspect labels
        full_train_loader = build_single_view_loader(
            dataset=train_dataset,
            data_module=self.data_module,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        _, Y_train = extract_graph_embeddings(self.model, full_train_loader, self.data_module, device=self.device)
        task_type = infer_task_type_from_labels(Y_train)

        # Sample labeled subset from train split for fine-tuning
        if task_type == "multiclass":
            labeled_indices = stratified_subset_indices(Y_train, fraction=self.label_rate, seed=self.random_state)
        else:
            num_train = len(train_dataset)
            generator = torch.Generator().manual_seed(self.random_state)
            perm = torch.randperm(num_train, generator=generator)
            num_labeled = max(1, int(num_train * self.label_rate))
            labeled_indices = perm[:num_labeled].tolist()

        labeled_train_dataset = Subset(train_dataset, labeled_indices)

        train_loader = build_single_view_loader(
            dataset=labeled_train_dataset,
            data_module=self.data_module,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = build_single_view_loader(
                dataset=val_dataset,
                data_module=self.data_module,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        test_loader = None
        if test_dataset is not None:
            test_loader = build_single_view_loader(
                dataset=test_dataset,
                data_module=self.data_module,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

        # Determine output dimension
        sample_batch = next(iter(full_train_loader))
        _, _, _, y_sample, _ = self.data_module.process_batch_for_model(sample_batch)

        if task_type == "multiclass":
            out_dim = int(torch.max(Y_train).item()) + 1
        elif task_type == "binary":
            out_dim = 1
        else:
            out_dim = y_sample.size(-1)

        clf = train_finetune_classifier(
            backbone=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            data_module=self.data_module,
            task_type=task_type,
            out_dim=out_dim,
            epochs=100,
            lr=1e-3,
            weight_decay=1e-5,
            freeze_backbone=freeze_backbone,
            device=self.device,
        )

        results = {"task_type": task_type}

        if val_loader is not None:
            results["val_metric"] = evaluate_model_on_loader(
                clf, val_loader, self.data_module, task_type, device=self.device
            )

        if test_loader is not None:
            results["test_metric"] = evaluate_model_on_loader(
                clf, test_loader, self.data_module, task_type, device=self.device
            )

        if task_type == "multiclass":
            results["metric_name"] = "accuracy"
        else:
            results["metric_name"] = "roc_auc"

        return results

    def _evaluate_implicit_split(self, freeze_backbone: bool = False) -> Dict[str, object]:
        dataset = self.data_module.get_entire_dataset()
        full_loader = build_single_view_loader(
            dataset=dataset,
            data_module=self.data_module,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        _, Y = extract_graph_embeddings(self.model, full_loader, self.data_module, device=self.device)
        task_type = infer_task_type_from_labels(Y)

        if task_type != "multiclass":
            raise ValueError(
                "Implicit K-fold protocol here is implemented for graph classification only."
            )

        k = max(2, int(round(1.0 / self.label_rate)))
        folds = stratified_kfold_indices(Y, n_splits=k, seed=self.random_state)

        scores = []

        for fold_id in range(k):
            labeled_idx = folds[fold_id]
            remaining_idx = [i for j in range(k) if j != fold_id for i in folds[j]]

            # create val/test from remaining
            test_idx, val_idx = stratified_train_val_split_from_indices(
                Y, remaining_idx, val_ratio=0.1, seed=self.random_state + fold_id
            )

            labeled_ds = Subset(dataset, labeled_idx)
            val_ds = Subset(dataset, val_idx)
            test_ds = Subset(dataset, test_idx)

            train_loader = build_single_view_loader(
                dataset=labeled_ds,
                data_module=self.data_module,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            val_loader = build_single_view_loader(
                dataset=val_ds,
                data_module=self.data_module,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            test_loader = build_single_view_loader(
                dataset=test_ds,
                data_module=self.data_module,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )

            out_dim = int(torch.max(Y).item()) + 1

            clf = train_finetune_classifier(
                backbone=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                data_module=self.data_module,
                task_type="multiclass",
                out_dim=out_dim,
                epochs=100,
                lr=1e-3,
                weight_decay=1e-5,
                freeze_backbone=freeze_backbone,
                device=self.device,
            )

            score = evaluate_model_on_loader(
                clf, test_loader, self.data_module, "multiclass", device=self.device
            )
            scores.append(score)

        scores_t = torch.tensor(scores)
        return {
            "metric_name": "accuracy",
            "mean": scores_t.mean().item(),
            "std": scores_t.std(unbiased=False).item(),
            "fold_scores": scores,
        }

    def evaluate(self, freeze_backbone: bool = False) -> Dict[str, object]:
        if self.mode == "unsupervised":
            return self.evaluate_unsupervised()
        if self.mode in {"semi_supervised", "transfer", "supervised"}:
            return self.evaluate_semi_or_transfer(freeze_backbone=freeze_backbone)
        raise ValueError(f"Unknown evaluation mode: {self.mode}")