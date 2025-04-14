from typing import Any, Dict, List, Tuple, Union
import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import Optimizer
from LARs import LARS
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.scheduler import CosineWarmupScheduler


class GraphLinearClassifier(LightningModule):
    def __init__(
            self,
            model: Module,
            batch_size_per_device: int,
            feature_dim: int = 2048,
            num_classes: int = 1000,
            topk: Tuple[int, ...] = (1, 5),
            freeze_model: bool = False,
            lr=None
    ) -> None:
        '''
        Linear classifier for benchmarking with graph neural networks.

        Adapted from the lightly library with modifications for graph data.

        Args:
            model:
                Graph model used for feature extraction. Must define a forward(x, edge_index) method
                that returns a feature tensor.
            batch_size_per_device:
                Batch size per device.
            feature_dim:
                Dimension of features returned by forward method of model.
            num_classes:
                Number of classes in the dataset.
            topk:
                Tuple of integers defining the top-k accuracy metrics to compute.
            freeze_model:
                If True, the model is frozen and only the classification head is
                trained. This corresponds to the linear eval setting. Set to False for
                finetuning.
        '''

        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.batch_size_per_device = batch_size_per_device
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.topk = topk
        self.freeze_model = freeze_model
        self.lr = lr

        self.classification_head = Linear(feature_dim, num_classes)
        self.criterion = CrossEntropyLoss()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.freeze_model:
            with torch.no_grad():
                features = self.model.forward(x, edge_index).flatten(start_dim=1)
        else:
            features = self.model.forward(x, edge_index).flatten(start_dim=1)
        output: Tensor = self.classification_head(features)
        return output

    def shared_step(
            self, batch: Tuple[Tuple[Tensor, Tensor], Tensor], batch_idx: int
    ) -> Tuple[Tensor, Dict[int, Tensor]]:
        # Adapt to graph data format: ((x, edge_index), targets)
        (x, edge_index), targets = batch
        predictions = self.forward(x, edge_index)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
        return loss, topk

    def training_step(self, batch: Tuple[Tuple[Tensor, Tensor], Tensor], batch_idx: int) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])  # targets
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        self.log(
            "train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
        )
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch: Tuple[Tuple[Tensor, Tensor], Tensor], batch_idx: int) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])  # targets
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def configure_optimizers(
            self,
    ) -> Tuple[List[Optimizer], List[Dict[str, Union[Any, str]]]]:
        parameters = list(self.classification_head.parameters())
        if not self.freeze_model:
            parameters += self.model.parameters()
        optimizer = LARS(parameters,
                         lr=0.3 * self.batch_size_per_device * self.trainer.world_size / 256 if self.lr is None else self.lr)
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    def on_train_epoch_start(self) -> None:
        if self.freeze_model:
            # Set model to eval mode to disable norm layer updates.
            self.model.eval()

    def test_step(self, batch: Tuple[Tuple[Tensor, Tensor], Tensor], batch_idx: int) -> Tensor:
        """Runs a single test step.

        Args:
            batch:
                A batch of graph data containing (x, edge_index) and corresponding targets.
            batch_idx:
                Index of the batch.

        Returns:
            loss: The loss calculated for the test batch.
        """
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])  # targets
        log_dict = {f"test_top{k}": acc for k, acc in topk.items()}
        self.log("test_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss