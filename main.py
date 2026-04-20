import argparse
import os
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from supconloss import Supervised_NTXentLoss
from data_module import GraphDataModule, SemiSupervisedDataModule, TransferDataModule
from eval import GraphLevelEvaluator
from graph_loss import SpectralGraphMatchingLoss
from models import build_model


class SpecMatchCLTrainer(pl.LightningModule):
    """
    Lightning wrapper for SpecMatchCL.

    Train batch format from data_module.py:
        ((x1, ei1, b1, y1, ea1), (x2, ei2, b2, y2, ea2))

    Validation/test batch format:
        (x, ei, b, y, ea)
    """

    def __init__(
        self,
        task: str,
        projection_dim: int = 128,
        dropout: float = 0.0,
        pooling: Optional[str] = None,
        chemistry_mode: bool = False,
        use_edge_attr: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.2,
        use_specmatch: bool = False,
        specmatch_weight: float = 1.0,
        similarity_threshold: float = 0.5,
        adaptive_threshold: bool = False,
        percentile: float = 90.0,
        min_edges_percent: float = 10.0,
        max_edges_percent: float = 50.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_model(
            task=task,
            projection_dim=projection_dim,
            dropout=dropout,
            pooling=pooling,
            chemistry_mode=chemistry_mode,
            use_edge_attr=use_edge_attr,
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.ntxent = Supervised_NTXentLoss(temperature=temperature)

        self.use_specmatch = use_specmatch
        self.specmatch_weight = specmatch_weight
        self.specmatch = None
        if use_specmatch:
            self.specmatch = SpectralGraphMatchingLoss(
                use_adaptive_threshold=adaptive_threshold,
                similarity_threshold=similarity_threshold,
                temperature=temperature,
                percentile=percentile,
                min_edges_percent=min_edges_percent,
                max_edges_percent=max_edges_percent,
            )

    def forward_once(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        batch_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_once(x, edge_index, batch_index, edge_attr)

    @staticmethod
    def _unpack_view(view: Tuple[torch.Tensor, ...]):
        x, edge_index, batch_index, y, edge_attr = view
        return x, edge_index, batch_index, y, edge_attr

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        view1, view2 = batch
        x1, ei1, b1, _, ea1 = self._unpack_view(view1)
        x2, ei2, b2, _, ea2 = self._unpack_view(view2)

        h1, z1 = self.forward_once(x1, ei1, b1, ea1)
        h2, z2 = self.forward_once(x2, ei2, b2, ea2)

        loss_cl = self.ntxent(z1, z2)
        loss = loss_cl

        self.log("train/loss_cl", loss_cl, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.use_specmatch:
            loss_spec = self.specmatch(h1, h2)
            loss = loss + self.specmatch_weight * loss_spec
            self.log("train/loss_specmatch", loss_spec, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, edge_index, batch_index, _, edge_attr = batch
        h, z = self.forward_once(x, edge_index, batch_index, edge_attr)

        self.log("val/h_norm", h.norm(dim=-1).mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/z_norm", z.norm(dim=-1).mean(), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def build_data_module(args):
    if args.mode == "unsupervised":
        return GraphDataModule(
            dataset_name=args.dataset,
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            mode="unsupervised",
            random_state=args.seed,
        )

    if args.mode == "semi_supervised":
        return SemiSupervisedDataModule(
            dataset_name=args.dataset,
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            label_rate=args.label_rate,
            random_state=args.seed,
        )

    if args.mode == "transfer":
        return TransferDataModule(
            pretrain_dataset="zinc-2m",
            finetune_dataset=args.dataset,
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            random_state=args.seed,
        )

    raise ValueError(f"Unsupported mode: {args.mode}")


def build_lightning_module(args) -> SpecMatchCLTrainer:
    if args.mode == "unsupervised":
        task = "unsupervised"
        chemistry_mode = False
        use_edge_attr = False
        pooling = "add"

    elif args.mode == "semi_supervised":
        task = "semi_supervised"
        chemistry_mode = False
        use_edge_attr = False
        pooling = "add"

    elif args.mode == "transfer":
        task = "transfer"
        chemistry_mode = True
        use_edge_attr = True
        pooling = "mean"

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    return SpecMatchCLTrainer(
        task=task,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
        pooling=pooling,
        chemistry_mode=chemistry_mode,
        use_edge_attr=use_edge_attr,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        use_specmatch=args.use_specmatch,
        specmatch_weight=args.specmatch_weight,
        similarity_threshold=args.similarity_threshold,
        adaptive_threshold=args.adaptive_threshold,
        percentile=args.percentile,
        min_edges_percent=args.min_edges_percent,
        max_edges_percent=args.max_edges_percent,
    )


def build_logger(args) -> WandbLogger:
    name_parts = ["SpecMatchCL", args.mode, args.dataset]
    if args.mode == "semi_supervised":
        name_parts.append(f"label{args.label_rate}")
    if args.use_specmatch:
        name_parts.append("with-specmatch")
    else:
        name_parts.append("no-specmatch")

    run_name = "-".join(name_parts)

    return WandbLogger(
        project=args.wandb_project,
        name=run_name,
        save_dir=args.wandb_dir,
        log_model=False,
    )


def build_trainer(args, logger) -> pl.Trainer:
    accelerator = "gpu" if torch.cuda.is_available() and args.devices > 0 else "cpu"
    devices = args.devices if accelerator == "gpu" else 1
    strategy = "ddp" if accelerator == "gpu" and devices > 1 else "auto"

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="{epoch:03d}-{train_loss:.4f}",
        monitor="train/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=32,
        deterministic=True,
        log_every_n_steps=50,
    )


def train(args) -> str:
    seed_everything(args.seed, workers=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    data_module = build_data_module(args)
    model = build_lightning_module(args)
    logger = build_logger(args)
    trainer = build_trainer(args, logger)

    if args.mode == "transfer":
        data_module.prepare_data()
        data_module.setup("fit")
        trainer.fit(
            model,
            train_dataloaders=data_module.pretrain_dataloader(),
            val_dataloaders=data_module.pretrain_val_dataloader(),
        )
    else:
        trainer.fit(model, datamodule=data_module)

    ckpt_path = trainer.checkpoint_callback.best_model_path
    if not ckpt_path:
        ckpt_path = os.path.join(args.checkpoint_dir, "last.ckpt")
        trainer.save_checkpoint(ckpt_path)

    print(f"Best checkpoint: {ckpt_path}")
    return ckpt_path


def run_unsupervised_evaluation(model: nn.Module, data_module) -> Dict[str, object]:
    evaluator = GraphLevelEvaluator(
        model=model,
        data_module=data_module,
        mode="unsupervised",
    )
    return evaluator.evaluate()


def run_semi_supervised_evaluation(
    model: nn.Module,
    data_module,
    label_rate: float = 0.1,
    freeze_backbone: bool = False,
) -> Dict[str, object]:
    evaluator = GraphLevelEvaluator(
        model=model,
        data_module=data_module,
        mode="semi_supervised",
        label_rate=label_rate,
    )
    return evaluator.evaluate(freeze_backbone=freeze_backbone)


def run_transfer_evaluation(
    model: nn.Module,
    data_module,
    label_rate: float = 0.1,
    freeze_backbone: bool = False,
) -> Dict[str, object]:
    evaluator = GraphLevelEvaluator(
        model=model,
        data_module=data_module,
        mode="transfer",
        label_rate=label_rate,
    )
    return evaluator.evaluate(freeze_backbone=freeze_backbone)


def evaluate(args, checkpoint_path: str):
    seed_everything(args.seed, workers=True)

    data_module = build_data_module(args)
    model = build_lightning_module(args)

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=True)

    if args.mode == "unsupervised":
        results = run_unsupervised_evaluation(model.model, data_module)

    elif args.mode == "semi_supervised":
        results = run_semi_supervised_evaluation(
            model.model,
            data_module,
            label_rate=args.label_rate,
            freeze_backbone=False,
        )

    elif args.mode == "transfer":
        results = run_transfer_evaluation(
            model.model,
            data_module.finetune_dm,
            label_rate=args.label_rate,
            freeze_backbone=False,
        )

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    print("Evaluation results:")
    for k, v in results.items():
        print(f"{k}: {v}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="SpecMatchCL")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--devices", type=int, default=1)

    parser.add_argument("--wandb_project", type=str, default="SpecMatchCL")
    parser.add_argument("--wandb_dir", type=str, default="wandb_logs")

    parser.add_argument(
        "--mode",
        type=str,
        default="unsupervised",
        choices=["unsupervised", "semi_supervised", "transfer"],
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--label_rate", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.2)

    parser.add_argument("--use_specmatch", action="store_true")
    parser.add_argument("--specmatch_weight", type=float, default=1.0)
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    parser.add_argument("--adaptive_threshold", action="store_true")
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--min_edges_percent", type=float, default=10.0)
    parser.add_argument("--max_edges_percent", type=float, default=50.0)

    parser.add_argument("--eval_after_train", action="store_false")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/last.ckpt")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "transfer":
        if args.dataset not in {"tox21", "toxcast", "sider", "clintox", "muv", "hiv", "bbbp", "bace"}:
            raise ValueError("Transfer downstream dataset must be one of the MoleculeNet graph-level datasets.")

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = train(args)

    if args.eval_after_train or args.checkpoint_path is not None:
        evaluate(args, checkpoint_path)


if __name__ == "__main__":
    main()