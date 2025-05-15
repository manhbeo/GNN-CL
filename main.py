import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from datasets import GraphDataModule, TransferDataModule, SemiSupervisedDataModule
from models import SpectralGIN, SpectralResGCN
from eval import SpectralEvaluator
from graph_loss import SpectralGraphMatchingLoss, AdaptiveSpectralGraphMatchingLoss


def train(args):
    """
    Train a graph contrastive learning model.

    Args:
        args: Command line arguments
    """
    # Set up data module
    if args.mode == "transfer":
        data_module = TransferDataModule(
            pretrain_dataset=args.pretrain_dataset,
            finetune_dataset=args.dataset,
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            random_state=args.seed
        )
        # For transfer learning, use the pretrain dataset
        num_features = None  # Will be determined in setup
    elif args.mode == "semi_supervised":
        data_module = SemiSupervisedDataModule(
            dataset_name=args.dataset,
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label_rate=args.label_rate,
            random_state=args.seed
        )
        num_features = None  # Will be determined in setup
    else:  # unsupervised
        data_module = GraphDataModule(
            dataset_name=args.dataset,
            root=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            mode="unsupervised",
            random_state=args.seed
        )
        num_features = None  # Will be determined in setup

    # Prepare data
    data_module.prepare_data()
    data_module.setup()

    # Determine the number of input features from the dataset
    if args.mode == "transfer":
        # For transfer learning, use the pretrain dataset
        data_module.pretrain_dm.setup("fit")
        train_loader = data_module.pretrain_dataloader()
        batch = next(iter(train_loader))
        x, edge_index, _ = data_module.pretrain_dm.process_batch_for_model(batch)
        num_features = x.size(1)
    else:
        # For unsupervised or semi-supervised, use the main dataset
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        x, edge_index, _ = data_module.process_batch_for_model(batch)
        num_features = x.size(1)

    # Create model
    if args.model == "gin":
        model = SpectralGIN(
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eta=args.eta,
            sigma=args.sigma,
            dataset=args.dataset,
            has_graph_loss=args.use_graph_loss,
            batch_size_per_device=args.batch_size
        )
    elif args.model == "resgcn":
        model = SpectralResGCN(
            num_features=num_features,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            dropout_rate=args.dropout,
            learning_rate=args.lr,
            eta=args.eta,
            sigma=args.sigma,
            weight_decay=args.weight_decay,
            has_graph_loss=args.use_graph_loss,
            batch_size_per_device=args.batch_size
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Add graph loss if requested
    if args.use_graph_loss:
        if args.adaptive_graph_loss:
            graph_loss = AdaptiveSpectralGraphMatchingLoss(
                percentile=args.percentile,
                min_edges_percent=args.min_edges,
                max_edges_percent=args.max_edges,
                temperature=args.temperature,
                k_eigvals=args.k_eigvals,
                gather_distributed=(args.devices > 1)
            )
        else:
            graph_loss = SpectralGraphMatchingLoss(
                similarity_threshold=args.similarity_threshold,
                temperature=args.temperature,
                k_eigvals=args.k_eigvals,
                gather_distributed=(args.devices > 1)
            )
        model.graph_loss = graph_loss

    # Set up logger
    logger_name = f"{args.model}-{args.dataset}-{args.mode}"
    if args.mode == "transfer":
        logger_name += f"-from-{args.pretrain_dataset}"
    elif args.mode == "semi_supervised":
        logger_name += f"-{args.label_rate * 100}pct"
    logger_name += f"-batch{args.batch_size}"
    if args.use_graph_loss:
        logger_name += "-GraphLoss"

    logger = WandbLogger(project="GraphContrastive", name=logger_name)

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.checkpoint_dir,
        filename=f"{args.model}-{args.dataset}-" + "{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min"
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp" if args.devices > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16 if args.mixed_precision else 32,
        deterministic=True
    )

    # Train model
    if args.mode == "transfer":
        data_module.setup("pretrain")
        trainer.fit(model, data_module.pretrain_dataloader(), data_module.pretrain_val_dataloader())
    else:
        trainer.fit(model, data_module)

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, f"{args.model}-{args.dataset}-{args.mode}-final.ckpt")
    trainer.save_checkpoint(final_path)

    return final_path


def evaluate(args):
    """
    Evaluate a trained graph contrastive learning model.

    Args:
        args: Command line arguments
    """
    # Set up data module
    data_module = GraphDataModule(
        dataset_name=args.dataset,
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        mode=args.eval_mode,  # "unsupervised" or "semi_supervised"
        label_rate=args.label_rate,
        random_state=args.seed
    )

    # Prepare data
    data_module.prepare_data()
    data_module.setup()

    # Load pretrained model
    if args.model == "gin":
        model = SpectralGIN.load_from_checkpoint(args.checkpoint_path)
    elif args.model == "resgcn":
        model = SpectralResGCN.load_from_checkpoint(args.checkpoint_path)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Create evaluator
    evaluator = SpectralEvaluator(
        model=model,
        dataset=data_module.get_entire_dataset(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        eval_mode=args.eval_mode,
        label_rate=args.label_rate,
        random_state=args.seed
    )

    # Set up logger
    logger_name = f"eval-{args.model}-{args.dataset}-{args.eval_mode}"
    if args.eval_mode == "semi_supervised":
        logger_name += f"-{args.label_rate * 100}pct"

    logger = WandbLogger(project="GraphEvaluation", name=logger_name)

    # Set up trainer for evaluation
    trainer = pl.Trainer(
        devices=1,  # Use single device for evaluation
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        deterministic=True
    )

    # Setup evaluator
    evaluator.setup()

    # Run evaluation
    results = evaluator.evaluate()

    logger.log_metrics(results)

    return results


def main():
    parser = argparse.ArgumentParser(description="Graph Contrastive Learning with Spectral Perturbation")

    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")

    # Dataset settings
    parser.add_argument("--dataset", type=str, default="MUTAG",
                        choices=["NCI1", "PROTEINS", "DD", "MUTAG", "COLLAB", "RDT-B", "RDT-M5K", "IMDB-B",
                                 "Tox21", "ToxCast", "Sider", "ClinTox", "MUV", "HIV", "BBBP", "Bace",
                                 "PPI", "ZINC", "PPI-306K", "ZINC-2M"],
                        help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")

    # Model settings
    parser.add_argument("--model", type=str, default="gin", choices=["gin", "resgcn"],
                        help="Model architecture")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--eta", type=float, default=0.1, help="Perturbation coefficient eta")
    parser.add_argument("--sigma", type=float, default=0.1, help="Gaussian noise variance sigma")

    # Training settings
    parser.add_argument("--mode", type=str, default="unsupervised",
                        choices=["unsupervised", "transfer", "semi_supervised"],
                        help="Training mode")
    parser.add_argument("--pretrain_dataset", type=str, default=None,
                        choices=["PPI-306K", "ZINC-2M"],
                        help="Dataset for pretraining (transfer learning mode)")
    parser.add_argument("--label_rate", type=float, default=0.1,
                        help="Portion of labeled data for semi-supervised learning")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--use_graph_loss", action="store_true", help="Use graph matching loss")
    parser.add_argument("--adaptive_graph_loss", action="store_true", help="Use adaptive graph matching loss")

    # Graph loss settings
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="Similarity threshold for graph construction")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for similarity calculation")
    parser.add_argument("--k_eigvals", type=int, default=10, help="Number of eigenvalues to use")
    parser.add_argument("--percentile", type=float, default=90,
                        help="Percentile of similarity values for adaptive threshold")
    parser.add_argument("--min_edges", type=float, default=10,
                        help="Minimum percentage of edges for adaptive threshold")
    parser.add_argument("--max_edges", type=float, default=50,
                        help="Maximum percentage of edges for adaptive threshold")

    # Evaluation settings
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--eval_mode", type=str, default="unsupervised",
                        choices=["unsupervised", "semi_supervised"],
                        help="Evaluation mode")

    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set random seed
    seed_everything(args.seed)

    # Check arguments consistency
    if args.mode == "transfer" and args.pretrain_dataset is None:
        raise ValueError("Pretrain dataset must be provided for transfer learning mode")

    if args.eval:
        if not args.checkpoint_path:
            raise ValueError("Checkpoint path must be provided for evaluation")
        results = evaluate(args)
        print("Evaluation results:", results)
    else:
        model_path = train(args)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()