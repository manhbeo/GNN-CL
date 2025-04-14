import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from datasets import get_dataset
from models import GINModule, ResGCN
from linear_classifier import GraphLinearClassifier
from graph_loss import SpectralGraphMatchingLoss, AdaptiveSpectralGraphMatchingLoss


def train(args):
    """
    Train a graph contrastive learning model.

    Args:
        args: Command line arguments
    """
    # Set up data module
    data_module = get_dataset(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Prepare data
    data_module.prepare_data()
    data_module.setup()

    # Get number of input features from dataset
    num_features = data_module.get_num_features()

    # Create model
    if args.model == "gin":
        model = GINModule(
            num_features=num_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dataset=args.dataset,
            has_graph_loss=args.use_graph_loss,
            batch_size_per_device=args.batch_size
        )
    elif args.model == "resgcn":
        model = ResGCN(
            in_channels=num_features,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            dropout_rate=args.dropout,
            learning_rate=args.lr
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Add graph loss if requested
    if args.use_graph_loss:
        if args.adaptive_graph_loss:
            model.graph_loss = AdaptiveSpectralGraphMatchingLoss(
                percentile=args.percentile,
                min_edges_percent=args.min_edges,
                max_edges_percent=args.max_edges,
                temperature=args.temperature,
                k_eigvals=args.k_eigvals,
                gather_distributed=(args.devices > 1)
            )
        else:
            model.graph_loss = SpectralGraphMatchingLoss(
                similarity_threshold=args.similarity_threshold,
                temperature=args.temperature,
                k_eigvals=args.k_eigvals,
                gather_distributed=(args.devices > 1)
            )

    # Set up logger
    logger_name = f"{args.model}-{args.dataset}-batch{args.batch_size}"
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
    trainer.fit(model, data_module)

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, f"{args.model}-{args.dataset}-final.ckpt")
    trainer.save_checkpoint(final_path)

    return final_path


def evaluate(args):
    """
    Evaluate a trained graph contrastive learning model.

    Args:
        args: Command line arguments
    """
    # Load pretrained model
    model = None
    if args.model == "gin":
        model = GINModule.load_from_checkpoint(args.checkpoint_path)
    elif args.model == "resgcn":
        model = ResGCN.load_from_checkpoint(args.checkpoint_path)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Set up data module
    data_module = get_dataset(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    # Prepare data
    data_module.prepare_data()
    data_module.setup()

    # Determine number of classes based on dataset
    if args.dataset in ["MUTAG", "IMDB-B", "RDT-B"]:
        num_classes = 2
    elif args.dataset == "NCI1":
        num_classes = 2
    elif args.dataset == "PROTEINS":
        num_classes = 2
    elif args.dataset == "DD":
        num_classes = 2
    elif args.dataset == "COLLAB":
        num_classes = 3
    elif args.dataset == "RDT-M5K":
        num_classes = 5
    else:
        raise ValueError(f"Unknown dataset for classification: {args.dataset}")

    # Create linear classifier
    linear_classifier = GraphLinearClassifier(
        model=model.encoder,  # Using only the encoder part of the model
        batch_size_per_device=args.batch_size,
        feature_dim=2048,  # Same as the encoder output dimension
        num_classes=num_classes,
        freeze_model=not args.finetune,
        lr=args.lr
    )

    # Set up logger
    logger_name = f"linear-{args.model}-{args.dataset}"
    if args.finetune:
        logger_name += "-finetune"
    else:
        logger_name += "-frozen"

    logger = WandbLogger(project="GraphLinearEval", name=logger_name)

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_top1",
        dirpath=args.checkpoint_dir,
        filename=f"linear-{args.model}-{args.dataset}-" + "{epoch:02d}-{val_top1:.4f}",
        save_top_k=3,
        mode="max"
    )

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=args.eval_epochs,
        devices=args.devices,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
        logger=logger,
        callbacks=[checkpoint_callback],
        precision=16 if args.mixed_precision else 32,
        deterministic=True
    )

    # Train and test linear classifier
    trainer.fit(linear_classifier, data_module)
    results = trainer.test(datamodule=data_module)

    return results


def main():
    parser = argparse.ArgumentParser(description="Graph Contrastive Learning")

    # General settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")

    # Dataset settings
    parser.add_argument("--dataset", type=str, default="MUTAG",
                        choices=["NCI1", "PROTEINS", "DD", "MUTAG", "COLLAB", "RDT-B", "RDT-M5K", "IMDB-B"],
                        help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")

    # Model settings
    parser.add_argument("--model", type=str, default="gin", choices=["gin", "resgcn"],
                        help="Model architecture")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")

    # Training settings
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
    parser.add_argument("--eval_epochs", type=int, default=100, help="Number of evaluation epochs")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune the model during evaluation")

    args = parser.parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set random seed
    seed_everything(args.seed)

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