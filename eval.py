import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from models import SpectralResGCN, SpectralGIN
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class SpectralEvaluator(pl.LightningModule):
    def __init__(self,
                 model,
                 dataset,
                 batch_size=32,
                 num_workers=4,
                 eval_mode='unsupervised',  # 'unsupervised' or 'semi_supervised'
                 label_rate=0.1,  # Directly use this value
                 random_state=42):
        super(SpectralEvaluator, self).__init__()

        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.eval_mode = eval_mode
        self.label_rate = label_rate
        self.random_state = random_state

        # Check if dataset has explicit train/val/test split
        self.has_explicit_split = hasattr(dataset, 'train_mask') and hasattr(dataset, 'val_mask') and hasattr(dataset,
                                                                                                              'test_mask')

    def setup(self, stage=None):
        # Prepare data splits based on evaluation protocol
        if self.has_explicit_split:
            # Use explicit splits if available
            self.train_idx = torch.where(self.dataset.train_mask)[0].tolist()
            self.val_idx = torch.where(self.dataset.val_mask)[0].tolist()
            self.test_idx = torch.where(self.dataset.test_mask)[0].tolist()
        else:
            # Create splits based on label rate
            if self.eval_mode == 'semi_supervised':
                # For semi-supervised: K-fold where K = 1/label_rate
                k = max(2, int(1 / self.label_rate))
                self.k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
            else:
                # For unsupervised: 10-fold cross-validation
                self.k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)

    def extract_embeddings(self, dataloader):
        """Extract graph embeddings from the model"""
        self.model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting embeddings"):
                # Unpack the batch
                x, edge_index, y = batch
                x, edge_index, y = x.to(self.device), edge_index.to(self.device), y.to(self.device)

                # Forward pass through the encoder (without perturbation)
                z = self.model.forward(x, edge_index, perturb=False)

                # Store embeddings and labels
                embeddings.append(z.cpu())
                labels.append(y.cpu())

        # Concatenate all embeddings and labels
        embeddings = torch.cat(embeddings, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()

        return embeddings, labels

    def calculate_top3_accuracy(self, y_true, y_pred_proba):
        """
        Calculate Top-3 accuracy

        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities for each class

        Returns:
            Top-3 accuracy score
        """
        # Get top 3 predictions for each sample
        top3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:]

        # Check if true label is in top 3 predictions
        correct = 0
        for i, true_label in enumerate(y_true):
            if true_label in top3_indices[i]:
                correct += 1

        return correct / len(y_true)

    def evaluate_unsupervised(self):
        """Evaluate in unsupervised setting with 10-fold CV using SVM"""
        # Train the model on the whole dataset first
        trainer = pl.Trainer(max_epochs=100, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        trainer.fit(self.model, train_loader)

        # Extract embeddings
        embeddings, labels = self.extract_embeddings(train_loader)

        # 10-fold CV with SVM
        cv_scores_top3 = []

        for train_idx, test_idx in self.k_fold.split(embeddings, labels):
            # Get unique class labels to determine number of classes
            num_classes = len(np.unique(labels))

            # Train SVM with probability estimates
            svm = SVC(kernel='linear', probability=True)
            svm.fit(embeddings[train_idx], labels[train_idx])

            # Get probability predictions
            y_pred_proba = svm.predict_proba(embeddings[test_idx])

            # Calculate top-3 accuracy
            top3_acc = self.calculate_top3_accuracy(labels[test_idx], y_pred_proba)
            cv_scores_top3.append(top3_acc)

        # Return mean and standard deviation of CV scores
        return {
            'mean_top3_accuracy': np.mean(cv_scores_top3),
            'std_top3_accuracy': np.std(cv_scores_top3),
            'cv_scores_top3': cv_scores_top3
        }

    def evaluate_semi_supervised(self):
        """Evaluate in semi-supervised setting"""
        if self.has_explicit_split:
            return self._evaluate_with_explicit_split()
        else:
            return self._evaluate_with_implicit_split()

    def _evaluate_with_explicit_split(self):
        """Evaluation for datasets with explicit train/val/test split"""
        # Create dataloaders for each split
        train_dataset = [self.dataset[i] for i in self.train_idx]
        val_dataset = [self.dataset[i] for i in self.val_idx]
        test_dataset = [self.dataset[i] for i in self.test_idx]

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Pre-train with Spectral
        pretrain_trainer = pl.Trainer(max_epochs=100, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
        pretrain_trainer.fit(self.model, train_loader)

        # Extract embeddings
        train_embeddings, train_labels = self.extract_embeddings(train_loader)
        val_embeddings, val_labels = self.extract_embeddings(val_loader)
        test_embeddings, test_labels = self.extract_embeddings(test_loader)

        # Finetune on partial training data
        subset_size = int(len(train_dataset) * self.label_rate)
        subset_indices = np.random.choice(len(train_embeddings), subset_size, replace=False)

        # Train a classifier using the embeddings
        num_classes = len(np.unique(train_labels))
        svm = SVC(kernel='linear', probability=True)
        svm.fit(train_embeddings[subset_indices], train_labels[subset_indices])

        # Evaluate on validation and test sets
        val_proba = svm.predict_proba(val_embeddings)
        test_proba = svm.predict_proba(test_embeddings)

        # Calculate top-3 accuracy
        val_top3_acc = self.calculate_top3_accuracy(val_labels, val_proba)
        test_top3_acc = self.calculate_top3_accuracy(test_labels, test_proba)

        return {
            'validation_top3_accuracy': val_top3_acc,
            'test_top3_accuracy': test_top3_acc
        }

    def _evaluate_with_implicit_split(self):
        """Evaluation for datasets without explicit split using K-fold CV"""
        # Create a dataloader for the whole dataset
        full_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Pre-train with Spectral on all data
        pretrain_trainer = pl.Trainer(max_epochs=100, accelerator='gpu' if torch.cuda.is_available() else 'cpu')
        pretrain_trainer.fit(self.model, full_loader)

        # Extract embeddings
        embeddings, labels = self.extract_embeddings(full_loader)

        # K-fold cross-validation (K = 1 / label_rate)
        cv_scores_top3 = []

        # Calculate K from the provided label_rate
        k = max(2, int(1 / self.label_rate))
        k_fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in k_fold.split(embeddings, labels):
            # Split train into finetune and validation sets
            finetune_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=self.random_state)

            # Train a classifier using the embeddings
            num_classes = len(np.unique(labels))
            svm = SVC(kernel='linear', probability=True)
            svm.fit(embeddings[finetune_idx], labels[finetune_idx])

            # Evaluate on validation and test sets
            val_proba = svm.predict_proba(embeddings[val_idx])
            test_proba = svm.predict_proba(embeddings[test_idx])

            # Calculate top-3 accuracy
            val_top3_acc = self.calculate_top3_accuracy(labels[val_idx], val_proba)
            test_top3_acc = self.calculate_top3_accuracy(labels[test_idx], test_proba)

            # Store test accuracy
            cv_scores_top3.append(test_top3_acc)

        # Return mean and standard deviation of CV scores
        return {
            'mean_top3_accuracy': np.mean(cv_scores_top3),
            'std_top3_accuracy': np.std(cv_scores_top3),
            'cv_scores_top3': cv_scores_top3
        }

    def evaluate(self):
        """Main evaluation function"""
        if self.eval_mode == 'unsupervised':
            return self.evaluate_unsupervised()
        else:  # 'semi_supervised'
            return self.evaluate_semi_supervised()


# Example usage
def run_evaluation(model_type, dataset, label_rate=0.1):
    """
    Run both unsupervised and semi-supervised evaluation

    Args:
        model_type: 'GIN' or 'ResGCN'
        dataset: PyTorch Geometric dataset
        label_rate: Portion of labeled data for semi-supervised learning

    Returns:
        Dictionary with evaluation results
    """
    # Initialize the model based on model_type
    if model_type == 'GIN':
        model = SpectralGIN(
            num_features=dataset.num_features,
            hidden_dim=32,
            num_layers=3
        )
    else:  # 'ResGCN'
        model = SpectralResGCN(
            num_features=dataset.num_features,
            hidden_channels=128,
            num_layers=5
        )

    # Unsupervised evaluation
    unsupervised_evaluator = SpectralEvaluator(
        model=model,
        dataset=dataset,
        eval_mode='unsupervised'
    )
    unsupervised_results = unsupervised_evaluator.evaluate()

    # Re-initialize the model for semi-supervised evaluation
    if model_type == 'GIN':
        model = SpectralGIN(
            num_features=dataset.num_features,
            hidden_dim=32,
            num_layers=3
        )
    else:  # 'ResGCN'
        model = SpectralResGCN(
            num_features=dataset.num_features,
            hidden_channels=128,
            num_layers=5
        )

    # Semi-supervised evaluation
    semi_supervised_evaluator = SpectralEvaluator(
        model=model,
        dataset=dataset,
        eval_mode='semi_supervised',
        label_rate=label_rate
    )
    semi_supervised_results = semi_supervised_evaluator.evaluate()

    return {
        'unsupervised': unsupervised_results,
        'semi_supervised': semi_supervised_results
    }