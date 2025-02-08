import torch
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree

from models import GIN, SimGRACE
from supconloss import Supervised_NTXentLoss

import wandb
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MUTAG',
                        choices=['NCI1', 'PROTEINS', 'DD', 'MUTAG', 'COLLAB', 'RDT-B', 'RDT-M5K', 'IMDB-B'])
    parser.add_argument('--mode', type=str, default='unsupervised',
                        choices=['unsupervised', 'semi_supervised'])
    parser.add_argument('--label_rate', type=float, default=0.1,
                        help='Label rate for semi-supervised learning (0.01 or 0.1)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--wandb_project', type=str, default='simgrace')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None)
    return parser.parse_args()


def init_wandb(args):
    """Initialize wandb with experiment configuration"""
    config = vars(args)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=config
    )


def train_epoch(model, loader, criterion, optimizer, device, prefix='train'):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        z, z_prime = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(z, z_prime, batch.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    wandb.log({f'{prefix}_loss': avg_loss})
    return avg_loss


def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            z, _ = model(batch.x, batch.edge_index, batch.batch)
            embeddings.append(z.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    return np.vstack(embeddings), np.concatenate(labels)


def unsupervised_evaluation(model, dataset, args, device, num_runs=5):
    """Unsupervised evaluation using SVM with 10-fold CV"""
    wandb.log({'mode': 'unsupervised'})
    accuracies = []

    loader = DataLoader(dataset, batch_size=args.batch_size)

    for run in range(num_runs):
        wandb.log({'run': run})
        embeddings, labels = get_embeddings(model, loader, device)

        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=run)
        fold_accuracies = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(embeddings, labels)):
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            svm = SVC(kernel='rbf')
            svm.fit(X_train, y_train)
            acc = svm.score(X_test, y_test)
            fold_accuracies.append(acc)

            wandb.log({
                'run': run,
                'fold': fold,
                'fold_accuracy': acc
            })

        run_acc = np.mean(fold_accuracies)
        accuracies.append(run_acc)
        wandb.log({
            'run': run,
            'run_accuracy': run_acc
        })

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    wandb.log({
        'final_accuracy': mean_acc,
        'accuracy_std': std_acc
    })

    return mean_acc, std_acc


def semi_supervised_evaluation(model, dataset, args, device, num_runs=5):
    """Semi-supervised evaluation with label rate"""
    wandb.log({
        'mode': 'semi_supervised',
        'label_rate': args.label_rate
    })
    accuracies = []
    k = int(1 / args.label_rate)

    for run in range(num_runs):
        wandb.log({'run': run})
        model.reset_parameters()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = Supervised_NTXentLoss(temperature=args.temperature)

        # Pre-training
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        for epoch in range(100):
            loss = train_epoch(model, loader, criterion, optimizer, device, prefix='pretrain')
            wandb.log({
                'run': run,
                'pretrain_epoch': epoch,
                'pretrain_loss': loss
            })

        # Finetune and evaluate
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=run)
        labels = [data.y.item() for data in dataset]
        fold_accuracies = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
            train_dataset = dataset[train_idx]
            test_dataset = dataset[test_idx]

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

            # Finetune
            for epoch in range(50):
                loss = train_epoch(model, train_loader, criterion, optimizer, device, prefix='finetune')
                wandb.log({
                    'run': run,
                    'fold': fold,
                    'finetune_epoch': epoch,
                    'finetune_loss': loss
                })

            # Evaluate
            embeddings, labels = get_embeddings(model, test_loader, device)
            pred = embeddings.argmax(axis=1)
            acc = accuracy_score(labels, pred)
            fold_accuracies.append(acc)

            wandb.log({
                'run': run,
                'fold': fold,
                'fold_accuracy': acc
            })

        run_acc = np.mean(fold_accuracies)
        accuracies.append(run_acc)
        wandb.log({
            'run': run,
            'run_accuracy': run_acc
        })

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    wandb.log({
        'final_accuracy': mean_acc,
        'accuracy_std': std_acc
    })

    return mean_acc, std_acc


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize wandb
    init_wandb(args)

    # Log system info
    wandb.log({
        'device': device.type,
        'dataset': args.dataset,
        'mode': args.mode
    })

    # Load dataset
    transform = OneHotDegree(max_degree=10)
    dataset = TUDataset(root='data', name=args.dataset, transform=transform)

    # Log dataset info
    wandb.log({
        'num_graphs': len(dataset),
        'num_features': dataset[0].x.size(1),
        'num_classes': len(torch.unique(dataset[0].y))
    })

    # Initialize models
    encoder = GIN(num_features=dataset[0].x.size(1),
                  hidden_dim=args.hidden_dim,
                  num_layers=args.num_layers)

    model = SimGRACE(encoder=encoder,
                     proj_hidden_dim=args.hidden_dim,
                     eta=args.eta,
                     sigma=args.sigma).to(device)

    # Log model architecture
    wandb.watch(model)

    # Perform evaluation based on mode
    if args.mode == 'unsupervised':
        mean_acc, std_acc = unsupervised_evaluation(model, dataset, args, device)
        print(f'Unsupervised Evaluation Results:')
        print(f'Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')
    else:
        mean_acc, std_acc = semi_supervised_evaluation(model, dataset, args, device)
        print(f'Semi-supervised Evaluation Results (Label Rate: {args.label_rate}):')
        print(f'Accuracy: {mean_acc:.4f} ± {std_acc:.4f}')

    wandb.finish()


if __name__ == '__main__':
    main()