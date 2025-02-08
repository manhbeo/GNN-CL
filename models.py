import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool

class GIN(nn.Module):
    def __init__(self, num_features, hidden_dim=32, num_layers=3):
        """
        GIN architecture used in SimGRACE paper
        Args:
            num_features: Number of input features
            hidden_dim: Hidden dimension size (default: 32 as per paper)
            num_layers: Number of GIN layers (default: 3 as per paper)
        """
        super(GIN, self).__init__()

        # Initialize layers list
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        initial_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(initial_mlp, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Remaining layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the network
        """
        h = x
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            h = conv(h, edge_index)
            h = batch_norm(h)
            h = F.relu(h)
        h = global_add_pool(h, batch)
        return h


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        """
        MLP projection head g(.) as described in the paper
        """
        super(ProjectionHead, self).__init__()
        self.g = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.g(x)


class SimGRACE(nn.Module):
    def __init__(self, encoder, proj_hidden_dim=32, eta=0.1, sigma=0.1):
        """
        SimGRACE model with parameter perturbation and stopped gradient for perturbed path
        Args:
            encoder: GIN encoder instance
            proj_hidden_dim: Hidden dimension for projection head
            eta: Coefficient that scales perturbation magnitude
            sigma: Standard deviation for Gaussian noise
        """
        super(SimGRACE, self).__init__()
        self.encoder = encoder
        self.projection = ProjectionHead(proj_hidden_dim, proj_hidden_dim)
        self.eta = eta
        self.sigma = sigma

    def get_perturbed_parameters(self):
        """
        Creates perturbed parameters dictionary without gradients
        """
        perturbed_params = {}
        with torch.no_grad():  # No gradients for perturbation generation
            for name, param in self.encoder.named_parameters():
                noise = torch.randn_like(param) * self.sigma
                perturbed_param = param + self.eta * noise
                perturbed_params[name] = perturbed_param.detach()  # Ensure no gradient tracking
        return perturbed_params

    def forward(self, x, edge_index, batch):
        """
        Forward pass with gradient stopped for perturbed encoder
        """
        # Original encoder path (with gradients)
        h = self.encoder(x, edge_index, batch)
        z = self.projection(h)

        # Store original parameters
        orig_params = {name: param.clone() for name, param in self.encoder.named_parameters()}

        # Get perturbed parameters (no gradients)
        perturbed_params = self.get_perturbed_parameters()

        # Apply perturbed parameters and compute representation without gradients
        with torch.no_grad():
            # Temporarily replace parameters with perturbed versions
            for name, param in self.encoder.named_parameters():
                param.data = perturbed_params[name]

            # Forward pass with perturbed parameters (no gradients)
            h_prime = self.encoder(x, edge_index, batch)
            z_prime = self.projection(h_prime)

            # Restore original parameters
            for name, param in self.encoder.named_parameters():
                param.data = orig_params[name]

        return z, z_prime.detach()  # Ensure z_prime has no gradients