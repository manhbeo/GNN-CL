import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, models
from supconloss import Supervised_NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from LARs import LARS
from lightly.utils.scheduler import CosineWarmupScheduler
import copy


class SpectralGIN(pl.LightningModule):
    def __init__(self,
                 num_features,
                 hidden_dim=32,  # 32 as per Image 2 for unsupervised tasks
                 num_layers=3,  # 3 as per Image 2
                 dropout=0.5,
                 lr=0.01,
                 weight_decay=5e-4,
                 eta=0.1,  # Perturbation coefficient from equation (2)
                 sigma=0.1,  # Variance for Gaussian noise
                 dataset="PPI",
                 has_graph_loss=False,
                 batch_size_per_device=128):
        super(SpectralGIN, self).__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        self.has_graph_loss = has_graph_loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size_per_device = batch_size_per_device
        self.eta = eta
        self.sigma = sigma

        # Initialize feature bank for contrastive learning
        self.register_buffer('feature_bank', None)
        self.register_buffer('feature_labels', None)

        # Initialize GIN model from PyTorch Geometric
        self.encoder = models.GIN(
            in_channels=num_features,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            out_channels=2048,
            dropout=dropout,
            jk='last',  # Using only the last layer output
            learn_eps=False
        )
        self.projection_head = SimCLRProjectionHead(2048, 2048, 1024)

    def forward(self, x, edge_index, perturb=False):
        # Normal forward pass through the encoder
        if not perturb:
            h = self.encoder(x, edge_index)
            z = self.projection_head(h)
            return z

        # Create a perturbed version of the encoder (as per equation 2 in Image 1)
        else:
            # Create a deep copy of the encoder and its parameters
            perturbed_encoder = copy.deepcopy(self.encoder)

            # Apply perturbation to the weights
            with torch.no_grad():
                for name, param in perturbed_encoder.named_parameters():
                    # Δθ_l ~ N(0, σ²)
                    delta_theta = torch.randn_like(param) * self.sigma

                    # θ'_l = θ_l + η · Δθ_l
                    param.add_(self.eta * delta_theta)

            # Forward pass through the perturbed encoder
            h_perturbed = perturbed_encoder(x, edge_index)
            z_perturbed = self.projection_head(h_perturbed)

            return z_perturbed

    def _update_feature_bank(self, features, labels):
        # Initialize feature bank if needed
        if self.feature_bank is None:
            self.feature_bank = features.detach()
            self.feature_labels = labels.detach()
        else:
            # Concatenate new features and labels
            self.feature_bank = torch.cat([self.feature_bank, features.detach()], dim=0)
            self.feature_labels = torch.cat([self.feature_labels, labels.detach()], dim=0)

            # Limit size of feature bank (keep most recent 10000 samples)
            if self.feature_bank.size(0) > 10000:
                self.feature_bank = self.feature_bank[-10000:]
                self.feature_labels = self.feature_labels[-10000:]

    def training_step(self, batch, batch_idx):
        # Unpack batch - now just one view and labels
        x, edge_index, fine_label = batch

        # Get representations using original encoder
        z = self.forward(x, edge_index, perturb=False)

        # Get representations using perturbed encoder
        z_perturbed = self.forward(x, edge_index, perturb=True)

        # Calculate contrastive loss between original and perturbed views
        loss = Supervised_NTXentLoss(z, z_perturbed, fine_label)

        # Add graph-level loss if specified
        if self.has_graph_loss and hasattr(self, 'CLOPLoss'):
            loss += self.CLOPLoss(z, z_perturbed, None, fine_label, None, None)

        self.log('train_loss', loss, sync_dist=True)

        # Update feature bank
        self._update_feature_bank(z, fine_label)

        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        x, edge_index, fine_label = batch

        # Get original and perturbed representations
        z = self.forward(x, edge_index, perturb=False)
        z_perturbed = self.forward(x, edge_index, perturb=True)

        # Calculate loss
        loss = Supervised_NTXentLoss(z, z_perturbed, fine_label)

        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = LARS(
            self.parameters(),
            lr=0.3 * self.batch_size_per_device * self.trainer.world_size / 256 if self.lr is None else self.lr,
            weight_decay=self.weight_decay
        )

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]


class SpectralResGCN(pl.LightningModule):
    def __init__(self,
                 num_features,
                 hidden_channels=128,  # 128 as per Image 2 for semi-supervised tasks
                 num_layers=5,  # 5 as per Image 2
                 dropout_rate=0.5,
                 learning_rate=0.001,
                 eta=0.1,  # Perturbation coefficient from equation (2)
                 sigma=0.1,  # Variance for Gaussian noise
                 weight_decay=5e-4,
                 has_graph_loss=False,
                 batch_size_per_device=128):
        super(SpectralResGCN, self).__init__()

        # Save hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.eta = eta
        self.sigma = sigma
        self.has_graph_loss = has_graph_loss
        self.weight_decay = weight_decay
        self.batch_size_per_device = batch_size_per_device

        # Initialize feature bank for contrastive learning
        self.register_buffer('feature_bank', None)
        self.register_buffer('feature_labels', None)

        # Input layer
        self.input_layer = GCNConv(num_features, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        # Residual GCN layers
        self.res_layers = nn.ModuleList()
        for _ in range(num_layers - 2):  # -2 because we have separate input and output layers
            self.res_layers.append(ResGCNLayer(hidden_channels, hidden_channels))

        # Output layer
        self.output_layer = GCNConv(hidden_channels, 2048)

        # Projection head
        self.projection_head = SimCLRProjectionHead(2048, 2048, 1024)

    def forward(self, x, edge_index, perturb=False):
        if not perturb:
            # Standard forward pass
            h = self._encoder_forward(x, edge_index)
            z = self.projection_head(h)
            return z
        else:
            # Create perturbed copies of all model components
            perturbed_input_layer = copy.deepcopy(self.input_layer)
            perturbed_input_bn = copy.deepcopy(self.input_bn)
            perturbed_res_layers = copy.deepcopy(self.res_layers)
            perturbed_output_layer = copy.deepcopy(self.output_layer)

            # Apply perturbation to all parameters
            with torch.no_grad():
                # Perturb input layer
                for param in perturbed_input_layer.parameters():
                    delta_theta = torch.randn_like(param) * self.sigma
                    param.add_(self.eta * delta_theta)

                # Perturb batch norm
                for param in perturbed_input_bn.parameters():
                    delta_theta = torch.randn_like(param) * self.sigma
                    param.add_(self.eta * delta_theta)

                # Perturb residual layers
                for layer in perturbed_res_layers:
                    for param in layer.parameters():
                        delta_theta = torch.randn_like(param) * self.sigma
                        param.add_(self.eta * delta_theta)

                # Perturb output layer
                for param in perturbed_output_layer.parameters():
                    delta_theta = torch.randn_like(param) * self.sigma
                    param.add_(self.eta * delta_theta)

            # Forward pass through perturbed encoder
            # Input layer
            x_perturbed = perturbed_input_layer(x, edge_index)
            x_perturbed = perturbed_input_bn(x_perturbed)
            x_perturbed = F.relu(x_perturbed)
            x_perturbed = F.dropout(x_perturbed, p=self.dropout_rate, training=self.training)

            # Residual layers
            for res_layer in perturbed_res_layers:
                x_perturbed = res_layer(x_perturbed, edge_index)
                x_perturbed = F.dropout(x_perturbed, p=self.dropout_rate, training=self.training)

            # Output layer
            h_perturbed = perturbed_output_layer(x_perturbed, edge_index)
            z_perturbed = self.projection_head(h_perturbed)

            return z_perturbed

    def _encoder_forward(self, x, edge_index):
        # Input layer
        x = self.input_layer(x, edge_index)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Residual layers
        for res_layer in self.res_layers:
            x = res_layer(x, edge_index)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Output layer
        h = self.output_layer(x, edge_index)
        return h

    def _update_feature_bank(self, features, labels):
        # Initialize feature bank if needed
        if self.feature_bank is None:
            self.feature_bank = features.detach()
            self.feature_labels = labels.detach()
        else:
            # Concatenate new features and labels
            self.feature_bank = torch.cat([self.feature_bank, features.detach()], dim=0)
            self.feature_labels = torch.cat([self.feature_labels, labels.detach()], dim=0)

            # Limit size of feature bank (keep most recent 10000 samples)
            if self.feature_bank.size(0) > 10000:
                self.feature_bank = self.feature_bank[-10000:]
                self.feature_labels = self.feature_labels[-10000:]

    def training_step(self, batch, batch_idx):
        # Unpack batch - now just one view and labels
        x, edge_index, fine_label = batch

        # Get representations using original encoder
        z = self.forward(x, edge_index, perturb=False)

        # Get representations using perturbed encoder
        z_perturbed = self.forward(x, edge_index, perturb=True)

        # Calculate contrastive loss between original and perturbed views
        loss = Supervised_NTXentLoss(z, z_perturbed, fine_label)

        # Add graph-level loss if specified
        if self.has_graph_loss and hasattr(self, 'CLOPLoss'):
            loss += self.CLOPLoss(z, z_perturbed, None, fine_label, None, None)

        self.log('train_loss', loss, sync_dist=True)

        # Update feature bank
        self._update_feature_bank(z, fine_label)

        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        x, edge_index, fine_label = batch

        # Get original and perturbed representations
        z = self.forward(x, edge_index, perturb=False)
        z_perturbed = self.forward(x, edge_index, perturb=True)

        # Calculate loss
        loss = Supervised_NTXentLoss(z, z_perturbed, fine_label)

        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = LARS(
            self.parameters(),
            lr=0.3 * self.batch_size_per_device * self.trainer.world_size / 256 if self.learning_rate is None else self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=int(self.trainer.estimated_stepping_batches),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]


class ResGCNLayer(nn.Module):
    """
    Residual Graph Convolutional Network Layer
    """

    def __init__(self, in_channels, out_channels):
        super(ResGCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

        # If dimensions don't match, use a linear projection for the skip connection
        self.use_projection = in_channels != out_channels
        if self.use_projection:
            self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        identity = x

        # Apply GCN and batch normalization
        out = self.gcn(x, edge_index)
        out = self.bn(out)

        # Apply skip connection
        if self.use_projection:
            identity = self.projection(identity)

        # Add skip connection and apply ReLU
        out = out + identity
        out = F.relu(out)
        return out