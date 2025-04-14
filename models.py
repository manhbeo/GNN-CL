from torch_geometric.nn import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from supconloss import Supervised_NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from LARs import LARS
from lightly.utils.scheduler import CosineWarmupScheduler


class GINModule(pl.LightningModule):
    def __init__(self,
                 num_features,
                 hidden_dim=64,
                 num_layers=3,
                 dropout=0.5,
                 lr=0.01,
                 weight_decay=5e-4,
                 dataset="PPI",
                 has_graph_loss=False,
                 batch_size_per_device = 128):
        super(GINModule, self).__init__()

        # Save hyperparameters
        self.has_graph_loss = has_graph_loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size_per_device = batch_size_per_device

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
        self.projection_head = SimCLRProjectionHead(output_dim=1024)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        z = self.projection_head(h)
        return z

    def shared_step(self, batch, label_por=None):
        (x_i, edge_index_i, x_j, edge_index_j), fine_label = batch
        z_i = self.forward(x_i, edge_index_i)
        z_j = self.forward(x_j, edge_index_j)

        loss = self.Supervised_NTXentLoss(z_i, z_j, fine_label)
        if self.has_graph_loss:
            loss += self.CLOPLoss(z_i, z_j, None, fine_label, label_por, None)
        return loss

    def training_step(self, batch, batch_idx):
        (x_i, edge_index_i, x_j, edge_index_j), fine_label = batch
        z_i = self.forward(x_i, edge_index_i)
        loss = self.shared_step(batch)
        self.log('train_loss', loss, sync_dist=True)
        self._update_feature_bank(z_i, fine_label)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 1.0)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = LARS(self.parameters(),
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

class ResGCN(pl.LightningModule):
    """
    Residual Graph Convolutional Network with 5 layers and 128 hidden dimensions
    """

    def __init__(self, in_channels, hidden_channels=128, num_layers=5, dropout_rate=0.5, learning_rate=0.001):
        super(ResGCN, self).__init__()

        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Input layer
        self.input_layer = GCNConv(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        # Residual GCN layers
        self.res_layers = nn.ModuleList()
        for _ in range(num_layers - 2):  # -2 because we have separate input and output layers
            self.res_layers.append(ResGCNLayer(hidden_channels, hidden_channels))

        # Output layer
        self.output_layer = GCNConv(hidden_channels, 2048)
        # check for compatability
        self.projection_head = SimCLRProjectionHead(output_dim=1024)

    def forward(self, x, edge_index):
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
        z = self.projection_head(h)
        return z

    def shared_step(self, batch, label_por=None):
        (x_i, edge_index_i, x_j, edge_index_j), fine_label = batch
        z_i = self.forward(x_i, edge_index_i)
        z_j = self.forward(x_j, edge_index_j)

        loss = self.Supervised_NTXentLoss(z_i, z_j, fine_label)
        if self.has_graph_loss:
            loss += self.CLOPLoss(z_i, z_j, None, fine_label, label_por, None)
        return loss

    def training_step(self, batch, batch_idx):
        (x_i, edge_index_i, x_j, edge_index_j), fine_label = batch
        z_i = self.forward(x_i, edge_index_i)
        loss = self.shared_step(batch)
        self.log('train_loss', loss, sync_dist=True)
        self._update_feature_bank(z_i, fine_label)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, 1.0)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = LARS(self.parameters(),
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



