from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_add_pool, global_mean_pool
from lightly.models.modules.heads import SimCLRProjectionHead


def _ensure_x(x: Optional[torch.Tensor], edge_index: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
    """Create a dummy node feature tensor when a dataset has no node features."""
    if x is not None:
        return x
    if batch is not None:
        num_nodes = batch.size(0)
        device = batch.device
    else:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 1
        device = edge_index.device
    return torch.ones((num_nodes, 1), dtype=torch.float, device=device)


class GraphCLProjectionHead(nn.Module):
    """2-layer SimCLR-style MLP head.

    Lightly exposes a standard SimCLR projection head as a reusable module.
    """

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, output_dim: int = 128):
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        self.head = SimCLRProjectionHead(input_dim, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CategoricalNodeEncoder(nn.Module):
    """Encoder for chemistry-style categorical node features.

    Assumes two columns by default:
      - atom type
      - chirality
    This matches the user-provided chem_loader.py representation.
    """

    def __init__(
        self,
        emb_dim: int,
        num_atom_type: int = 120,
        num_chirality_tag: int = 4,
    ):
        super().__init__()
        self.atom_emb = nn.Embedding(num_atom_type, emb_dim)
        self.chirality_emb = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.atom_emb.weight)
        nn.init.xavier_uniform_(self.chirality_emb.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if x.size(-1) < 2:
            raise ValueError("Categorical chemistry node features are expected to have at least 2 columns.")
        x = x.long()
        return self.atom_emb(x[:, 0]) + self.chirality_emb(x[:, 1])


class CategoricalEdgeEncoder(nn.Module):
    """Encoder for chemistry-style categorical edge features.

    Assumes two columns by default:
      - bond type
      - bond direction
    """

    def __init__(
        self,
        emb_dim: int,
        num_bond_type: int = 6,
        num_bond_direction: int = 3,
    ):
        super().__init__()
        self.bond_emb = nn.Embedding(num_bond_type, emb_dim)
        self.dir_emb = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.bond_emb.weight)
        nn.init.xavier_uniform_(self.dir_emb.weight)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.size(-1) < 2:
            raise ValueError("Categorical chemistry edge features are expected to have at least 2 columns.")
        edge_attr = edge_attr.long()
        return self.bond_emb(edge_attr[:, 0]) + self.dir_emb(edge_attr[:, 1])


class FlexibleInputEncoder(nn.Module):
    """Handles either categorical chemistry features or dense features.

    - chemistry-like integer features with shape [N, 2] use embedding tables
    - dense / float features use a linear projection
    - if x is None, a 1D all-ones feature is created and projected
    """

    def __init__(self, out_dim: int, chemistry_mode: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.chemistry_mode = chemistry_mode
        self.dense_proj: Optional[nn.Module] = None
        self.cat_encoder: Optional[nn.Module] = None

    def forward(self, x: Optional[torch.Tensor], edge_index: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        x = _ensure_x(x, edge_index, batch)
        if self.chemistry_mode:
            if self.cat_encoder is None:
                self.cat_encoder = CategoricalNodeEncoder(self.out_dim)
                self.add_module("cat_encoder_impl", self.cat_encoder)
            return self.cat_encoder(x)

        # auto-detect chemistry-like integer features if not explicitly set
        if torch.is_floating_point(x):
            if self.dense_proj is None:
                self.dense_proj = nn.LazyLinear(self.out_dim)
                self.add_module("dense_proj_impl", self.dense_proj)
            return self.dense_proj(x.float())

        if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            if x.dim() == 2 and x.size(-1) == 2:
                if self.cat_encoder is None:
                    self.cat_encoder = CategoricalNodeEncoder(self.out_dim)
                    self.add_module("cat_encoder_impl", self.cat_encoder)
                return self.cat_encoder(x)
            if self.dense_proj is None:
                self.dense_proj = nn.LazyLinear(self.out_dim)
                self.add_module("dense_proj_impl", self.dense_proj)
            return self.dense_proj(x.float())

        if self.dense_proj is None:
            self.dense_proj = nn.LazyLinear(self.out_dim)
            self.add_module("dense_proj_impl", self.dense_proj)
        return self.dense_proj(x.float())


class FlexibleEdgeEncoder(nn.Module):
    """Encodes edge attributes when the backbone uses them.

    - chemistry-like integer edge_attr with shape [E, 2] use embeddings
    - dense / float edge_attr use a linear projection
    - if edge_attr is None, returns None
    """

    def __init__(self, out_dim: int, chemistry_mode: bool = False):
        super().__init__()
        self.out_dim = out_dim
        self.chemistry_mode = chemistry_mode
        self.dense_proj: Optional[nn.Module] = None
        self.cat_encoder: Optional[nn.Module] = None

    def forward(self, edge_attr: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if edge_attr is None:
            return None
        if self.chemistry_mode:
            if self.cat_encoder is None:
                self.cat_encoder = CategoricalEdgeEncoder(self.out_dim)
                self.add_module("cat_edge_encoder_impl", self.cat_encoder)
            return self.cat_encoder(edge_attr)

        if torch.is_floating_point(edge_attr):
            if self.dense_proj is None:
                self.dense_proj = nn.LazyLinear(self.out_dim)
                self.add_module("dense_edge_proj_impl", self.dense_proj)
            return self.dense_proj(edge_attr.float())

        if edge_attr.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            if edge_attr.dim() == 2 and edge_attr.size(-1) == 2:
                if self.cat_encoder is None:
                    self.cat_encoder = CategoricalEdgeEncoder(self.out_dim)
                    self.add_module("cat_edge_encoder_impl", self.cat_encoder)
                return self.cat_encoder(edge_attr)
            if self.dense_proj is None:
                self.dense_proj = nn.LazyLinear(self.out_dim)
                self.add_module("dense_edge_proj_impl", self.dense_proj)
            return self.dense_proj(edge_attr.float())

        if self.dense_proj is None:
            self.dense_proj = nn.LazyLinear(self.out_dim)
            self.add_module("dense_edge_proj_impl", self.dense_proj)
        return self.dense_proj(edge_attr.float())


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GINEncoder(nn.Module):
    """GIN encoder for graph-level representations.

    Used for:
      - unsupervised TU setting: 3 layers, 32 hidden dim
      - transfer setting: 5 layers, 300 hidden dim
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        pooling: Literal["add", "mean"] = "add",
        chemistry_mode: bool = False,
        use_edge_attr: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.use_edge_attr = use_edge_attr

        self.input_encoder = FlexibleInputEncoder(hidden_dim, chemistry_mode=chemistry_mode)
        self.edge_encoder = FlexibleEdgeEncoder(hidden_dim, chemistry_mode=chemistry_mode) if use_edge_attr else None

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP(hidden_dim, hidden_dim)
            if use_edge_attr:
                from torch_geometric.nn import GINEConv
                conv = GINEConv(mlp, edge_dim=hidden_dim)
            else:
                conv = GINConv(mlp)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.out_dim = hidden_dim

    def forward(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_encoder(x, edge_index, batch)
        edge_emb = self.edge_encoder(edge_attr) if self.edge_encoder is not None else None

        for layer, bn in zip(self.convs, self.batch_norms):
            if self.use_edge_attr:
                x = layer(x, edge_index, edge_emb)
            else:
                x = layer(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        return global_add_pool(x, batch)


class ResGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv = GCNConv(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x, edge_index)
        out = self.bn(out)
        out = F.relu(out + identity)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class ResGCNEncoder(nn.Module):
    """ResGCN-style graph encoder for semi-supervised setting.

    The semi-supervised GraphCL experiments use ResGCN with 5 layers and 128
    hidden dimensions per the user's specified setting.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 5,
        dropout: float = 0.0,
        pooling: Literal["add", "mean"] = "add",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling

        self.input_encoder = FlexibleInputEncoder(hidden_dim, chemistry_mode=False)
        self.input_conv = GCNConv(hidden_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)

        self.res_layers = nn.ModuleList(
            [ResGCNLayer(hidden_dim, dropout=dropout) for _ in range(num_layers - 1)]
        )

        self.out_dim = hidden_dim

    def forward(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del edge_attr  # ResGCN variant here does not use edge_attr
        x = self.input_encoder(x, edge_index, batch)
        x = self.input_conv(x, edge_index)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.res_layers:
            x = layer(x, edge_index)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        return global_add_pool(x, batch)


class GraphCLModel(nn.Module):
    """Backbone + projector only.

    This file intentionally contains no loss or training logic. It is designed
    to work with a two-view data pipeline: you can call `forward_once` on each
    view separately, or `forward_pair` on two views.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        projector_hidden_dim: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = GraphCLProjectionHead(
            input_dim=encoder.out_dim,
            hidden_dim=projector_hidden_dim or encoder.out_dim,
            output_dim=projection_dim,
        )
        self.normalize = normalize

    def encode(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(x, edge_index, batch, edge_attr)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        z = self.projector(h)
        if self.normalize:
            z = F.normalize(z, dim=-1)
        return z

    def forward_once(
        self,
        x: Optional[torch.Tensor],
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x, edge_index, batch, edge_attr)
        z = self.project(h)
        return h, z

    def forward(self, *args, **kwargs):
        return self.forward_once(*args, **kwargs)

    def forward_pair(self, view1, view2):
        h1, z1 = self.forward_once(*view1)
        h2, z2 = self.forward_once(*view2)
        return (h1, z1), (h2, z2)


def build_unsupervised_model(
    projection_dim: int = 128,
    dropout: float = 0.0,
    pooling: Literal["add", "mean"] = "add",
) -> GraphCLModel:
    encoder = GINEncoder(
        hidden_dim=32,
        num_layers=3,
        dropout=dropout,
        pooling=pooling,
        chemistry_mode=False,
        use_edge_attr=False,
    )
    return GraphCLModel(encoder=encoder, projection_dim=projection_dim)


def build_semi_supervised_model(
    projection_dim: int = 128,
    dropout: float = 0.0,
    pooling: Literal["add", "mean"] = "add",
) -> GraphCLModel:
    encoder = ResGCNEncoder(
        hidden_dim=128,
        num_layers=5,
        dropout=dropout,
        pooling=pooling,
    )
    return GraphCLModel(encoder=encoder, projection_dim=projection_dim)


def build_transfer_model(
    projection_dim: int = 128,
    dropout: float = 0.0,
    pooling: Literal["add", "mean"] = "mean",
    chemistry_mode: bool = False,
    use_edge_attr: bool = True,
) -> GraphCLModel:
    """Transfer encoder following the common GraphCL / pretrain-gnns default style.

    Uses a 5-layer GIN-style encoder with 300 hidden dimensions.
    Set `chemistry_mode=True` for molecule tasks so node/edge categorical
    features are embedded in the chemistry-style format used by GraphCL and
    pretrain-gnns.
    """
    encoder = GINEncoder(
        hidden_dim=300,
        num_layers=5,
        dropout=dropout,
        pooling=pooling,
        chemistry_mode=chemistry_mode,
        use_edge_attr=use_edge_attr,
    )
    return GraphCLModel(encoder=encoder, projection_dim=projection_dim)


def build_model(
    task: Literal["unsupervised", "semi_supervised", "transfer"],
    projection_dim: int = 128,
    dropout: float = 0.0,
    pooling: Optional[Literal["add", "mean"]] = None,
    chemistry_mode: bool = False,
    use_edge_attr: bool = True,
) -> GraphCLModel:
    task = task.lower()
    if task == "unsupervised":
        return build_unsupervised_model(
            projection_dim=projection_dim,
            dropout=dropout,
            pooling=pooling or "add",
        )
    if task == "semi_supervised":
        return build_semi_supervised_model(
            projection_dim=projection_dim,
            dropout=dropout,
            pooling=pooling or "add",
        )
    if task == "transfer":
        return build_transfer_model(
            projection_dim=projection_dim,
            dropout=dropout,
            pooling=pooling or "mean",
            chemistry_mode=chemistry_mode,
            use_edge_attr=use_edge_attr,
        )
    raise ValueError(f"Unknown task: {task}")
