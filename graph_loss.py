import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class _GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all ranks with backward support.
    """

    @staticmethod
    def forward(ctx, x):
        if not dist.is_initialized():
            return (x,)

        world_size = dist.get_world_size()
        outputs = [torch.zeros_like(x) for _ in range(world_size)]
        dist.all_gather(outputs, x)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        if not dist.is_initialized():
            return grads[0]

        rank = dist.get_rank()
        grad_out = grads[rank].contiguous()
        dist.all_reduce(grad_out, op=dist.ReduceOp.SUM)
        return grad_out


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return x
    return torch.cat(_GatherLayer.apply(x), dim=0)


class SpectralGraphMatchingLoss(nn.Module):
    """
    Graph-level spectral matching loss using a soft adjacency matrix.

    Main idea:
    - build similarity matrix from graph embeddings
    - convert similarity to a soft adjacency matrix
    - compute random-walk Laplacians
    - minimize Laplacian disagreement between two views
    """

    def __init__(
        self,
        use_adaptive_threshold: bool = False,
        similarity_threshold: float = 0.5,
        temperature: float = 0.1,
        percentile: float = 90.0,
        min_edges_percent: float = 10.0,
        max_edges_percent: float = 50.0,
        sharpness: float = 10.0,
        normalize_loss: bool = True,
    ):
        super().__init__()
        self.use_adaptive_threshold = use_adaptive_threshold
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.percentile = percentile
        self.min_edges_percent = min_edges_percent / 100.0
        self.max_edges_percent = max_edges_percent / 100.0
        self.sharpness = sharpness
        self.normalize_loss = normalize_loss

    def _adaptive_threshold(self, S: torch.Tensor) -> torch.Tensor:
        """
        Determine a threshold from the similarity matrix S.
        """
        n = S.size(0)
        if n < 2:
            return S.new_tensor(self.similarity_threshold)

        indices = torch.triu_indices(n, n, offset=1, device=S.device)
        sim_values = S[indices[0], indices[1]]

        threshold = torch.quantile(sim_values, self.percentile / 100.0)

        # Estimate density using a hard mask only for threshold selection,
        # not for the final graph construction.
        mask = (S > threshold).float()
        mask.fill_diagonal_(0.0)
        edge_density = mask.sum() / (n * (n - 1))

        if edge_density < self.min_edges_percent:
            threshold = torch.quantile(sim_values, 1.0 - self.min_edges_percent)
        elif edge_density > self.max_edges_percent:
            threshold = torch.quantile(sim_values, 1.0 - self.max_edges_percent)

        return threshold

    def _soft_adjacency(self, S: torch.Tensor) -> torch.Tensor:
        """
        Build a soft adjacency matrix from similarities.

        A_ij = sigmoid(sharpness * (S_ij - threshold))
        """
        if self.use_adaptive_threshold:
            threshold = self._adaptive_threshold(S)
        else:
            threshold = S.new_tensor(self.similarity_threshold)

        A = torch.sigmoid(self.sharpness * (S - threshold))
        A.fill_diagonal_(0.0)
        return A

    def _compute_laplacian(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute the random-walk Laplacian:
            L_rw = I - D^{-1} A
        """
        S = torch.matmul(Z, Z.T) / self.temperature
        A = self._soft_adjacency(S)

        D = A.sum(dim=1) + 1e-10
        P = A / D.unsqueeze(1)
        L = torch.eye(A.size(0), device=A.device, dtype=A.dtype) - P
        return L

    def forward(self, Z_i: torch.Tensor, Z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral disagreement between two batches of graph embeddings.
        Supports distributed training by gathering graph embeddings across ranks.
        """
        Z_i = F.normalize(Z_i, dim=1)
        Z_j = F.normalize(Z_j, dim=1)

        Z_i_all = gather_with_grad(Z_i)
        Z_j_all = gather_with_grad(Z_j)

        L1 = self._compute_laplacian(Z_i_all)
        L2 = self._compute_laplacian(Z_j_all)

        loss = ((L1 - L2) ** 2).sum()

        if self.normalize_loss:
            n = L1.size(0)
            loss = loss / (n * n)

        return loss