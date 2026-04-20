import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# The previous version of the loss had a bug where the graph was built separately on each rank, leading to inconsistent graphs and incorrect loss values.
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
    def __init__(
        self,
        use_adaptive_threshold: bool = False,
        similarity_threshold: float = 0.5,
        temperature: float = 0.1,
        percentile: float = 90,
        min_edges_percent: float = 10,
        max_edges_percent: float = 50,
    ):
        super().__init__()
        self.use_adaptive_threshold = use_adaptive_threshold
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.percentile = percentile
        self.min_edges_percent = min_edges_percent / 100.0
        self.max_edges_percent = max_edges_percent / 100.0

    def _adaptive_threshold(self, S: torch.Tensor) -> torch.Tensor:
        """
        Adaptively determine threshold from similarity matrix S.
        """
        n = S.size(0)
        if n < 2:
            return S.new_tensor(self.similarity_threshold)

        indices = torch.triu_indices(n, n, offset=1, device=S.device)
        sim_values = S[indices[0], indices[1]]

        threshold = torch.quantile(sim_values, self.percentile / 100.0)

        # density over off-diagonal entries only
        mask = (S > threshold).float()
        mask.fill_diagonal_(0.0)
        edge_density = mask.sum() / (n * (n - 1))

        if edge_density < self.min_edges_percent:
            threshold = torch.quantile(sim_values, 1.0 - self.min_edges_percent)
        elif edge_density > self.max_edges_percent:
            threshold = torch.quantile(sim_values, 1.0 - self.max_edges_percent)

        return threshold

    def _compute_laplacian(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute random-walk Laplacian:
            L_rw = I - D^{-1} A
        """
        S = torch.matmul(Z, Z.T) / self.temperature

        if self.use_adaptive_threshold:
            threshold = self._adaptive_threshold(S)   
            A = (S > threshold).float()
        else:
            A = (S > self.similarity_threshold).float()

        A.fill_diagonal_(0.0)

        D = A.sum(dim=1) + 1e-10
        P = A / D.unsqueeze(1)   # D^{-1} A
        L = torch.eye(A.size(0), device=A.device, dtype=A.dtype) - P
        return L

    def forward(self, Z_i: torch.Tensor, Z_j: torch.Tensor) -> torch.Tensor:
        """
        Correct distributed version:
        build the graph on the global batch across all ranks.
        """
        Z_i = F.normalize(Z_i, dim=1)
        Z_j = F.normalize(Z_j, dim=1)

        # Gather embeddings across all ranks so the graph is global
        Z_i_all = gather_with_grad(Z_i)
        Z_j_all = gather_with_grad(Z_j)

        L1 = self._compute_laplacian(Z_i_all)
        L2 = self._compute_laplacian(Z_j_all)

        loss = ((L1 - L2) ** 2).sum()
        return loss