import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.utils import dist


class Supervised_NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels: torch.Tensor):
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        use_distributed = dist.world_size() > 1

        if use_distributed:
            out0_large = torch.cat(dist.gather(out0), dim=0)
            out1_large = torch.cat(dist.gather(out1), dim=0)
            labels_large = torch.cat(dist.gather(labels), dim=0)
        else:
            out0_large = out0
            out1_large = out1
            labels_large = labels

        logits = torch.matmul(out0_large, out1_large.T) / self.temperature

        positive_mask = (
            (labels_large[:, None] == labels_large[None, :])
            & (labels_large[:, None] != -1)
            & (labels_large[None, :] != -1)
        )

        identity_mask = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        positive_mask = positive_mask | identity_mask

        log_denom = torch.logsumexp(logits, dim=1)

        # log(sum(exp(logits_pos)))
        logits_pos = logits.masked_fill(~positive_mask, float("-inf"))
        log_num = torch.logsumexp(logits_pos, dim=1)

        valid_mask = torch.isfinite(log_num)
        loss = -(log_num[valid_mask] - log_denom[valid_mask]).mean()

        return loss