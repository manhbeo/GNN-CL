import torch.nn as nn
import torch
from lightly.utils import dist
import torch.nn.functional as F

class Supervised_NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5, gather_distributed: bool = False):
        '''
        Parameters:
        - temperature (float): Temperature for the ntx-ent loss.
        - gather_distributed (bool): Whether to gather data across multiple distributed processes for multi-GPU training.
        '''
        super(Supervised_NTXentLoss, self).__init__()
        self.temperature = temperature
        self.use_distributed = gather_distributed and dist.world_size() > 1

    def forward(self, out0: torch.Tensor, out1: torch.Tensor, labels: torch.Tensor):
        '''
        Parameters:
        - out0 (torch.Tensor): Embeddings from one view, shape (batch_size, embedding_dim).
        - out1 (torch.Tensor): Embeddings from another view, shape (batch_size, embedding_dim).
        - labels (torch.Tensor): Class labels corresponding to each embedding, shape (batch_size,).
                               Labels with value -1 are treated as negative samples.

        Returns:
        - torch.Tensor: The computed loss.
        '''
        out0 = F.normalize(out0, dim=1)
        out1 = F.normalize(out1, dim=1)

        if self.use_distributed:
            out0_large = torch.cat(dist.gather(out0), 0)
            out1_large = torch.cat(dist.gather(out1), 0)
            labels_large = torch.cat(dist.gather(labels), 0)
        else:
            out0_large = out0
            out1_large = out1
            labels_large = labels

        # Compute the cosine similarity matrix
        logits = torch.matmul(out0_large, out1_large.T) / self.temperature
        logits_exp = torch.exp(logits)

        # Create positive mask only for samples with same valid labels
        positive_mask = (labels_large.unsqueeze(1) == labels_large.unsqueeze(0)) & \
                       (labels_large.unsqueeze(1) != -1) & \
                       (labels_large.unsqueeze(0) != -1)

        # Add self-pairs for samples with label -1
        identity_mask = torch.eye(logits.size(0), device=logits.device).bool()
        positive_mask = positive_mask | identity_mask

        positive_logits = logits_exp * positive_mask
        all_logits_sum = logits_exp.sum(dim=1, keepdim=True) + 1e-8

        # Compute loss only for samples that have positive pairs
        positive_sum = positive_logits.sum(dim=1)
        valid_positive_mask = positive_sum > 0
        sup_contrastive_loss = -torch.log(positive_sum[valid_positive_mask] / all_logits_sum[valid_positive_mask].squeeze()).mean()

        return sup_contrastive_loss
