import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGEUnsupervisedLoss(nn.Module):
    def __init__(
        self, 
        num_negative: int = 5
    ):
        """
        Unsupervised GraphSAGE loss using negative sampling.

        Parameters:
        - num_negative (int): number of negative samples per positive pair
        """
        super().__init__()
        self.num_negative = num_negative

    def forward(
        self,
        z_u: torch.Tensor,
        z_v_pos: torch.Tensor,
        z_v_neg: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the unsupervised loss for a batch of nodes.

        Parameters:
        - z_u: (B, D) anchor node embeddings
        - z_v_pos: (B, D) positive context node embeddings
        - z_v_neg: (B, num_negative, D) negative samples for each anchor

        Returns:
        - Scalar loss tensor
        """
        pos_score = torch.sum(z_u * z_v_pos, dim=1)  # (B,)
        pos_loss = F.logsigmoid(pos_score)           # (B,)

        neg_score = torch.bmm(z_v_neg, z_u.unsqueeze(2)).squeeze(2)  # (B, Q)
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)       # (B,)

        return -torch.mean(pos_loss + neg_loss)
