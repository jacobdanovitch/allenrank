import torch
from torch.nn import functional as F

from allenrank.modules.letor.letor_base import LearningToRankLayer


@LearningToRankLayer.register('bce')
class BinaryClassificationLETOR(LearningToRankLayer):
    def forward(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor,
        label_mask: torch.Tensor = None
    ):
        if label_mask is None:
            return F.binary_cross_entropy_with_logits(predictions, labels)
        
        return masked_binary_cross_entropy_with_logits(predictions, labels, label_mask)

def masked_binary_cross_entropy_with_logits(predictions, labels, label_mask):
    loss = F.binary_cross_entropy_with_logits(predictions, labels, reduction='none')
    return loss.masked_fill(~label_mask, 0).sum() / label_mask.sum()
    