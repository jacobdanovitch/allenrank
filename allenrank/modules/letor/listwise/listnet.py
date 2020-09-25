import torch
from torch.nn import functional as F

from allennlp.nn import util
from allenrank.modules.letor.letor_base import LearningToRankLayer

@LearningToRankLayer.register('listnet')
class ListNetLETOR(LearningToRankLayer):
    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self._eps = eps

    # adapted from https://github.com/allegro/allRank/blob/master/allrank/models/losses/listNet.py
    def forward(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor,
        label_mask: torch.Tensor = None
    ):  
        if label_mask is None:
            label_mask = torch.ones_like(labels)

        preds_smax = util.masked_softmax(predictions, label_mask, dim=1)
        true_smax = util.masked_softmax(labels, label_mask, dim=1)

        preds_smax = preds_smax + self._eps
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))