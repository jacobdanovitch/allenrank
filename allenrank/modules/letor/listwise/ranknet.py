import itertools

import torch
from torch.nn import functional as F

from allenrank.modules.letor.letor_base import LearningToRankLayer
from allenrank.modules.letor.pairwise.bce import masked_binary_cross_entropy_with_logits

@LearningToRankLayer.register('ranknet')
class RankNetLETOR(LearningToRankLayer):
    # adapted from https://github.com/allegro/allRank/blob/master/allrank/models/losses/rankNet.py
    def forward(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor,
        label_mask: torch.Tensor = None
    ):  
        if label_mask is None:
            label_mask = torch.ones_like(labels)

        predictions = predictions.masked_fill(~label_mask, float('-inf'))
        labels = labels.masked_fill(~label_mask, float('-inf'))

        document_pairs_candidates = list(itertools.product(range(labels.size(1)), repeat=2))

        selected_pred = predictions[:, document_pairs_candidates]
        pairs_true = labels[:, document_pairs_candidates]

        true_diffs = pairs_true[..., 0] - pairs_true[..., 1]
        pred_diffs = selected_pred[..., 0] - selected_pred[..., 1]

        loss_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))
        true_diffs = (true_diffs > 0).float()

        return masked_binary_cross_entropy_with_logits(pred_diffs, true_diffs, loss_mask)