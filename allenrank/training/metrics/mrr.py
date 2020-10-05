from typing import Optional, List

import numpy as np
import torch
from sklearn.metrics import label_ranking_average_precision_score

from allennlp.training.metrics.metric import Metric
from allenrank.training.metrics.ranking_metric import RankingMetric

import torchsnooper

@Metric.register("sk-mrr")
class SKLearnMRR(RankingMetric):
    """
    Sanity check for MRR. Should be moved to unit test.
    """
    def get_metric(self, reset: bool = False):
        num_labels, num_labels_check = self.predictions.size(-1), self.gold_labels.size(-1)
        assert num_labels == num_labels_check
        y_pred, y_true = map(lambda x: x.masked_select(self.masks).view(-1, num_labels).numpy(), (self.predictions, self.gold_labels))
        
        # https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision
        # If there is exactly one relevant label per sample, label ranking average precision is equivalent to the mean reciprocal rank.
        assert (y_true.sum(axis=-1).astype(int) == 1).all(), "Only compatible with one relevant document per instance"
        # raise ValueError(np.unique(y_true))
        score = label_ranking_average_precision_score(y_true.astype(int), y_pred, )

        if reset:
            self.reset()
        return score

@Metric.register("mrr")
class MRR(RankingMetric):
    def get_metric(self, reset: bool = False):
        score = mrr(self.predictions, self.gold_labels, self.masks).item()

        if reset:
            self.reset()
        return score
    
    
# https://stackoverflow.com/a/60202801/6766123
def first_nonzero(t):
    t = t.masked_fill(t != 0, 1)
    idx = torch.arange(t.size(-1), 0, -1).type_as(t)
    indices = torch.argmax(t * idx, 1, keepdim=True)
    return indices


def mrr(y_pred, y_true, mask):
    # Set the largest value in the ground truth to 1, and everything else to 0.
    # If `y_true = [0.65, 0.4, 0.1]`, then `binarized_y_true=[1,0,0]`.
    binarized_y_true = y_true.ge(y_true.max(dim=-1, keepdim=True).values).long()
    y_pred = y_pred.masked_fill(~mask, -1)

    # Sort the predictions to get the predicted ranking.
    _, rank = y_pred.sort(descending=True, dim=-1)

    # Sort the ground truth labels by their predicted rank.
    # If `y_pred=[0.25, 0.55, 0.05]` and `binarized_y_true=[1, 0, 0]`,
    # then `rank=[1,0,2]` and `ordered_truth=`[0,1,0]`.
    ordered_truth = binarized_y_true.gather(-1, rank)

    # Ordered indices: [1, 2, ..., batch_size]
    indices = torch.arange(1, binarized_y_true.size(-1) + 1).expand_as(y_pred).type_as(y_pred)

    # Calculate the reciprocal rank for each position, 0ing-out masked values.
    # Following above example, `_mrr = [0/1, 1/2 ,0/3] = [0, 0.5, 0]`.
    _mrr = (ordered_truth / indices) * mask

    # Get the mrr at the first non-zero position for each instance in the batch, and take the mean.
    # Following above example, `first_nonzero(ordered_truth) = 1`, so we grab `mrr[1] = torch.tensor(0.5)`.
    # The example only has one instance, but this works for batched input also.
    return _mrr.gather(-1, first_nonzero(ordered_truth)).mean()