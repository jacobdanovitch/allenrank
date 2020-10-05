from typing import List, Optional
import itertools

import numpy as np
import torch

from allennlp.training.metrics.metric import Metric

class RankingMetric(Metric):
    def __init__(
        self,
        padding_value: int = -1
    ) -> None:
        self._padding_value = padding_value
        self._num_labels = None
        self.reset()
        
    def __call__(
            self,
            predictions: torch.LongTensor,
            gold_labels: torch.LongTensor,
            mask: torch.LongTensor = None
        ):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of real-valued predictions of shape (batch_size, slate_length).
        gold_labels : ``torch.Tensor``, required.
            A tensor of real-valued labels of shape (batch_size, slate_length).
        """
        
        if mask is None:
            mask = torch.ones_like(gold_labels).bool()

        predictions, gold_labels, mask = self.detach_tensors(predictions, gold_labels, mask)

        num_labels, num_labels_check = predictions.size(-1), gold_labels.size(-1)
        assert num_labels == num_labels_check, f"Predictions have size {num_labels} but labels have size {num_labels_check}."

        # how is _this_ faster than pad_sequence by 5-6x???
        if (self._num_labels is not None) and (num_labels < self._num_labels):
            padded_preds = self._padding_value * self._padded_like(predictions)
            padded_labels = self._padding_value * self._padded_like(gold_labels)
            padded_mask = self._padded_like(mask)
            
            for i, (pred, label, msk) in enumerate(zip(predictions, gold_labels, mask)):
                padded_preds[i, :pred.size(-1)] = pred
                padded_labels[i, :label.size(-1)] = label
                padded_mask[i, :msk.size(-1)] = msk
            
            predictions = padded_preds
            gold_labels = padded_labels
            mask = padded_mask

        if self._num_labels is None:
            self._num_labels = num_labels

        self._all_predictions.append(predictions)
        self._all_gold_labels.append(gold_labels) 
        self._all_masks.append(mask)

    def _padded_like(self, t: torch.Tensor):
        return torch.ones(t.size(0), self._num_labels, device=t.device, dtype=t.dtype)

    def _concatenate_tensors(self, arr: List[torch.Tensor]):
        return torch.cat(arr, dim=0)
        # arr = list(itertools.chain(*arr))
        # return torch.nn.utils.rnn.pad_sequence(arr, batch_first=True, padding_value=self._padding_value)

    @property
    def predictions(self):
        return self._concatenate_tensors(self._all_predictions)
    
    @property
    def gold_labels(self):
        return self._concatenate_tensors(self._all_gold_labels)
    
    @property
    def masks(self):
        return self._concatenate_tensors(self._all_masks)
        
    def get_metric(self, reset: bool = False):
        raise NotImplementedError()
    
    def reset(self):
        self._all_predictions = []
        self._all_gold_labels = []
        self._all_masks = []